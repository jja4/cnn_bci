# -*- coding: utf-8 -*-
# Authors: Seul-Ki Yeom, Chang-Hee Han, ............., Benjamin Blankertz, and Klaus-Robert Muller

import logging
import importlib
from scipy import io
import sys
import re
from braindecode.experiments.loggers import Printer
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.signalproc import lowpass_cnt, highpass_cnt, exponential_running_standardize
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.splitters import split_into_two_sets, split_into_train_test
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.optimizers import AdamW
import torch.nn.functional as F
import numpy as np
from torch import nn
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing
from braindecode.datautil.iterators import get_balanced_batches
from numpy.random import RandomState
import torch as th
from braindecode.experiments.monitors import compute_preds_per_trial_from_crops
import os
import glob


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#default_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG/30channel_0529/'  #62 channel datsets
default_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG/62channel_0711/'
#default_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG_hanyang/'
#default_path ='/home/joel/PycharmProjects/DeepBCI/data/Only_LR/' #88 and 84 channel datsets
save_path = '/home/joel/PycharmProjects/DeepBCI/code/Ind_BCImodel_noprep_original.pt'


filename = default_path+'*labels.mat'
subj_files = glob.glob(filename)

num_epochs = 30
#regex = re.compile(r'\d+')
#subjs = [int(x) for i in range(0,len(subj_files)) for x in regex.findall(subj_files[i])]
#subjs.sort()

#load list of subjects that have only left and right imagined movements
subjs_path = '/home/joel/PycharmProjects/DeepBCI/results/LR_subjs.tar'
subjs = th.load(subjs_path)
print(subjs)
num_folds = 5

Accuracies = np.zeros((len(subjs),num_folds))
Losses = np.zeros((len(subjs),num_epochs+1))

var_save_path = '/home/joel/PycharmProjects/DeepBCI/results/Ind_results_62chan_noprep_original.tar'  # type: str
#measures = th.load(var_save_path)
#Accuracies = measures['Acc']
#Losses = measures['Losses']

count=0
### Training loop #############################################################
for i in subjs[:]:
    print('CROSS VALIDATION on Subject No.{}'.format(i))
    if not i == 100:
        raw_data = io.loadmat(default_path + 'sub_' + str(i) + '.mat')
        labels = io.loadmat(default_path + 'sub_' + str(i) + '_labels.mat')
        FeatVect = raw_data['FeatVect']
        y_labels = labels['Triggers']

        # preprocessing
        for ii in range(0, FeatVect.shape[0]):
            # 1. Data reconstruction
            temp_data = FeatVect[ii, :, :]
            temp_data = temp_data.transpose()
            # 2. Lowpass filtering
            lowpassed_data = lowpass_cnt(temp_data, 100, 200, filt_order=3)
            # 3. Highpass filtering
            bandpassed_data = highpass_cnt(lowpassed_data, 4, 200, filt_order=3)
            # 4. Exponential running standardization
            ExpRunStand_data = exponential_running_standardize(bandpassed_data, factor_new=0.001, init_block_size=None,
                                                               eps=0.0001) #used to be bandpassed_data
            # 5. Renewal preprocessed data
            ExpRunStand_data = ExpRunStand_data.transpose()
            FeatVect[ii, :, :] = ExpRunStand_data
            #del temp_data, lowpassed_data, bandpassed_data, ExpRunStand_data

        print('preprocessed data')
        # 2nd phase: Convert data to Braindecode format
        # Convert data from volt to millivolt
        # Pytorch expects float32 for input and int64 for labels.
        # X: trials x channels x timesteps
        X = FeatVect.astype(np.float32)
        X = X[:, :, :]
        y = (y_labels[:,0] - 0).astype(np.int64)


        for CV in np.arange(0, num_folds):
            print('Subject No.{} CV {}'.format(i, CV))
            # 5th phase: Model evaluation (test)
            train_set = SignalAndTarget(X, y=y)
            # 80% training, 20% test
            train_set, test_set = split_into_train_test(train_set, n_folds = num_folds, i_test_fold = CV, rng=RandomState((2019, 28, 6))) #RandomState((2019, 28, 6))
            # 5% training, 95% test
            #test_set, train_set = split_into_train_test(train_set, n_folds = num_folds, i_test_fold = CV, rng=None)

            cuda = th.cuda.is_available()
            set_random_seeds(seed=20190628, cuda=cuda)

            n_classes = 2
            in_chans = train_set.X.shape[1]  # number of channels = 128
            input_time_length = 100  # length of time of each epoch/trial = 4000

            model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                    input_time_length=input_time_length,
                                    final_conv_length='auto', )
            #model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
            #                        input_time_length=input_time_length,
            #                        final_conv_length='auto', )


            if cuda:
                model.cuda()
            print('cuda: ' + str(cuda) + ', num chans: ' + str(in_chans))
            optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.1*0.001)  # weight_decay=0.1*0.001
            #model.network.conv_time.kernel_size = (in_chans, 30)
            #model.network.conv_spat.kernel_size = (30, in_chans-10)
            model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True)
            print('compiled')


            model.fit(train_set.X, train_set.y, epochs=num_epochs, batch_size=64, scheduler='cosine',
                      input_time_length=input_time_length)
            print(model.epochs_df)

            model.network.eval()
            print(model.predict(test_set.X))

            scores = model.evaluate(test_set.X, test_set.y)
            Accuracy = 1 - scores['misclass']
            print('Accuracy (%) :', Accuracy)

            # save key values
            Accuracies[count,CV] = Accuracy
            Losses[count, :] = model.epochs_df['train_loss']

            th.save({'Acc':Accuracies,'Losses': Losses}, var_save_path)
        print('Overall Acc Subject {}: {}'.format(i,np.mean(Accuracies[count])))
        count += 1

print('last_step')
