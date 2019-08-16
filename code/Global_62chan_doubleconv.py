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
#default_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG/30channel_0529/' #62 channel datsets
default_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG/62channel_0711/'
#default_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG_hanyang/'
#default_path ='/home/joel/PycharmProjects/DeepBCI/data/Only_LR/' #88 and 84 channel datasets
save_path = '/home/joel/PycharmProjects/DeepBCI/code/Global_BCImodel_doubleconv.pt'

filename = default_path+'*labels.mat'
subj_files = glob.glob(filename)

num_epochs = 30
#extract subject numbers from filenames
regex = re.compile(r'\d+')
subjs = [int(x) for i in range(0,len(subj_files)) for x in regex.findall(subj_files[i])]
subjs.sort()

#load list of subjects that only did left/right imagined movements
subjs_path = '/home/joel/PycharmProjects/DeepBCI/results/LR_subjs.tar'
subjs = th.load(subjs_path)

#save important numbers
num_folds = 5

Accuracies = np.zeros((len(subjs),num_folds))
noFT_Accuracies = np.zeros((len(subjs),num_folds))
Losses = np.zeros((len(subjs),num_epochs+1))
Train_Losses = np.zeros((len(subjs), num_epochs + 1))
FT_Losses = np.zeros((len(subjs), num_epochs + 1))

var_save_path = '/home/joel/PycharmProjects/DeepBCI/results/Global_results_62chan_doubleconv.tar'  # type: str

#measures = th.load(var_save_path)
#Accuracies = measures['Acc']
#Losses = measures['Losses']



# option to start training at a specific index
idx =25
### Training loop #############################################################
for test_subj in subjs[idx:]:
    print('CROSS VALIDATION on Subject No.{}'.format(test_subj))
    model_created = False
    sub_idx = 0

    # create model
    cuda = th.cuda.is_available()
    set_random_seeds(seed=20170629, cuda=cuda)

    n_classes = 2
    in_chans = 62  # train_set.X.shape[1]  # number of channels = 128
    input_time_length = 100  # length of time of each epoch/trial = 4000

    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=input_time_length,
                            final_conv_length='auto', )  # .create_network() # 'auto')

    #model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
    #                 input_time_length=input_time_length,
    #                 final_conv_length='auto', )
    if cuda:
        model.cuda()
    print('cuda: ' + str(cuda))

    for i in subjs:  # subject loop
        if not any([i == test_subj]):
            print('Test Subj {}, Train Subject No.{}'.format(test_subj,i))
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
                                                                   eps=0.0001)
                # 5. Renewal preprocessed data
                ExpRunStand_data = ExpRunStand_data.transpose()
                FeatVect[ii, :, :] = ExpRunStand_data
                del temp_data, lowpassed_data, bandpassed_data, ExpRunStand_data

            print('preprocessed data')
            # 2nd phase: Convert data to Braindecode format
            # Convert data from volt to millivolt
            # Pytorch expects float32 for input and int64 for labels.
            # X: trials x channels x timesteps
            X = FeatVect.astype(np.float32)
            X = X[:, :, :]
            y = (y_labels[:,0] - 0).astype(np.int64)

            #select only certain trials (left,right hand imagery)
           # indices = np.where((y == 0) | (y == 1))
           # X = np.take(X, indices[0], axis=0)
           # y = np.take(y, indices[0])

            # del FeatVect, y_labels, labels, raw_data

            train_set = SignalAndTarget(X, y=y)
            # train_set, valid_set, test_set = split_into_train_valid_test(train_set, 3, 2, rng=RandomState((2019,22,3)))
            # train_set, test_set = split_into_two_sets(train_set, first_set_fraction=0.8)

            print('data prepared')


            optimizer = AdamW(model.parameters(), lr= 0.01, weight_decay= 0.1*0.001)
            print('optimizer created')

            #if model_created:
            #    model.network.load_state_dict(th.load(save_path))
            #    model.network.train()
            model_created = True
            model.network.conv_time.kernel_size = (in_chans-10, 30)
            model.network.conv_spat.kernel_size = (30, in_chans-10)
            model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True)
            print('compiled')


            model.fit(train_set.X, train_set.y, epochs=num_epochs, batch_size=64, scheduler= 'cosine',
                    input_time_length=input_time_length)
            print(model.epochs_df)

            Train_Losses[sub_idx,:] = model.epochs_df['train_loss']

            #save model
            th.save(model.network.state_dict(), save_path)
            sub_idx += 1

    ## Test
    # Cross validation for test data

    raw_data = io.loadmat(default_path + 'sub_' + str(test_subj) + '.mat')
    labels = io.loadmat(default_path + 'sub_' + str(test_subj) + '_labels.mat')
    FeatVect = raw_data['FeatVect']
    y_labels = labels['Triggers']
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
                                                           eps=0.0001)
        # 5. Renewal preprocessed data
        ExpRunStand_data = ExpRunStand_data.transpose()
        FeatVect[ii, :, :] = ExpRunStand_data
        del temp_data, lowpassed_data, bandpassed_data, ExpRunStand_data

    X = FeatVect.astype(np.float32)
    X = X[:, :, :]
    y = (y_labels[:, 0] - 0).astype(np.int64)

    # select only certain trials (left,right hand imagery)
   # indices = np.where((y == 0) | (y == 1))
   # X = np.take(X, indices[0], axis=0)
   # y = np.take(y, indices[0])

    for CV in np.arange(0, num_folds):
        print('Subject No.{} CV {}'.format(test_subj, CV))
        model.network.load_state_dict(th.load(save_path))
        model.network.eval()
        # 5th phase: Model evaluation (test)
        test_set = SignalAndTarget(X, y=y)
        no_fine_tuning_score = model.evaluate(test_set.X, test_set.y)
        noFT_Accuracy = 1 - no_fine_tuning_score['misclass']
        noFT_Accuracies[idx, CV] = noFT_Accuracy


        model.network.train()
        # make different training sets, needed to switch order of train and test so test was the larger one
        # 80% training, 20% test
        train_set, test_set = split_into_train_test(test_set, n_folds=num_folds, i_test_fold=CV, rng=None)
        # 5% training, 95% test
        #test_set, train_set = split_into_train_test(test_set, n_folds = 20, i_test_fold = CV, rng=None)
        #train_set, test_set = split_into_two_sets(train_set, first_set_fraction=0.2)

        optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.1*0.001)  # weight_decay=0.1*0.001
        print('optimizer created')

        model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True)
        print('compiled')

        #num_epochs = 30
        model.fit(train_set.X, train_set.y, epochs=num_epochs, batch_size=64, scheduler='cosine',
                  input_time_length=input_time_length)
        print(model.epochs_df)

        model.network.eval()
        print(model.predict(test_set.X))

        scores = model.evaluate(test_set.X, test_set.y)
        Accuracy = 1 - scores['misclass']

        print('Accuracy (%) :', Accuracy)
        print('all done')

        # save key values
        FT_Losses[idx, :] = model.epochs_df['train_loss']
        Accuracies[idx,CV] = Accuracy
        print('yay')

        th.save({'Acc': Accuracies, 'noFT_Acc': noFT_Accuracies, 'Train_Losses': Train_Losses, 'FT_Losses': FT_Losses}, var_save_path)
    print('Overall Acc Subject {}: {},{}'.format(test_subj, np.mean(Accuracies[idx]), np.mean(noFT_Accuracies[idx])))
    idx += 1
print('last_step')
