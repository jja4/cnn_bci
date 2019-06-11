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
train_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG/30channel_0604_3class/' #62 channel datsets
test_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG_hanyang/30channel_0529/'
#default_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG_hanyang/'
#default_path ='/home/joel/PycharmProjects/DeepBCI/data/Only_LR/' #88 and 84 channel datasets
save_path = '/home/joel/PycharmProjects/DeepBCI/code/3class_BCImodel.pt'

os.chdir(test_path)
filename = '*labels.mat'
subj_files = glob.glob(filename)
#extract subject numbers from filenames
regex = re.compile(r'\d+')
test_subjs = [int(x) for i in range(0, len(subj_files)) for x in regex.findall(subj_files[i])]
test_subjs.sort()

os.chdir(train_path)
filename = '*labels.mat'
subj_files = glob.glob(filename)
#extract subject numbers from filenames
regex = re.compile(r'\d+')
train_subjs = [int(x) for i in range(0, len(subj_files)) for x in regex.findall(subj_files[i])]
train_subjs.sort()

#load list of subjects that only did left/right imagined movements
#subjs_path = '/home/joel/PycharmProjects/DeepBCI/results/LR_subjs.tar'
#subjs = th.load(subjs_path)

num_epochs = 30
#save important numbers
Accuracies = np.zeros((len(test_subjs), 5))
noFT_Accuracies = np.zeros((len(test_subjs), 5))
Train_Losses = np.zeros((len(train_subjs), num_epochs + 1))
FT_Losses = np.zeros((len(test_subjs), num_epochs + 1))

var_save_path = '/home/joel/PycharmProjects/DeepBCI/results/Global_resultsCV_30chan_actualTL.tar'  # type: str

#measures = th.load(var_save_path)
#Accuracies = measures['Acc']
#Losses = measures['Losses']



# option to start training at a specific index
idx =0
### Training loop #############################################################
print('TRAINING')
sub_idx = 0

# create model
cuda = th.cuda.is_available()
set_random_seeds(seed=20170629, cuda=cuda)

n_classes = 3
in_chans = 30  # train_set.X.shape[1]  # number of channels = 128
input_time_length = 500  # length of time of each epoch/trial = 4000

model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=input_time_length,
                        final_conv_length=15, )
if cuda:
    model.cuda()
print('cuda: ' + str(cuda))

'''
for i in train_subjs:  # subject loop
    print('Train Subject No.{}'.format(i))
    raw_data = io.loadmat(train_path + 'sub_' + str(i) + '.mat')
    labels = io.loadmat(train_path + 'sub_' + str(i) + '_labels.mat')
    FeatVect = raw_data['FeatVect']
    y_labels = labels['Triggers']

    # preprocessing
    for ii in range(0, FeatVect.shape[0]):
        # 1. Data reconstruction
        temp_data = FeatVect[ii, :, :]
        temp_data = temp_data.transpose()
        # 2. Lowpass filtering
        lowpassed_data = lowpass_cnt(temp_data, 100, 200, filt_order=5)
        # 3. Highpass filtering
        bandpassed_data = highpass_cnt(lowpassed_data, 4, 200, filt_order=3)
        # 4. Exponential running standardization
        ExpRunStand_data = exponential_running_standardize(bandpassed_data, factor_new=0.01, init_block_size=None,
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
    print(y)
    #select only certain trials (left,right hand imagery)
   # indices = np.where((y == 0) | (y == 1))
   # X = np.take(X, indices[0], axis=0)
   # y = np.take(y, indices[0])

    train_set = SignalAndTarget(X, y=y)
    print('data prepared')
    optimizer = AdamW(model.parameters(), lr= 0.01, weight_decay= 0.1*0.001)
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True)
    print('fitting data')

    model.fit(train_set.X, train_set.y, epochs=num_epochs, batch_size=64, scheduler= 'cosine',
            input_time_length=input_time_length)
    print(model.epochs_df)

    Train_Losses[sub_idx, :] = model.epochs_df['train_loss']

    #save model
    th.save(model.network.state_dict(), save_path)
    sub_idx += 1
'''
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.1 * 0.001)  # weight_decay=0.1*0.001
model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True)
model.network.load_state_dict(th.load(save_path))
pretrained_dict = model.network.state_dict()
## Test
# Cross validation for test data
for test_subj in test_subjs:
    raw_data = io.loadmat(test_path + 'sub_' + str(test_subj) + '.mat')
    labels = io.loadmat(test_path + 'sub_' + str(test_subj) + '_labels.mat')
    FeatVect = raw_data['FeatVect']
    y_labels = labels['Triggers']
    for ii in range(0, FeatVect.shape[0]):
        # 1. Data reconstruction
        temp_data = FeatVect[ii, :, :]
        temp_data = temp_data.transpose()
        # 2. Lowpass filtering
        lowpassed_data = lowpass_cnt(temp_data, 100, 200, filt_order=5)
        # 3. Highpass filtering
        bandpassed_data = highpass_cnt(lowpassed_data, 4, 200, filt_order=3)
        # 4. Exponential running standardization
        ExpRunStand_data = exponential_running_standardize(bandpassed_data, factor_new=0.01, init_block_size=None,
                                                           eps=0.0001)
        # 5. Renewal preprocessed data
        ExpRunStand_data = ExpRunStand_data.transpose()
        FeatVect[ii, :, :] = ExpRunStand_data
        del temp_data, lowpassed_data, bandpassed_data, ExpRunStand_data

    X = FeatVect.astype(np.float32)
    X = X[:, :, :]
    y = (y_labels[:, 0] - 0).astype(np.int64)
    print(y)
    # select only certain trials (left,right hand imagery)
    # indices = np.where((y == 0) | (y == 1))
    # X = np.take(X, indices[0], axis=0)
    # y = np.take(y, indices[0])

    for CV in np.arange(0, 5):
        print('Subject No.{} CV {}'.format(test_subj, CV))
        n_classes = 2
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                input_time_length=input_time_length,
                                final_conv_length=15, )
        optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.1*0.001)  # weight_decay=0.1*0.001
        model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, cropped=True)
        model_dict = model.network.state_dict()
        pretrained_dict_reduced = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict_reduced)
        #model_dict['conv_classifier.weight'] = np.zeros(model_dict['conv_classifier.weight'].shape)
        # 3. load the new state dict
        model.network.load_state_dict(model_dict)

        #model.network.load_state_dict(th.load(save_path))
        model.network.eval()
        # 5th phase: Model evaluation (test)
        test_set = SignalAndTarget(X, y=y)
        #no_fine_tuning_score = model.evaluate(test_set.X, test_set.y)
        #noFT_Accuracy = 1 - no_fine_tuning_score['misclass']
        #noFT_Accuracies[idx, CV] = noFT_Accuracy


        model.network.train()
        # make different training sets, needed to switch order of train and test so test was the larger one
        test_set, train_set = split_into_train_test(test_set, n_folds = 5, i_test_fold = CV, rng=None)


        print('fitting data')

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

        th.save({'Acc':Accuracies,'noFT_Acc':noFT_Accuracies,'Train_Losses': Train_Losses,'FT_Losses': FT_Losses}, var_save_path)
    print('Overall Acc Subject {}: {},{}'.format(test_subj, np.mean(Accuracies[idx]), np.mean(noFT_Accuracies[idx])))
    idx += 1
print('last_step')
