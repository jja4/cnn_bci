from scipy import io
import re
import numpy as np
import torch as th
import os
import glob


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#default_path = '/home/seulki/PycharmProjects/data/Open_BCI_EEG/'
default_path = '/home/joel/PycharmProjects/DeepBCI/data/Only_LR/'
save_path = '/home/joel/PycharmProjects/DeepBCI/code/BCImodel.pt'

filename = default_path+'*labels.mat'
subj_files = glob.glob(filename)

regex = re.compile(r'\d+')
subjs = [int(x) for i in range(0,len(subj_files)) for x in regex.findall(subj_files[i])]
subjs.sort()

#save important numbers
var_save_path = '/home/joel/PycharmProjects/DeepBCI/results/LR_subjs.tar'  # type: str
th.save(subjs, var_save_path)

#measures = th.load(var_save_path)
#Accuracies = measures['Acc']
num_chans = []

idx =0
### Training loop #############################################################
for test_subj in subjs:
    raw_data = io.loadmat(default_path + 'sub_' + str(test_subj) + '.mat')
    labels = io.loadmat(default_path + 'sub_' + str(test_subj) + '_labels.mat')
    FeatVect = raw_data['FeatVect']
    y_labels = labels['Triggers']
    in_chans = FeatVect.shape[1]
    print('Subj {}: num_chans {}'.format(test_subj,in_chans))
    if in_chans ==88:
        num_chans.append(test_subj)
        idx +=1
#th.save(num_chans, var_save_path)
print('done')

