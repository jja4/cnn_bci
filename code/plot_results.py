import numpy as np
import torch as th
import matplotlib.pyplot as plt

global_save_path = '/home/joel/PycharmProjects/DeepBCI/results/Global_resultsCV_30chan_TL.tar'  # type: str
g_measures = th.load(global_save_path)
G_Accuracies = g_measures['Acc']
#noFT_G_Accuracies = g_measures['noFT_Acc']
G_Losses = g_measures['FT_Losses']

ind_save_path = '/home/joel/PycharmProjects/DeepBCI/results/Global_resultsCV_30chan_actualTL.tar'  # type: str
i_measures = th.load(ind_save_path)
I_Accuracies = i_measures['Acc']
#noFT_I_Accuracies = i_measures['noFT_Acc']
#I_Losses = i_measures['Losses']

#X = np.mean(G_Accuracies,axis=1)
#X = X.tolist()
#X[5:5]=np.zeros(6)
#X=np.asarray(X)

plt.figure()
plt.title('Global Models, Motor Imagery BCI')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
#plt.plot(np.arange(1,noFT_G_Accuracies.shape[0]+1),np.mean(noFT_G_Accuracies,axis=1),color='green',label='No Fine Tuning Global 80 mean: {0:.2f}'.format(noFT_G_Accuracies[:8,0].mean()))
plt.plot(np.arange(1,G_Accuracies.shape[0]+1),np.mean(G_Accuracies,axis=1),color='blue',label='Global bbci train korean test mean: {0:.2f}'.format(np.mean(np.mean(G_Accuracies[G_Accuracies!=0]))))
#plt.plot(np.arange(1,G_Accuracies.shape[0]+1),np.mean(G_Accuracies,axis=1),color='blue',label='Global mean: {0:.2f}'.format(np.mean(np.mean(G_Accuracies[G_Accuracies!=0]))))
#plt.plot(np.arange(1,noFT_I_Accuracies.shape[0]+1),np.mean(noFT_I_Accuracies,axis=1),color='orange',label='No Fine Tuning Global 20 mean: {0:.2f}'.format(noFT_I_Accuracies[:8,0].mean()))
plt.plot(np.arange(1,I_Accuracies.shape[0]+1),np.mean(I_Accuracies,axis=1),color='red',label='Global TL mean: {0:.2f}'.format(np.mean(np.mean(I_Accuracies[I_Accuracies!=0]))))
plt.legend()
plt.show()

G_Accuracies.mean(axis=1)
print('thats it')