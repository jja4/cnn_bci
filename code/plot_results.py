import numpy as np
import torch as th
import matplotlib.pyplot as plt

global_save_path = '/home/joel/PycharmProjects/DeepBCI/results/Global_resultsCV_62chan.tar'  # type: str
g_measures = th.load(global_save_path)
G_Accuracies = g_measures['Acc']
G_Losses = g_measures['Losses']

ind_save_path = '/home/joel/PycharmProjects/DeepBCI/results/Individual_resultsCV_8884chan_.tar'  # type: str
i_measures = th.load(ind_save_path)
I_Accuracies = i_measures['Acc']
I_Losses = i_measures['Losses']

#X = np.mean(G_Accuracies,axis=1)
#X = X.tolist()
#X[5:5]=np.zeros(6)
#X=np.asarray(X)

plt.figure()
plt.title('Individual Model, Motor Imagery BCI')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
#plt.plot(np.arange(1,len(X)+1),X,color='blue',label='Global mean: {0:.2f}'.format(np.mean(X[X!=0])))
#plt.plot(np.arange(1,G_Accuracies.shape[0]+1),np.mean(G_Accuracies,axis=1),color='blue',label='Global mean: {0:.2f}'.format(np.mean(np.mean(G_Accuracies,axis=1))))
plt.plot(np.arange(1,I_Accuracies.shape[0]+1),np.mean(I_Accuracies,axis=1),color='red',label='Individual mean: {0:.2f}'.format(np.mean(np.mean(I_Accuracies,axis=1))))
plt.legend()
plt.show()
print('thats it')