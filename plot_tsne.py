import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import pandas as pd

data = np.load("../tsne_save/2_fold_0_epoch_train.npz", allow_pickle=True)
treatment_list = data['treatment_list']
psi_im_list = data['psi_im_list']
psi_cli_list = data['psi_cli_list']
traumatic_list = data['traumatic_list']

treatment_list = treatment_list[traumatic_list==1]
psi_im_list = psi_im_list[traumatic_list==1]
psi_cli_list = psi_cli_list[traumatic_list==1]



# tsne_im = manifold.TSNE(n_components=2,perplexity=25,n_iter=300,verbose=1) 
# im_tsne = tsne_im.fit_transform(psi_im_list)

# x0_0 = im_tsne[treatment_list==0][:,0]
# x0_1 = im_tsne[treatment_list==0][:,1]

# x1_0 = im_tsne[treatment_list==1][:,0]
# x1_1 = im_tsne[treatment_list==1][:,1]

# x2_0 = im_tsne[treatment_list==2][:,0]
# x2_1 = im_tsne[treatment_list==2][:,1]

# plt.scatter(x0_0,x0_1,c='b',s=15,alpha=0.7,marker='x')
# plt.scatter(x1_0,x1_1,c='g',s=15,alpha=0.7,marker='+')
# plt.scatter(x2_0,x2_1,c='r',s=15,alpha=0.7,marker='o')



tsne_cli = manifold.TSNE(n_components=2, n_iter=3000,verbose=1)
cli_tsne = tsne_cli.fit_transform(psi_cli_list)
x0_0 = cli_tsne[treatment_list==0][:,0]
x0_1 = cli_tsne[treatment_list==0][:,1]

x1_0 = cli_tsne[treatment_list==1][:,0]
x1_1 = cli_tsne[treatment_list==1][:,1]

x2_0 = cli_tsne[treatment_list==2][:,0]
x2_1 = cli_tsne[treatment_list==2][:,1]

plt.scatter(x0_0,x0_1,c='b',s=30,alpha=0.7,marker='x')
plt.scatter(x1_0,x1_1,c='g',s=40,alpha=0.7,marker='+')
plt.scatter(x2_0,x2_1,c='r',s=30,alpha=0.7,marker='o')

plt.xticks([])
plt.yticks([])

plt.savefig('./im_tsne.pdf',dpi=120,bbox_inches='tight')

# df = pd.DataFrame(treatment_list)
# df.to_csv('0_treatment_list.csv')

# df = pd.DataFrame(psi_im_list)
# df.to_csv('0_psi_im_list.csv')

# df = pd.DataFrame(psi_cli_list)
# df.to_csv('0_psi_cli_list.csv')
