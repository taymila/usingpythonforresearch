# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 11:33:37 2020

@author: user
"""
import numpy as np
import pandas as pd 
w=r'C:\Users\user\Using python for research\week4\whiskies.txt'
r=r'C:\Users\user\Using python for research\week4\regions.txt'
whisky=pd.read_csv(w)
whisky['region']=pd.read_csv(r)


import matplotlib.pyplot as plt
corr_flavors=pd.DataFrame.corr(whisky.iloc[:,2:14])
plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig('corr_flavors.pdf')


flavors=whisky.iloc[:,2:14]
corr_whisky=pd.DataFrame.corr(flavors.transpose())
corr_flavors=pd.DataFrame.corr(whisky.iloc[:,2:14])
plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
plt.axis('tight')
plt.colorbar()
plt.savefig('corr_whisky.pdf')

from sklearn.cluster.bicluster import SpectralCoclustering
model=SpectralCoclustering(n_clusters=6,random_state=0)
model.fit(corr_whisky)
model.rows_

np.sum(model.rows_,axis=1)

whisky['Group']=pd.Series(model.row_labels_,index=whisky.index)