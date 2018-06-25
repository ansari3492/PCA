# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:00:52 2018

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("crime_data.csv")

features=data.iloc[:,[1,2,-1]].values

#clustering
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("no of clusters")
plt.ylabel("wcss")
plt.show()

    





#apply the PCA
 
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
features=pca.fit_transform(features)

explained_variance=pca.explained_variance_ratio_




kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(features)

#visualizing
plt.scatter(features[y_kmeans==0,0],features[y_kmeans==0,1],s=100,c='red',label='murder')
plt.scatter(features[y_kmeans==1,0],features[y_kmeans==1,1],s=100,c='green',label='assault')
plt.scatter(features[y_kmeans==2,0],features[y_kmeans==2,1],s=100,c='blue',label='rape')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='pink',label='centroid')
plt.title("murder vs rape vs assault")

plt.legend()
plt.show()





