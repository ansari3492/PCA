# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:55:16 2018

@author: Lenovo
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()
iris = iris.data


#apply the PCA
 
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
iris_data=pca.fit_transform(iris)

explained_variance=pca.explained_variance_ratio_

#clustering
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(iris_data)

#visualizing
plt.scatter(iris_data[y_kmeans==0,0],iris_data[y_kmeans==0,1],s=100,c='red',label='setosa')
plt.scatter(iris_data[y_kmeans==1,0],iris_data[y_kmeans==1,1],s=100,c='green',label='versicolor')
plt.scatter(iris_data[y_kmeans==2,0],iris_data[y_kmeans==2,1],s=100,c='blue',label='verginica')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='pink',label='centroid')
plt.title("setosa vs versicolor vs verginica")

plt.legend()
plt.show()

