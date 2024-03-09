import sys
import math as mt
import numpy as np
class KMeans:
    def __init__(self,k=1,rand_state = 231, max_iter = 100):
        self.k = k
        self.x = None
        self.rand = rand_state
        self.clusters = None
        self.max_iter = max_iter
    def __distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def __setdata(self,x):
        self.x = x

    def __setclusters(self,cl):
        self.clusters = cl

    def getclusters(self):
        return self.clusters
    def fit(self,x):
        self.__setdata(x)

        clusters = {}
        np.random.seed(self.rand)

        for idx in range(self.k):
            center = (2*np.random.random((self.x.shape[1],))-1)
            points = []
            cluster = {
                'center': center,
                'points':points
            }
            clusters[idx] = cluster

        clusters = self.__assign_clusters(clusters)
        clusters = self.__update_clusters(clusters)
        self.__setclusters(clusters)
        return 

    def __assign_clusters(self,clusters):
        for idx in range(self.x.shape[0]):
            distance = []
            current_x = self.x[idx]
    
            for i in range(self.k):
                dist = self.__distance(current_x,clusters[i]['center'])
                distance.append(dist)
            current_cluster = np.argmax(distance)
            clusters[current_cluster]['points'].append(current_x)

        return clusters

    def __update_clusters(self,clusters):
        for i in range(self.k):
            points = np.array(clusters[i]['points'])
            if points.shape[0] > 0:
                newcenter = points.mean(axis = 0)
                clusters[i]['center'] = newcenter
                clusters[i]['points'] = []

        return clusters


    def predict_clusters(self):
        pred = []
        for i in range(self.x.shape[0]):
            dist = []
            for j in range(self.k):
                dist.append(self.__distance(self.x[i],self.clusters[j]['center']))
            pred.append(np.argmin(dist))
        return pred