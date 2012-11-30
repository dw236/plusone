"""framework for projector clustering algorithm"""
import numpy as np
from random import sample as rsample
from matplotlib.pylab import *

def cos_sim(a, b):
    return a.dot(b) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))

def euclidean_dist(a, b):
    return np.linalg.linalg.norm(a - b)

class Kmeans():
    """
    k: number of cluster centers
    centers: final cluster centers
    points: datapoints to be clustered
    labels: the cluster each datapoint belongs to
    """
    def __init__(self, k, init="points", metric="cosine"):
        """
        init: how to initialize the cluster centers (defaults to picking random
        observations as clusters)
        metric: distance metric for datapoints
        """
        self.k = k
        self.init = init
        if metric == "cosine":
            self.metric = cos_sim
        elif metric == "euclidean":
            self.metric = euclidean_dist
        else:
            raise Exception("unrecognized metric: " + str(metric))
        
    def cluster(self, points):
        self.points = points
        iterations = 0
        
        if self.init == "points":
            self.centers = rsample(points, self.k)
            self.labels = self.assign()
            self.centers = self.update()
            while True:
                new_labels = self.assign()
                if all(new_labels == self.labels):
                    print "converged in", iterations, "iterations"
                    break
                else:
                    iterations += 1
                    self.labels = new_labels
                    self.centers = self.update()

    def assign(self, points=None):
        """
        points: the points to be clustered
        centers: the cluster centers that each point will be assigned to
        """
        if points == None:
            points = self.points
        
        labels = []
        for point in self.points:
            distances = [self.metric(point, center) for center in self.centers]
            labels.append(np.argmin(distances))
            
        return np.array(labels)
    
    def update(self):
        """
        """
        centers = []
        for cluster in range(self.k):
            cluster_points = self.points[np.where(self.labels == cluster)[0]]
            centers.append(np.sum(cluster_points, axis=0) / len(cluster_points))
        
        return np.array(centers)
    
    def plot(self):
        """only works for 2D points
        """
        colors = 'brgmyck'
        
        for cluster in range(self.k):
            color = colors[cluster % len(colors)]
            cluster_points = self.points[np.where(self.labels == cluster)[0]]
            plot([point[0] for point in cluster_points], 
                 [point[1] for point in cluster_points],
                 color + 'o')
            plot(self.centers[cluster][0], self.centers[cluster][1],
                 color + 'x')

test = Kmeans(2, metric='euclidean')
