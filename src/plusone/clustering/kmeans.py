"""framework for projector clustering algorithm"""
import numpy as np
from numpy.lib.npyio import loadtxt 
from random import sample as rsample
from random import randint
from matplotlib.pylab import *
import argparse

def cos_sim(a, b):
    return a.dot(b) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))

def euclidean_dist(a, b):
    return np.linalg.linalg.norm(a - b)

def get_points(filename):
    try:
        points = loadtxt(filename)
    except:
        print "loadtxt failed--reading points manually"
        with open(filename, 'r') as f:
            lines = [line.strip(' \n').split(' ') for line in f.readlines()]
            points = []
            for line in lines:
                points.append([float(num) for num in line])
        points = np.array(points)
    
    return points

class Kmeans():
    """
    k: number of cluster centers
    centers: final cluster centers
    points: datapoints to be clustered
    labels: the cluster each datapoint belongs to
    """
    RECOGNIZED_INITS = ['points', 'random']
    RECOGNIZED_TYPES = ['normal', 'fuzzy']
    
    def __init__(self, k, init="points", metric="cosine", type="fuzzy"):
        """
        init: how to initialize the cluster centers (defaults to picking random
        observations as clusters)
        metric: distance metric for datapoints
        type: standard kmeans clustering (normal) or fuzzy clustering (fuzzy)
        """
        self.k = k
        if init in self.RECOGNIZED_INITS:
            self.init = init
        else:
            raise Exception("unrecognized initialization: " + str(init))
        if type in self.RECOGNIZED_TYPES:
            self.type = type
        else:
            raise Exception("unrecognized type: " + str(type))
        if metric == "cosine":
            self.metric = cos_sim
            self.select = np.argmax
        elif metric == "euclidean":
            self.metric = euclidean_dist
            self.select = np.argmin
        else:
            raise Exception("unrecognized metric: " + str(metric))
        
    def cluster(self, points, max_iter=np.inf):
        self.points = points
        iterations = 0
        
        if self.init == "points":
            self.centers = rsample(points, self.k)
            self.labels = self.assign()
        elif self.init == "random":
            self.labels = np.array([np.random.randint(0, self.k)
                                    for point in len(self.points)])
        self.centers = self.update()
        while iterations <= max_iter:
            if iterations % 10 == 0:
                print iterations
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
            if self.type == 'normal':
                labels.append(self.select(distances))
            elif self.type == 'fuzzy':
                distances = np.array(distances)
                labels.append(distances / sum(distances))
            
        return np.array(labels)
    
    def update(self):
        """
        n: number of points
        m: dimension of each point
        """
        centers = []
        n, m = np.shape(self.points)
        for cluster in range(self.k):
            if self.type == 'normal':
                cluster_points = self.points[np.where(self.labels == cluster)
                                             [0]]
                centers.append(np.sum(cluster_points, axis=0) 
                               / len(cluster_points))
            elif self.type == 'fuzzy':
                contribution = self.labels[:,cluster]
                contributions = np.array([ones(m) * label 
                                          for label in contribution])
                centers.append(np.sum(self.points * contributions, axis=0)
                               / np.sum(contribution))
        
        return np.array(centers)
    
    def plot(self):
        """only works for 2D points
        """
        colors = 'brgmyck'
        
        for cluster in range(self.k):
            color = colors[cluster % len(colors)]
            if self.type == 'normal':
                cluster_points = self.points[np.where(self.labels 
                                                      == cluster)[0]]
                plot([point[0] for point in cluster_points], 
                     [point[1] for point in cluster_points],
                     color + 'o')
            elif self.type == 'fuzzy':
                plot([point[0] for point in points],
                     [point[1] for point in points],
                     'bo')
            plot(self.centers[cluster][0], self.centers[cluster][1],
                 color + 'x')
    
    def write_labels(self, filename):
        with open(filename, 'w') as f:
            for label in self.labels:
                f.write(str(label + 1) + ' ')
            f.write('\n')
            

def main():
    parser = argparse.ArgumentParser(description="Fuzzy kmeans algorithm \
                                     (default values are in parentheses)")
    parser.add_argument('f', metavar="filename", action="store",
                        help="filename of datapoints to be clustered")
    parser.add_argument('-k', action="store", metavar='num_clusters', type=int,
                        default=2, 
                        help="number of clusters to split data into (2)")
    parser.add_argument('-m', action="store", metavar='metric', 
                        default='euclidean', 
                        help='distance metric used for clustering (euclidean)')
    parser.add_argument('-w', action="store", metavar='write filename',
                        help='filename for writing cluster labels (False)')
    parser.add_argument('-t', action="store_true", default=False, 
                        help='flag to use fuzzy clustering (False)')
    parser.add_argument('-i', action="store_true", default=False,
                        help='flag to use random init (False)')
    
    args = parser.parse_args()
    
    if args.f == None:
        print "no points supplied: using default random points"
        points = np.array(zip([rsample([1,10], 1)[0] for i in range(15)], 
                              [randint(10, 20) for i in range(15)]))
    else:
        print "loading points from", args.f
        points = get_points(args.f)
        print "points succesfully loaded [ shape:", np.shape(points), "]"
    if args.t:
        type = "fuzzy"
    else:
        type = "normal"
    if args.i:
        init = "points"
    else:
        init = "random"
        
    cluster = Kmeans(args.k, init=init, metric=args.m, type=type)
    cluster.cluster(points)
    
    if args.w != None:
        print "writing cluster labels to file:", args.w
        cluster.write_labels(args.w)
    
    return points, cluster

if __name__ == '__main__':
    points, km = main()
