"""framework for projector clustering algorithm"""
import numpy as np
from numpy.lib.npyio import loadtxt 
from random import sample as rsample
from random import randint
from matplotlib.pylab import *
import argparse

def cos_sim(a, b):
    return a.dot(b) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))

def dist_cos(a, b):
    """use cosine similarity as a distribution
    """
    return np.exp(10 * cos_sim(a, b))

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

def get_top(dist, num):
    """
    dist: distribution
    num: number of elements to keep
    """
    threshold = sorted(dist, reverse=True)[:num][-1]
    dist[dist < threshold] = 0
    return dist / np.sum(dist)

class Kmeans():
    """
    k: number of cluster centers
    centers: final cluster centers
    points: datapoints to be clustered
    labels: the cluster each datapoint belongs to
    total_distance: distance to each point's cluster center (non-fuzzy only)
    """
    RECOGNIZED_INITS = ['points', 'random']
    RECOGNIZED_TYPES = ['normal', 'fuzzy']
    TOLERANCE = 1e-3
    
    def __init__(self, k, init="points", metric="cosine", type="normal"):
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
            self.farthest = np.min
            self.better = lambda x, y: x > y
        elif metric == "euclidean":
            self.metric = euclidean_dist
            self.select = np.argmin
            self.farthest = np.max
            self.better = lambda x, y: x < y
        elif metric == "distcos":
            self.metric = dist_cos
            self.select = np.argmax
            self.farthest = np.min
            self.better = lambda x, y: x > y
        elif metric == "dot":
            self.metric = lambda x, y: x.dot(y)
            self.select = np.argmax
            self.farthest = np.min
            self.better = lambda x, y: x > y
        else:
            raise Exception("unrecognized metric: " + str(metric))
        
    def cluster(self, points, max_iter=np.inf, verbose=False):
        """
        n: number of points
        m: dimension of each point
        """
        self.points = points
        self.n, self.m = np.shape(self.points)
        iterations = 0
        
        if self.init == "points":
            self.centers = rsample(points, self.k)
            self.labels = self.assign()
        elif self.init == "random":
            if self.type == "normal":
                self.labels = np.array([np.random.randint(0, self.k)
                                        for point in self.points])
            elif self.type == "fuzzy":
                labels = []
                for point in self.points:
                    cluster = np.zeros(self.k)
                    cluster[np.random.randint(0, self.k)] += 1
                    labels.append(np.array(cluster))
                self.labels = np.array(labels)
        self.centers = self.update()
        while iterations <= max_iter:
            if iterations % 10 == 0:
                if verbose:
                    print iterations#, self.centers
            new_labels = self.assign()
            if all(new_labels == self.labels) or (np.all(abs(new_labels - 
                                                             self.labels) 
                                                  < self.TOLERANCE)):
                if verbose:
                    print "converged in", iterations, "iterations"
                break
            else:
                iterations += 1
                self.labels = new_labels
                self.centers = self.update()
        self.total_distance = self.get_total_distance()

    def assign(self, points=None):
        """
        points: the points to be clustered
        centers: the cluster centers that each point will be assigned to
        """
        if points == None:
            points = self.points
            n, m = self.n, self.m
        else:
            n, m = np.shape(points)
        
        labels = []
        farthest_distance = None
        farthest_point_index = None
        for i in range(n):
            point = points[i]
            distances = [self.metric(point, center) for center in self.centers]
            if self.type == 'normal':
                labels.append(self.select(distances))
                distance = self.farthest(distances)
                if farthest_distance == None or not self.better(distance, 
                                                             farthest_distance):
                    farthest_distance = distance
                    farthest_point_index = i
            elif self.type == 'fuzzy':
                distances = np.array(distances)
                #distances = get_top(distances, 5) #Hack -- remove ASAP
                labels.append(distances / sum(distances))
        self.farthest_point = points[farthest_point_index]
        self.farthest_point_index = farthest_point_index
        
        return np.array(labels)
    
    def update(self):
        """
        """
        centers = []
        for cluster in range(self.k):
            if self.type == 'normal':
                cluster_points = self.points[np.where(self.labels == cluster)
                                             [0]]
                points_in_cluster = len(cluster_points)
                if points_in_cluster == 0:
                    print "warning, empty cluster encountered"
                    centers.append(self.farthest_point)
                    self.labels[self.farthest_point_index] = cluster
                else:
                    centers.append(np.sum(cluster_points, axis=0) 
                                   / len(cluster_points))
            elif self.type == 'fuzzy':
                contribution = self.labels[:,cluster]
                contributions = np.array([ones(self.m) * label 
                                          for label in contribution])
                centers.append(np.sum(self.points * contributions, axis=0)
                               / np.sum(contribution))
        
        return np.array(centers)
    
    def get_total_distance(self):
        """
        """
        if self.type != "normal":
            return None
        
        total_distance = 0
        for i in range(self.n):
            point = self.points[i]
            center = self.centers[self.labels[i]]
            distance = self.metric(point, center)
            
            total_distance += distance
        return total_distance
    
    def plot(self, clear=True):
        """only works for 2D points
        """
        if clear:
            figure()
            clf()
        
        colors = 'brgmyck'
        
        for cluster in range(self.k):
            color = colors[cluster % len(colors)]
            if self.type == 'normal':
                cluster_points = self.points[np.where(self.labels 
                                                      == cluster)[0]]
            elif self.type == 'fuzzy':
                labels = np.array([np.argmax(label) for label in self.labels])
                cluster_points = self.points[np.where(labels
                                                      == cluster)[0]]
            plot([point[0] for point in cluster_points], 
                 [point[1] for point in cluster_points],
                 color + 'o')

            plot(self.centers[cluster][0], self.centers[cluster][1],
                 color + 'x')
            show()
    
    def write_labels(self, filename):
        with open(filename, 'w') as f:
            for label in self.labels:
                if self.type == 'normal':
                    to_write = str(label + 1) + ' ' 
                elif self.type == 'fuzzy':
                    to_write = ' '.join([str(dimension) 
                                         for dimension in label]) + ' \n'
                f.write(to_write)
            f.write('\n')
    
    def write_centers(self, filename):
        with open(filename, 'w') as f:
            for center in self.centers:
                for dimension in center:
                    f.write(str(dimension) + ' ')
                f.write('\n')

def repeated_kmeans(k, init, metric, type, iterations=1):
    """
    iterations: number of times to run clustering (returns best one)
    rest are same as Kmeans.cluster
    """
    best_cluster = Kmeans(0)
    best_cluster.total_distance = None
    for iteration in range(args.i):
        cluster = Kmeans(args.k, init="points", metric=args.m, type=type)
        cluster.cluster(points, 100)
        #print cluster.total_distance
        if best_cluster.total_distance == None or \
           cluster.better(cluster.total_distance, best_cluster.total_distance):
            best_cluster = cluster
    return best_cluster

def main():
    parser = argparse.ArgumentParser(description="kmeans algorithm \
                                     (default values are in parentheses)")
    parser.add_argument('f', metavar="filename", action="store",
                        help="filename of datapoints to be clustered")
    parser.add_argument('-k', action="store", metavar='num clusters', 
                        type=int, default=2, 
                        help="number of clusters to split data into (2)")
    parser.add_argument('-m', action="store", metavar='metric', 
                        default='euclidean', 
                        help='distance metric used for clustering (euclidean)')
    parser.add_argument('-w', action="store", metavar='write filename',
                        help='filename for writing cluster labels (False)')
    parser.add_argument('-i', action="store", metavar='iterations',
                        default=1, type=int,
                        help='number of times to run clustering (1)')
    parser.add_argument('-e', action="store", metavar='empty action',
                        default="singleton",
                        help="action to perform when empty cluster arises \
                        (singleton--OTHER OPTIONS CURRENTLY UNSUPPORTED)")
    parser.add_argument('-t', action="store_true", default=False, 
                        help='flag to use fuzzy clustering (Off)')
    parser.add_argument('-q', action="store_true", default=False,
                        help='flag to plot output (Off)')
    
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
    
    cluster = repeated_kmeans(args.k, init="points", metric=args.m, type=type,
                              iterations=args.i)
    
    if args.q:
        print "suppressing plot of output"
    else:
        cluster.plot()
    
    if args.w != None:
        print "writing cluster labels to file:", args.w
        cluster.write_labels(args.w)
        centers_filename = args.w[:-6] + 'centers' #hack--remove ASAP
        print "writing centers to file: ", centers_filename
        cluster.write_centers(centers_filename)
    
    return points, cluster

if __name__ == '__main__':
    points, km = main()
