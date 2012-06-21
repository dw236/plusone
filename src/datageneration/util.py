"""Contains some useful classes and methods
"""
import numpy as np
import operator
import pickle

import math
from math import e
from math import gamma

import random
from random import random as rand

import matplotlib
from matplotlib.pyplot import *

class Poisson(object):
    """A class to represent a poisson distribution.
    
    Attributes:
        lamb:
            the parameter to the poisson distribution
    Methods:
        sample():
            returns a sample from the distribution
    """
    def __init__(self, L=15):
        self.lamb = L
        
    def sample(self):
        """Samples the poisson distribution.
        
        Args:
            none
            
        Returns:
            a number sampled from the poisson distribution with parameter
            self.lamb
        """
        L = e ** (-self.lamb)
        k, p = 1, rand()
        while p > L:
            k += 1
            p *= rand()
        return k - 1

def get_cdf(dist):
    """Calculates the cdf of a distribution.
    
    Given a distribution, calculates (and stores) the cdf for quick sampling.
    
    Args:
        dist:    distribution to construct a cdf from
    
    Returns: 
        cdf to sample from
    """
    cdf = []
    total = 0
    for i in range(len(dist)):
        total += dist[i]
        cdf.append(total)
    return np.array(cdf)

def sample(cdf):
    """Takes a distribution and samples from it. 
    
    Given a list of probabilities (that obey a distribution), samples from it 
    and returns an index of the list. This method assumes that dist obeys
    a multinomial distribution.
    
    Args:
        dist:
            A multinomial distribution represented by a list, 
            where each entry is the index's probability.
    
    Returns:
        a sample from the distribution, a random index in the list
    
    Sample usage:
        sample([.5, .5])    sample a distribution with two elements, 
                            each equally likely
    """
    p = rand()
    #this line is for rounding errors which will cause binary_search to return
    #an index that is out of bounds
    if p == 1.0:
        return cdf[-1]
    else:
        return binary_search(cdf, p)

def binary_search(elements, to_find, lo=0, hi=None):
    """Performs a binary search on an array.
    
    Unlike standard binary search, this will return the index of the nearest
    index (rounding down) if the element to find is between two values. This is
    because here binary_search is being used for the purpose of sampling from a
    cdf (see anomaly below).
    
    Args:
        elements:
            an array of values
        to_find:
            the value desired
        lo:
            the leftward bound of the search
        hi:
            the rightward bound of the search
    
    Returns:
        the index corresponding to where to_find is in elements; if to_find is
        between two indices, returns the lower one
        
    **ANOMALY**
    binary_search will return len(elements) if to_find is equal to the last
    element of elements (ie if to_find == 1.0)
    """
    if hi is None:
        hi = len(elements)
    while lo < hi:
        mid = (lo+hi)//2
        midval = elements[mid]
        if midval < to_find:
            lo = mid+1
        elif midval > to_find: 
            hi = mid
        else:
            return hi
    return hi

def normalize(dist):
    """Normalizes an array so it obeys a multinomial distribution.
    
    Assumes dist is a numpy array. Divides each element in dist by the total
    so that all entries add up to 1.
    
    Args:
        dist: an array of numbers
        
    Returns:
        a normalized version of dist
    """
    return np.array(dist, 'double') / np.sum(dist)

def dirichlet_pdf(x, alpha):
    """Calculates the probability of the given sample from a dirichlet 
    distribution.
    
    Given a sample x and parameter alpha, calculates the probability that
    x was sampled from Dirichlet(alpha).
    
    Args:
        x:
            a list of numbers in the interval [0,1] that sum to 1
        alpha:
            the parameter to a dirichlet distribution; represented as a list
    
    Returns:
        the probability that x was sampled from Dirichlet(alpha)
    """
    density = reduce(operator.mul, 
                   [x[i]**(alpha[i]-1.0) for i in range(len(alpha))])
    norm_top = gamma(np.sum(alpha))
    norm_bot = reduce(operator.mul, [gamma(a) for a in alpha])
    return (norm_top / norm_bot) * density

def count(words):
    """Creates a histogram of occurrences in an array.
    
    Given a list, counts how many times each instance occurs.
    
    Args:
        words:
            a list of values
    Returns:
        a dictionary with keys as the values that appear in words and values
        as the number of times they occur 
    """
    word_count = {}
    num_words = 0
    unique_words = 0
    for word in words:
        num_words += 1
        if word_count.has_key(word):
            word_count[word] += 1
        else:
            word_count[word] = 1
            unique_words += 1
    word_count["total"] = num_words
    word_count["unique"] = unique_words
    return word_count

def get_sig_words(word_dists, amount=0.8):
    word_cdfs = [get_cdf(sorted(dist, reverse=1)) for dist in word_dists]
    sig_words = []
    for topic in word_cdfs:
        index = binary_search(topic, amount)
        sig_words.append(index + 1)
    return sig_words

def plot_dist(types, color='b', labels=None, bottom=0, clear=None):
    """Plots a distribution as a bar graph.
    
    Given a distribution, plots a bar graph. Each bar is an element in the
    distribution, and its value is the element's probability.
    
    Args:
        types:
            a distribution, represented as a list
    Returns:
        none, but plots the distribution
    """
    if clear == None:
        clf()
    offset = 0
    width = 0.01
    if labels == None:
        labels = range(len(types))
    for dist in types:
        bar(offset, dist, width, bottom, color=color)
        offset += width
    xticks(np.arange(width / 2, width * len(types), .01), labels)

def plot_dists(types, color='b', labels=None, scale=0):
    """plots several distributions vertically stacked for easier visualization
    
    TODO: scale y-axis so labels make sense
    """
    clf()
    bottom = 0
    for dist in types:
        if len(dist) > 100:
            plot(dist + bottom)
        else:
            plot_dist(dist, color, labels, bottom, "don't clear")
        if scale == 1.0:
            to_add = scale
        else:
            to_add = max(dist) * 1.1
        bottom += to_add

def show_dists(dists):
    for dist in dists:
        plot_dist(dist)
        if raw_input('q to quit...') == 'q':
            break

def plot_cdfs(dists):
    clf()
    dists = [get_cdf(sorted(dist, reverse=1)) for dist in dists]
    bottom = 0
    for dist in dists:
        plot([bottom] * len(dist), 'black')
        plot(np.insert((1 - dist), 0, 1) + bottom)
        bottom += 1

"""
THESE FUNCTIONS ARE FOR LDA.java TO CHEAT AND ARE STILL QUITE HACKY
"""
def perplexity(docs, probabilities, indices=None, holdout=0.7):
    if indices == None:
        indices = range(len(docs))
    numerator, denominator = 0.0, 0.0
    for i in indices:
        p = 0
        words = int(len(docs[i]) * holdout)
        for word in docs[i][:words]:
            p += np.log(probabilities[i, word])
        numerator += p
        denominator += words
        print i, p, words
    print denominator
    perp = np.exp(-(numerator / denominator))
    print "LDA perplexity:", perp
    return perp

def get_probabilities(pickle_file):
    if len(pickle_file) >= 6 and pickle_file[-6:] != "pickle":
        with open('src/datageneration/output/documents_model-out', 'r') as f:
            v = False
            rbeta = []
            rgamma = []
            for line in f.readlines():
                if line == 'V\n':
                    v = True
                    continue
                if v:
                    rgamma.append([float(word) for word in line.strip(' \n').split(' ')])
                else:
                    rbeta.append([float(word) for word in line.strip(' \n').split(' ')])
        rbeta = np.matrix(rbeta)
        rgamma = np.matrix(rgamma)
        probabilities = rgamma * rbeta
    else:
        with open(pickle_file, 'r') as f:
            docs, doc_topics, words, topics, args = pickle.load(f)
        probabilities = np.matrix(topics) * np.matrix(words)
    
    return probabilities

def write_cheats(data, alpha):
    docs, doc_topics, words, topics = data
    
    with open('output/final.gamma', 'w') as f:
        gammas = [count(topic) for topic in doc_topics]
        for doc in range(len(docs)):
            for topic in range(len(topics[0])):
                if topic in gammas[doc]:
                    f.write(str(gammas[doc][topic] + alpha) + " ")
                else:
                    f.write("0 ")
            f.write('\n') 
    with open('output/final.other', 'w') as f:
        num_topics = len(words)
        num_terms = len(topics)
        to_write = "num_topics " + str(num_topics) + "\n" 
        to_write += "num_terms " + str(num_terms) + "\n" 
        to_write += "alpha " + str(alpha) + "\n"  
        f.write(to_write)
    with open('output/final.beta', 'w') as f:
        noise = 1e-323
        for topic in words:
            for word in topic:
                f.write(str(np.log(word + noise)) + " ")
            f.write('\n')
