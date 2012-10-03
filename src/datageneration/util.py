"""Contains some useful classes and methods
"""
import operator
import pickle
import time

import numpy as np
from numpy.random.mtrand import poisson as p
from numpy.random.mtrand import dirichlet

import math
from math import e
from math import gamma

import random
from random import random as rand

import matplotlib
from matplotlib.pyplot import *

def poisson(l, max_val=None, min_val=1):
    """samples a poisson distribution, but has a bounded max and min value
    
    Args:
        l:
            poisson parameter
        max_val:
            the maximum value that can be returned (if the sampled number is
            higher than max_val, max_val is returned)
            if no max_val is given, does not cap the max value
        min_val:
            the minimum value that can be returned (if the sampled number is
            smaller than min_val, min_val is returned)
    
    Returns:
        a sample p ~ Poisson(l), but is constrained to the range 
        [min_val, max_val]
    """
    if max_val == None:
        return max(1, p(l))
    else:
        return max(1, min(p(l), max_val))

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
    """calculates the number of significant elements in a distribution
    
    Given a distribution, calculates the number of elements (sorted by
    decreasing probability, so the most likely element is first) that comprise
    a specified percentage of the cdf.
    
    Args:
        word_dists:
            a list of distributions
        amount:
            percentage of cdf to calculate significant elements
            
    Returns:
        a list, where each element corresponds to how many elements in that
        distribution contribute "amount" percent of the cdf in that distribution
    """
    word_cdfs = [get_cdf(sorted(dist, reverse=1)) for dist in word_dists]
    sig_words = []
    for topic in word_cdfs:
        index = binary_search(topic, amount)
        sig_words.append(index + 1)
    return sig_words

def plot_dist(types, color='b', labels=None, bottom=0, clear=True):
    """Plots a distribution as a bar graph.
    
    Given a distribution, plots a bar graph. Each bar is an element in the
    distribution, and its value is the element's probability.
    
    Args:
        types:
            a distribution, represented as a list
    Returns:
        none, but plots the distribution
    """
    if clear:
        clf()
    offset = 0
    width = 0.01
    if labels == None:
        labels = range(len(types))
    for dist in types:
        bar(offset, dist, width, bottom, color=color)
        offset += width
    xticks(np.arange(width / 2, width * len(types), .01), labels)

def plot_dists(types, color='b', labels=None, scale=0, clear=True):
    """plots several distributions vertically stacked for easier visualization
    
    TODO: scale y-axis so labels make sense
    """
    if clear:
        clf()
    bottom = 0
    y_indices = []
    for dist in types:
        y_indices.append(bottom)
        if len(dist) > 50:
            plot(dist + bottom, color=color)
        else:
            plot_dist(dist, color, labels, bottom, clear=False)
        if scale != 0:
            to_add = scale
        else:
            to_add = max(dist) * 1.1
        bottom += to_add
    if labels != None:
        yticks(y_indices, labels)

def show_dists(dists):
    for dist in dists:
        plot_dist(dist)
        if raw_input('q to quit...') == 'q':
            break

def plot_cdfs(dists, offset=1):
    clf()
    dists = [get_cdf(sorted(dist, reverse=1)) for dist in dists]
    bottom = 0
    for dist in dists:
        plot([bottom] * len(dist), 'black')
        plot(np.insert((1 - dist), 0, 1) + bottom)
        bottom += offset

def perplexity(docs, probabilities, indices=None, holdout=0.7):
    """deprecated: feel free to remove this function
    """
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
    """Calculates the full NxM word distribution per document
    
    Given a file containing the model parameters used to generate a dataset,
    calculates the probability for words in each document by multiplying the
    topic distribution per document (a NxK matrix) by the word distribution
    per topic (a KxM matrix).
    
    Args:
        pickle_file:
            Name of the file to load data from; if it is a .pickle file, unpacks
            the relevant values and computes the resulting matrix. Otherwise,
            assumes a text file reads the entries from it.
    
    Returns:
        an NxM matrix, where each entry is the probability of a word given a
        document (the document is specified by the row, and the word is
        specified by the column) 
    """
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
                    rgamma.append([float(word) \
                                   for word in line.strip(' \n').split(' ')])
                else:
                    rbeta.append([float(word) \
                                  for word in line.strip(' \n').split(' ')])
        rbeta = np.matrix(rbeta)
        rgamma = np.matrix(rgamma)
        probabilities = rgamma * rbeta
    else:
        with open(pickle_file, 'r') as f:
            docs, doc_topics, words, topics, args = pickle.load(f)
        probabilities = np.matrix(topics) * np.matrix(words)
    
    return probabilities

def write_cheats(data, args, dir):
    """writes files that will help lda cheat (replaces the files in lda/)
    
    Writes the three files to replace the three that lda-c-dist creates 
    after training and inference. Used to cheat by giving the prediction task
    the true model parameters instead of the learned ones.
    
    Note: this code does not replace the files; it only creates the files so
    that they can be used to replace lda's files later.
    
    Args:
        data:
            tuple containing all the information returned by 
            documents.generate_documents
        args:
            namespace containing parameters given to generate the data
        dir:
            directory to write the files (to be copied later)
    Returns:
        none, but writes three files to be used later
    """
    docs, doc_topics, words, topics = data
    alpha = args.a
    
    with open(dir + '/final.gamma', 'w') as f:
        gammas = [count(topic) for topic in doc_topics]
        for doc in range(len(docs)):
            for topic in range(len(topics[0])):
                if topic in gammas[doc]:
                    f.write(str(gammas[doc][topic] + alpha) + " ")
                else:
                    f.write(str(alpha) + " ")
            f.write('\n') 
    with open(dir + '/final.other', 'w') as f:
        num_topics = args.k
        num_terms = args.m
        to_write = "num_topics " + str(num_topics) + "\n" 
        to_write += "num_terms " + str(num_terms) + "\n" 
        to_write += "alpha " + str(alpha) + "\n"  
        f.write(to_write)
    with open(dir + '/final.beta', 'w') as f:
        noise = 1e-323
        for topic in words:
            for word in topic:
                f.write(str(np.log(word + noise)) + " ")
            f.write('\n')

def match_beta(input_beta='../../projector/data/final.beta',
               real_beta='output/results.pickle', 
               metric='cosine', plot=True):
    with open(real_beta, 'r') as f:
        real_beta = pickle.load(f)[2]
    
    with open(input_beta, 'r') as f:
        input_beta = [line.strip(' \n').split(' ') for line in f.readlines()]
    lines_to_nums = []
    for line in input_beta:
        lines_to_nums.append([np.exp(float(num)) for num in line])
    input_beta = np.array(lines_to_nums)
    
    r_shape, i_shape = np.shape(real_beta), np.shape(input_beta)
    assert(r_shape[0] == i_shape[0])
    if r_shape[1] != i_shape[1]:
        print "input beta is missing words--padding with 0's"
        #pad input_beta
        lines = []
        for line in input_beta:
            lines.append(list(line) + [0]*(r_shape[1] - i_shape[1]))
        input_beta = np.array(lines)
    
    #generate preference list
    male_distances = np.zeros((np.shape(real_beta)[0], 
                               np.shape(input_beta)[0]))
    female_distances = np.zeros((np.shape(input_beta)[0],
                                 np.shape(real_beta)[0]))
    for i in range(len(real_beta)):
        for j in range(len(input_beta)):
            if metric == 'cosine':
                sim = real_beta[i].dot(input_beta[j]) / \
                      (np.sqrt(real_beta[i].dot(real_beta[i])) * 
                       np.sqrt(input_beta[j].dot(input_beta[j])))
                invert = False
            elif metric == 'L1':
                sim = np.sum(np.abs(real_beta[i] - input_beta[j]))
                #invert so it makes larger values less preferable
                invert = True
            else:
                raise Exception("invalid metric: " + str(metric))
            male_distances[i][j] = sim
            female_distances[j][i] = sim
    pairings = tma(male_distances, female_distances, invert=invert)
    labels = [pairings[i][0] for i in sorted(pairings.keys())]
    labels = [labels.index(female) for female in sorted(pairings.keys())]
    reordered_input_beta = np.array([input_beta[i] for i in labels])
    labels = zip(labels, [round(male_distances[i][labels[i]], 2)
                          for i in range(len(labels))])
    if plot:
        plot_dists(real_beta, color='green', scale=0.25)
        plot_dists(reordered_input_beta, color='red', scale=0.25, labels=labels,
                   clear=False)

    return real_beta, input_beta, pairings, labels, male_distances
    
            
def tma(male_distances, female_distances, invert=False, verbose=False):
    """ run stable marriage algorithm
    Note: input is distances, not rankings (ie higher is more preferred)
        Use the invert flag to use distances as rankings
    """
    if invert:
        male_distances = -male_distances
        female_distances = -female_distances
    #rank from highest to lowest
    male_preferences = [sorted(range(len(male)), cmp=ind_cmp(male),
                               reverse=True)
                        for male in male_distances]
    female_preferences = [sorted(range(len(female)), cmp=ind_cmp(female),
                                 reverse=True)
                          for female in female_distances]
    if verbose:
        print "Male preference list:", male_preferences
        print "Female preference list:", female_preferences
    #each male crosses off his top female and proposes to her
    proposals = [male.pop(0) for male in male_preferences]
    if verbose: 
        print "Proposals:", proposals
    pairings = {}
    for i in range(len(proposals)):
        female = proposals[i]
        if pairings.has_key(female):
            pairings[female].append(i)
        else:
            pairings[female] = [i]

    while(len(pairings.keys()) != len(female_distances)):
        if verbose:
            print "Pairings:", pairings
        #at least one woman has more than one proposal
        rejected_men = []
        #find these women, then reject the men
        for female in pairings:
            if len(pairings[female]) > 1:
                if verbose:
                    print "Female", female, "has suitors:", pairings[female]
                ordered_men = sorted(pairings[female], 
                                     cmp=ind_cmp(female_distances[female]),
                                     reverse=True)
                #put favorite man on string, reject the rest
                pairings[female] = [ordered_men[0]]
                if verbose:
                    print "Female", female, 
                    print "says 'maybe' to Male", ordered_men[0]
                rejected_men += ordered_men[1:]
        #rejected men must now propose to the next female on his list
        if verbose:
            print "Rejections:", rejected_men
        for rejected_man in rejected_men:
            next_female = male_preferences[rejected_man].pop(0)
            if verbose:
                print "Male", rejected_man, 
                print "now proposes to Female", next_female
            if pairings.has_key(next_female):
                pairings[next_female].append(rejected_man)
            else:
                pairings[next_female] = [rejected_man]
    if verbose:
        print "Final pairings:", pairings
            
    return pairings

def ind_cmp(values):
    """returns a cmp function to sort a list of indices based on their values
       
        Args:
           values: a list (or dict) of values
        
        Returns:
            a function that compares two indices x,y by 
            cmp(values[x], values[y])
    """
    return lambda x,y: cmp(values[x], values[y])

def top_three_words(topic):
    top_three = sorted(range(len(topic)), cmp=ind_cmp(topic), reverse=True)[:3]
    top_three = zip(top_three, [round(topic[i], 2) for i in top_three])
    
    return top_three

"""for HLDA--still hacky"""
def display_children(tree, parent="root"):
    print "topic", tree.topic_number, "[parent:", str(parent) + "]"
    for child in tree.children:
        if len(child.children) == 0:
            print "topic", child.topic_number, "(leaf)", "[parent:", \
            str(tree.topic_number) + "]"
        else:
            display_children(child, tree.topic_number)
            
def stick_break(probs):
    probabilities = [probs[0]]
    for i in range(1, len(probs)):
        probabilities.append(probs[i] * np.product([(1 - p) \
                                                    for p in probs[:i]]))
    #add probability for "other"
    probabilities.append(1 - np.sum(probabilities))
    return probabilities

def show_path(tree, path_indices):
    topic_indices = [tree.topic_number]
    for index in path_indices:
        topic_indices.append(tree.children[index].topic_number)
        tree = tree.children[index]
    return topic_indices

def avg_sig_words(alpha, num, trials=1000):
    return np.average(get_sig_words(dirichlet([alpha]*num, trials)))

class Dirichlet_Test(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.d = np.zeros(len(alpha))
        self.total = 0
    def test(self, times):
        how_long = time.time()
        for i in range(times):
            self.d += dirichlet(self.alpha)
            self.total += 1
        print "time:", time.time() - how_long, "seconds"
    def reset(self):
        self.d = np.zeros(len(self.alpha))
        self.total = 0
    def __str__(self):
        if self.total == 0:
            return str(self.d)
        else:
            return str(self.d / self.total)
    def get_d(self):
        if self.total == 0:
            return self.d
        else:
            return self.d / self.total
