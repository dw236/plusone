import src.datageneration.util as util
import argparse
import numpy as np
import pylab

import math
import sys

def showTopics(file='lda/trained/'):
    matrix = get_matrix(file + 'final.beta')
    for topicNum in range(0,len(matrix)):
        topic = np.power(math.e, matrix[topicNum])
        topicDist = (np.array(topic)[0].tolist())
        unsortedTopic = list(topicDist)
        topicDist.sort()
        topicDist.reverse()
        print
        print 'Topic ' + str(topicNum) + ':' 
        words = [line.strip() for line in open('data/wordMap.txt')]
        i = 0
        while i < 20:
            ithMostPopular = topicDist[i]
            for j in range(0, len(unsortedTopic)):
                if unsortedTopic[j] == ithMostPopular:
                    print words[j] + ' '
                    i += 1
"""
        firstK = util.get_sig_words([topicDist])[0]
        newDist = []
        total = 0
        for i in range(0, len(topicDist)):
            if i < firstK:
                total += topicDist[i]
                newDist += [topicDist[i]]
        newDist = [x/total for x in newDist]
    util.plot_dist(newDist)
    pylab.show()
"""

def showSigWords(file='lda/trained/'):
    matrix = get_matrix(file + 'final.beta')
    sigWordsPerTopic = []
    for topicNum in range(0,len(matrix)):
        topic = np.power(math.e, matrix[topicNum])
        topicDist = (np.array(topic)[0].tolist())
        topicDist.sort()
        topicDist.reverse()
        sigWordsPerTopic.append(util.get_sig_words([topicDist]))
    util.plot_dist(sigWordsPerTopic)
    pylab.show()

def showSigTopics(file='lda/trained/'):
    gamma = get_matrix(file + 'final.gamma')
    numDocs, numTopics = gamma.shape
    sigTopicsPerDoc = []
    for docNum in range(0, numDocs):
        doc = np.array(gamma)[docNum].tolist()
        doc.sort()
        doc.reverse()
        total = 0
        newDist = []
        for i in range(0, len(doc)):
            total += doc[i]
            newDist += [doc[i]]
        newDist = [x/total for x in newDist]
        cdf = 0
        numSigTopics = 0
        while (cdf < 0.8):
            cdf += newDist[numSigTopics]
            numSigTopics += 1
        sigTopicsPerDoc += [numSigTopics]
    util.plot(sigTopicsPerDoc)
    pylab.show()

def get_matrix(file):
    with open(file, 'r') as f:
        mat = []
        for line in f.readlines():
            mat.append([float(word) \
                for word in line.strip(' \n').split(' ')])
    ret = np.matrix(mat)
    return ret

def main():
    parser = argparse.ArgumentParser(description="Takes in the location of a \
    directory with a beta and gamma matrix and displays information about them")
    parser.add_argument('f', metavar='directory', action="store", 
                        help="path to the directory with matrices to be read")
    
    args = parser.parse_args()
    print "reading matrices from:", args.f
    
    showTopics(args.f)
    showSigWords(args.f)
    if args.f == 'lda/trained/':
        showSigTopics(args.f)

if __name__ == '__main__':
    results = main()
