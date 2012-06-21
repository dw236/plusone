import argparse
import pickle
import os

import random
from random import random as rand
from random import sample as rsample

import numpy as np
from numpy.random.mtrand import dirichlet

import util
from util import *

def generate_docs(num_topics, num_docs, words_per_doc=50, vocab_size=30,
                  alpha=0.001, beta=0.01, noise=-1, plsi=False):
    """Generates documents according to plsi or lda
    
    Args:
        num_topics: 
            the number of underlying latent topics
        num_docs: 
            the number of documents to generate
        words_per_doc: 
            parameter to a Poisson distribution;
            determines the average words in a documents
        vocab_size: 
            the number of words in the vocabulary
        DIRICHLET PARAMETERS
        ---------------------
        Assumes symmetric dirichlet distributions (ie all elements in the
        parameter vector have the same value)
        ---------------------
        alpha: 
            parameter to dirichlet distribution for topics
        beta: 
            parameter to dirichlet distribution for words
        noise: 
            given as a probability; each word will be replaced with a random
            word with noise probability
        plsi:
            flag to determine which distribution to draw from,
            a random distribution or a sample from a dirichlet distribution
            
    Returns:
        docs:
            the list of documents, each a list of words (represented by their
            indices in range(vocab_size)
        topics:
            a list of documents, each a list of topics (represented by their
            indices in range(num_topics)
        word_dist:
            the distribution over words for each topic; 
            each row is the distribution for a different topic 
        topics_dist:
            the distribution over topics for each document;
            each row is the distribution for a different document
    """
    p = Poisson(words_per_doc)
    
    alpha = [alpha] * num_topics
    beta = [beta] * vocab_size

    if plsi:
        word_dist = [normalize([rand() for w in range(vocab_size)])
                     for t in range(num_topics)]
    else:
        word_dist = [dirichlet(beta) for i in range(num_topics)]
    word_cdfs = []
    for topic in word_dist:
        word_cdfs.append(get_cdf(topic))
    
    topic_cdfs = []
    docs = []
    topics = []
    topic_dists = []
    doc_index = 0
    for i in range(num_docs):
        if doc_index % 100 == 0:
            print "reached document", doc_index
        words_per_doc = p.sample()
        doc = []
        if plsi:
            topic_dist = normalize([rand() for t in range(num_topics)])
        else:
            topic_dist = dirichlet(alpha)
        topic_dists.append(topic_dist)
        topic_cdf = get_cdf(topic_dist)
        topic_cdfs.append(topic_cdf)
        doc_topics = []
        for word in range(words_per_doc):
            if rand() < noise:
                doc.append(rsample(range(vocab_size), 1))
                doc_topics.append(-1)
            else:
                topic = sample(topic_cdf)
                doc.append(sample(word_cdfs[topic]))
                doc_topics.append(topic)
        docs.append(doc)
        topics.append(doc_topics)
        doc_index += 1
    return docs, topics, word_dist, topic_dists

def write(data, args):
    """writes the data generated by generate_docs to various files
    
    Writes four files, one containing the generated data, one containing the
    model used to generate the data, one containing the options given at
    the command line, and one containing various statistics from the data 
    (see the file description below for details). 
    Also dumps to a pickle file for future reading in python.
    The files can be found in a directory with the options used to generate the
    data.
    
    Returns:
        none, but writes four text files and one pickle file
    
    FILES
    -----
    documents-out:    
        file containing all of the generated data (raw data); each line is one
        document, and each entry is a word
    documents_model-out:
        file containing the topic distributions for each document and the beta
        matrix; the two are separated by the character 'V' on its own line
    documents_options-out:
        file containing the exact command that was given to run this program 
        from the command line, ie "python documents.py [options...]"
        Note: does not end in a newline character
    documents_other-out:
        file containing the following: 
            -number of words that constitute 80% of the cdf for each word 
            distribution by topic (separated by spaces)
            -sum of squares for each topic distribution (separated by spaces)
            -number of documents
            -words per document (separated by spaces)
        each value listed is separated a newline character
    results.pickle:
        file containing the documents, the topic*word distributions, and the
        document*topic distributions
    """
    docs, doc_topics, words, topics = data
    
    dir = 'output/'
    dir += "k" + str(args.k) + "."
    dir += "n" + str(args.n) + "."
    dir += "l" + str(args.l) + "."
    dir += "m" + str(args.m) + "."
    if not args.plsi:
        dir += "a" + str(args.a) + "."
        dir += "b" + str(args.b) + "."
    if args.s != 0:
        dir += "s" + str(args.s) + "."
    if args.plsi:
        dir += "plsi"
    if dir[-1] == '.':
        dir = dir[:-1]
    try:
        os.mkdir(dir)
    except:
        print "overwriting existing data in directory:", dir, "...",
    
    with open(dir + '/documents-out', 'w') as f:
        for doc in docs:
            for word in doc:
                f.write(str(word) + " ")
            f.write('\n')
    with open(dir + '/documents-topics-out', 'w') as f:
        for topic_list in doc_topics:
            for topic in topic_list:
                f.write(str(topic) + " ")
            f.write('\n')
    with open(dir + '/documents_model-out', 'w') as f:
        for topic in words:
            for word in topic:
                f.write(str(word) + " ")
            f.write('\n')
        f.write('V\n')
        for doc in topics:
            for topic in doc:
                f.write(str(topic) + " ")
            f.write('\n')
    with open(dir + '/documents_options-out', 'w') as f:
        f.write("python documents.py ")
        f.write("-k " + str(args.k) + " ")
        f.write("-n " + str(args.n) + " ")
        f.write("-l " + str(args.l) + " ")
        f.write("-m " + str(args.m) + " ")
        if not args.plsi:
            f.write("-a " + str(args.a) + " ")
            f.write("-b " + str(args.b) + " ")
        if args.s != 0:
            f.write("-s " + str(args.s) + " ")
        if args.plsi:
            f.write(str("-plsi"))
    with open(dir + '/documents_other-out', 'w') as f:
        sig_words = np.average(util.get_sig_words(words))
        f.write(str(sig_words) + '\n')
        sum_squares = np.average([sum([topic**2 for topic in doc]) \
                                  for doc in topics])
        f.write(str(sum_squares) + '\n')
        f.write(str(len(docs)) + '\n')
        f.write(str(np.average([len(doc) for doc in docs])) + '\n')
    with open(dir + '/results.pickle', 'w') as f:
        pickle.dump([docs, doc_topics, words, topics, args], f)
    os.system("cp " + dir + "/* output")
    if not args.plsi:
        print "writing cheats for lda...",
        util.write_cheats(data, args.a)

def main():
    parser = argparse.ArgumentParser(description="Document generator. Default\
    parameters are noted in parentheses.")
    parser.add_argument('-w', action="store_true", default=False,
                        help="write flag (false)")
    parser.add_argument('-k', action="store", metavar='num_topics', type=int,
                        default=4, help="number of latent topics (4)")
    parser.add_argument('-n', action="store", metavar='num_docs', type=int,
                        default=20, help="number of documents to generate (20)")
    parser.add_argument('-l', action="store", type=int, default=50, 
                        help="average number of words per document (50)")
    parser.add_argument('-m', action="store", type=int, default=30,
                        help="size of the vocabulary (30)")
    parser.add_argument('-a', action="store", metavar='alpha', 
                        type=float, default=0.1, 
                        help="dirichlet parameter for topics (0.1)")
    parser.add_argument('-b', action="store", metavar='beta', 
                        type=float, default=0.01, 
                        help="dirichlet parameter for words (0.01)")
    parser.add_argument('-s', action="store", metavar='noise', type=float, 
                        default=0, help="probability each word is generated\
                        randomly (0)")
    parser.add_argument('-plsi', action="store_true", default=False,
                        help="flag to use plsi instead of lda (false)")
    
    args = parser.parse_args()
    
    print ""
    print "generating documents with parameters:"
    print "k    = ", args.k, "(number of topics)"
    print "n    = ", args.n, "(number of documents)"
    print "l    = ", args.l, "(average number of words)"
    print "m    = ", args.m, "(size of vocabulary)"
    if not args.plsi:
        print "a    = ", args.a, "(topics dirichlet parameter)"
        print "b    = ", args.b, "(words dirichlet parameter)"
    print "s    = ", args.s, "(noise probability)"
    print "plsi = ", args.plsi, "(whether to draw from plsi or lda model)"
    print ""
    
    if args.s == 0:
        noise = -1
    else:
        noise = args.s
    
    data = generate_docs(args.k, args.n, args.l, args.m, args.a, args.b,
                         noise=noise, plsi=args.plsi)
    if args.w:
        print "writing data to file...",
        write(data, args)
        print "done"
    return data, args

if __name__ == '__main__':
    (docs, doc_topics, words, topics), args = main()
