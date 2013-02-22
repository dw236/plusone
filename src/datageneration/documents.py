import argparse
import pickle
import os
import itertools

import random
from random import random as rand
from random import sample as rsample

import numpy as np
from numpy.random.mtrand import dirichlet
from numpy.random.mtrand import multivariate_normal as N

import util
from util import *

def generate_docs(num_topics, num_docs, words_per_doc=50, vocab_size=30,
                  alpha=None, beta=None, noise=-1, plsi=False, ctm=False, 
                  pareto=False):
    """Generates documents according to plsi, ctm, or lda
    
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
        DISTRIBUTION PARAMETERS
        ---------------------
        depending on which model, alpha and beta are parameters to different
        distributions
        
        LDA: Assumes symmetric dirichlet distributions (ie all elements in the
        parameter vector have the same value)
            alpha: 
                parameter to dirichlet distribution for topics
            beta: 
                parameter to dirichlet distribution for words
            
        PLSI:
            alpha:
                parameter to poisson distribution to determine the number of
                topics per document (each topic will have uniform
                probability; all other topics will have probability 0)
            beta:
                as alpha, but poisson distribution instead controls the number
                of words per topic (each word will have uniform
                probability; all other words will have probability 0)
        ---------------------
        noise: 
            given as a probability; each word will be replaced with a random
            word with noise probability
        plsi:
            flag to draw distributions according to plsi (ie random 
            distributions)
        ctm:
            flag to draw distributions according to ctm (ie a multivariate
            gaussian distribution) 
        pareto:
            flag to make dirichlet distribution pareto (ie for the dirichlet
            parameter, set each alpha_i = alpha / alpha_i)
            
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
    #@TODO: integrate ctm parameters (ie mu and sigma) into alpha and beta
    mu = np.zeros(num_topics)
    sigma = np.ones((num_topics, num_topics))
    
    if plsi and ctm:
        print "plsi and ctm flags cannot both be active (returning None)"
        return None
    
    if not plsi and not ctm:
        if pareto:
            alpha = [alpha / i for i in range(1, num_topics + 1)]
            beta = [np.sqrt(beta / i) for i in range(1, vocab_size + 1)]
            #beta = [beta / i for i in range(1, vocab_size + 1)]
        else:
            alpha = [alpha] * num_topics
            beta = [beta] * vocab_size

    if plsi or ctm:
        sig_words = [rsample(range(vocab_size), util.poisson(beta, vocab_size))\
                     for t in range(num_topics)]
        word_dist = [np.zeros(vocab_size) for t in range(num_topics)]
        for i in range(num_topics):
            word_dist[i][sig_words[i]] = 1.0 / len(sig_words[i])
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
        if plsi:
            sig_topics = rsample(range(num_topics), 
                                 util.poisson(alpha, num_topics))
            topic_dist = np.zeros(num_topics)
            topic_dist[sig_topics] = 1.0 / len(sig_topics)
        elif ctm:
            eta = N(mu, sigma)
            topic_dist = np.exp(eta) / np.sum(np.exp(eta))
        else:
            topic_dist = dirichlet(alpha)
        num_words = util.poisson(words_per_doc)
        doc = []
        topic_dists.append(topic_dist)
        topic_cdf = get_cdf(topic_dist)
        topic_cdfs.append(topic_cdf)
        doc_topics = []
        for word in range(num_words):
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
    Lastly, writes files to help lda cheat (see util.write_cheats), then copies
    all files to outer-most directory for access by external methods.
    
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
            distribution by topic (averaged across topics)
            -sum of squares for each word distribution by topic (averaged
            across topics)
            -number of topics that constitute 80% of the cdf for each topic 
            distribution by document (averaged across documents)
            -sum of squares for each topic distribution by document (averaged
            across documents)
            -median of the cosine of pairwise word distributions (per topic)
        each value listed is separated a newline character
    results.pickle:
        file containing the documents, the topic*word distributions, and the
        document*topic distributions
    """
    docs, doc_topics, words, topics = data
    
    if args.plsi and args.ctm:
        print "plsi and ctm flags cannot both be active (returning None)"
        return None
    
    output_dir = 'output'
    try:
        os.mkdir(output_dir)
    except OSError:
        pass
    dir = output_dir + '/'
    dir += "k" + str(args.k) + "."
    dir += "n" + str(args.n) + "."
    dir += "l" + str(args.l) + "."
    dir += "m" + str(args.m) + "."
    dir += "a" + str(args.a) + "."
    dir += "b" + str(args.b) + "."
    if args.s != 0:
        dir += "s" + str(args.s) + "."
    if args.plsi:
        dir += "plsi"
    if args.ctm:
        dir += "ctm"
    if dir[-1] == '.':
        dir = dir[:-1]
    try:
        os.mkdir(dir)
    except OSError:
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
        f.write('sig_words ' + str(round(sig_words, 2)) + '\n')
        sum_squares_words = np.average([sum([word**2 for word in topic]) \
                                  for topic in words])
        f.write('sum_squares_words ' + str(round(sum_squares_words, 2)) + '\n')
        
        sig_topics = np.average(util.get_sig_words(topics))
        f.write('sig_topics ' + str(round(sig_topics, 2)) + '\n')
        sum_squares_topics = np.average([sum([topic**2 for topic in doc]) \
                                  for doc in topics])
        f.write('sum_squares_topics ' + str(round(sum_squares_topics, 
                                                  2)) + '\n')
        
        med = np.median([np.dot(i[0], i[1]) / (np.sqrt(np.dot(i[1], i[1])) * \
                                           np.sqrt(np.dot(i[0], i[0]))) \
                                for i in itertools.combinations(words, 2)])
        if round(med, 2) == 0:
            med = format(med, ".2e")
        else:
            med = round(med, 2)
        f.write('median ' + str(med) + '\n')
    with open(dir + '/results.pickle', 'w') as f:
        pickle.dump([docs, doc_topics, words, topics, args], f)
    if not args.plsi and not args.ctm:
        print "writing cheats for lda ...",
        util.write_cheats(data, args, dir)
    print "copying files to top-level folder ... ",
    os.system("cp " + dir + "/* output")
    print "done"
    print "archiving files for future use...",
    archive_base_dir = output_dir + '/archived/'
    try:
        os.mkdir(archive_base_dir)
    except:
        pass
    index = 0
    while True:
        if index == 0:
                ext = ''
        else:
            ext = "_" + str(index)
        archive_dir = archive_base_dir + dir[len(output_dir):] + ext
        try:
            os.mkdir(archive_dir)
            break
        except:
            index += 1
    os.system("cp " + dir + "/* " + archive_dir)
    print "done"

def main():
    parser = argparse.ArgumentParser(description="Document generator. Default\
    parameters are noted in parentheses.")
    parser.add_argument('-w', action="store_true", default=False,
                        help="write flag (false)")
    parser.add_argument('-k', action="store", metavar='num_topics', type=int,
                        default=15, help="number of latent topics (15)")
    parser.add_argument('-n', action="store", metavar='num_docs', type=int,
                        default=1000, help="number of documents to generate \
                        (1000)")
    parser.add_argument('-l', action="store", type=int, default=75, 
                        help="average number of words per document (75)")
    parser.add_argument('-m', action="store", type=int, default=1000,
                        help="size of the vocabulary (1000)")
    parser.add_argument('-a', action="store", metavar='alpha', 
                        type=float,  
                        help="parameter for topics (0.1 for lda, \
                        3 for plsi)")
    parser.add_argument('-b', action="store", metavar='beta', 
                        type=float, 
                        help="parameter for words (0.01 for lda, \
                        5 for plsi)")
    parser.add_argument('-s', action="store", metavar='noise', type=float, 
                        default=0, help="probability each word is generated\
                        randomly (0)")
    parser.add_argument('-plsi', action="store_true", default=False,
                        help="flag to use plsi instead of lda (false)")
    parser.add_argument('-ctm', action="store_true", default=False,
                        help="flag to use ctm instead of lda (false)")
    parser.add_argument('-p', action="store_true", default=False,
                        help="flag to make dirichlet pareto--must use\
                        lda (false)")
    
    args = parser.parse_args()
    
    if args.plsi and args.ctm:
        print "both -plsi and -ctm flags cannot be active (returning None)"
        return None
    
    if args.plsi:
        what_is_alpha = "(significant topics poisson parameter)"
        what_is_beta = "(significant words poisson parameter)"
        if args.a == None:
            args.a = 3
        if args.b == None:
            args.b = 5
    
    if not args.plsi and not args.ctm:
        what_is_alpha = "(topics dirichlet parameter)"
        what_is_beta = "(words dirichlet parameter)"
        if args.a == None:
            args.a = 0.1
        if args.b == None:
            args.b = 0.01
    
    print ""
    print "generating documents with parameters:"
    print "k    =  ", args.k, "(number of topics)"
    print "n    =  ", args.n, "(number of documents)"
    print "l    =  ", args.l, "(average number of words)"
    print "m    =  ", args.m, "(size of vocabulary)"
    if not args.plsi and not args.ctm:
        print "a    =  ", args.a, what_is_alpha
        print "b    =  ", args.b, what_is_beta
    print "s    =  ", args.s, "(noise probability)"
    print "plsi =  ", args.plsi, "(whether to draw from plsi)"
    print "ctm  =  ", args.ctm, "(whether to draw from ctm)"
    print "pareto= ", args.p, "(whether to make alpha pareto)" 
    print ""
    
    if args.s == 0:
        noise = -1
    else:
        noise = args.s
    
    data = generate_docs(args.k, args.n, args.l, args.m, args.a, args.b,
                         noise=noise, plsi=args.plsi, ctm=args.ctm, 
                         pareto=args.p)
    if args.w:
        print "writing data to file...",
        write(data, args)
        print "done"
    return data, args

if __name__ == '__main__':
    (docs, doc_topics, words, topics), args = main()
