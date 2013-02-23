import argparse

import numpy as np

import util

def read_matrix(filename):
    """
    filename: name of file containing matrix to be read
    """
    try:
        return np.loadtxt(filename)
    except:
        print "loadtxt failed; reading points manually"
        with open(filename, 'r') as f:
            lines = [line.strip(' \n').split(' ') for line in f.readlines()]
        rows = []
        for line in lines:
            rows.append([float(num) for num in line])
        
        return np.array(rows)
    
def convert_mallet(filename):
    """
    takes mallet doc-topics and converts it to matrix form
    """
    with open(filename, 'r') as f:
        f.readline()
        lines = [line.strip(' \n').split(' ')[2:] for line in f.readlines()]
    rows = []
    for line in lines:
        rows.append([float(num) for num in line])
    doc_topics = []
    for row in rows:
        topic_dist = [0] * (len(row) / 2)
        i = 0
        while i < len(row):
            topic_dist[int(row[i])] = row[i + 1]
            i += 2
        doc_topics.append(topic_dist)
    
    return np.array(doc_topics)
    
def generate(topics, words, words_per_doc):
    num_docs = len(topics)
    word_cdfs = [util.get_cdf(topic) for topic in words]
    
    docs = []
    for i in range(num_docs):
        if i % 100 == 0:
            print "reached document", i
        num_words = util.poisson(words_per_doc)
        topic_dist = topics[i]
        topic_cdf = util.get_cdf(topic_dist)
        
        doc = []
        doc_topics = []
        for word in range(num_words):
            topic = util.sample(topic_cdf)
            doc.append(util.sample(word_cdfs[topic]))
            doc_topics.append(topic)
        docs.append(doc)
    
    return docs, doc_topics
    
def get_matrix(filename):
    if "Mallet" in filename:
        print "converting mallet output from file:", filename
        return convert_mallet(filename)
    else:
        print "reading matrix from file:", filename
        return read_matrix(filename)
    
def main():
    parser = argparse.ArgumentParser(description="Document generator that \
    takes a model. Default parameters are noted in parentheses.")
    parser.add_argument('-l', action="store", type=int, default=75, 
                        help="average number of words per document (75)")
    parser.add_argument('-a', action="store",
                        help="filename of topic distribution over documents \
                        (None)")
    parser.add_argument('-b', action="store",
                        help="filename of word distribution over topics \
                        (None)")
    args = parser.parse_args()
    
    if args.a == None:
        print "no alpha supplied (returning None)"
        return None
    if args.b == None:
        print "no beta supplied (returning None)"
        return None
    
    alpha_matrix = get_matrix(args.a)
    beta_matrix = read_matrix(args.b)
    
    docs, doc_topics = generate(alpha_matrix, beta_matrix, args.l)
    
    return docs, doc_topics, alpha_matrix, beta_matrix
    
if __name__ == '__main__':
    docs, doc_topics, a, b = main()
