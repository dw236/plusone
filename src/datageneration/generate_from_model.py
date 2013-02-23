import argparse
import os
import pickle

import numpy as np

import util

def read_matrix(filename, log_form=False):
    """
    filename: name of file containing matrix to be read
    log_form: whether or not the data is in log form
    """
    try:
        matrix = np.loadtxt(filename)
        if log_form:
            return np.exp(matrix)
        else:
            return matrix
    except:
        print "loadtxt failed; reading points manually"
        with open(filename, 'r') as f:
            lines = [line.strip(' \n').split(' ') for line in f.readlines()]
        rows = []
        for line in lines:
            if log_form:
                rows.append([np.exp(float(num)) for num in line])
            else:
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
    doc_topics = []
    for i in range(num_docs):
        if i % 100 == 0:
            print "reached document", i
        num_words = util.poisson(words_per_doc)
        topic_dist = topics[i]
        topic_cdf = util.get_cdf(topic_dist)
        
        doc = []
        word_topics = []
        for word in range(num_words):
            topic = util.sample(topic_cdf)
            doc.append(util.sample(word_cdfs[topic]))
            word_topics.append(topic)
        docs.append(doc)
        doc_topics.append(word_topics)
    
    return docs, doc_topics
    
def get_matrix(filename):
    if "Mallet" in filename:
        print "converting mallet output from file:", filename
        return convert_mallet(filename)
    else:
        print "reading matrix from file:", filename
        return read_matrix(filename)
    
def write(docs, doc_topics, topics, words):
    dir = 'output'
    try:
        os.mkdir(dir)
    except OSError:
        pass
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
    with open(dir + '/results.pickle', 'w') as f:
        pickle.dump([docs, doc_topics, topics, words], f)

def main():
    parser = argparse.ArgumentParser(description="Document generator that \
    takes a model. Default parameters are noted in parentheses.")
    parser.add_argument('-w', action="store_true", default=False,
                        help="write flag (False)")
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
    print "read shape: [", np.shape(alpha_matrix), " ]"
    beta_matrix = read_matrix(args.b, log_form=True)
    print "read shape: [", np.shape(beta_matrix), " ]"
    
    docs, doc_topics = generate(alpha_matrix, beta_matrix, args.l)
    if args.w:
        print "writing to file..."
        write(docs, doc_topics, alpha_matrix, beta_matrix)
        print "done"
    
    return docs, doc_topics, alpha_matrix, beta_matrix
    
if __name__ == '__main__':
    docs, doc_topics, a, b = main()
