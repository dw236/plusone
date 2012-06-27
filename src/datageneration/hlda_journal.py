"""
hierarchical LDA document generator (from journal version of paper: infinite 
trees)

Based on
"The Nested Chinese Restaurant Process and Bayesian Nonparametric Inference of 
Topic Hierarchies"
by Blei, Griffiths and Jordan
JACM 2010
"""

import argparse
from numpy.random import beta
from numpy.random.mtrand import dirichlet
import pickle
from random import random as rand
import util

class Topic_node:
    def __init__(self, params):
        # The word distribution of this node's topic.
        self.word_dist = dirichlet(params["topic_to_word_param"])
        self.word_cdf = util.get_cdf(self.word_dist)

        # The number of documents that pass through this node.
        self.num_documents = 0

        # Those children of this node which have looked below this level.
        # Documents that reached this node but never looked below aren't
        # represented here; this is okay because the Chinese Restaurant
        # Process is exchangeable (doesn't depend on order).
        self.children = []

        # The number of documents which looked below this level.  This
        # should always be equal to sum(c.num_documents for c in
        # self.children).
        self.num_documents_in_children = 0

    def pick_child(self, params):
        """
        Picks a child according to the Chinese Restaurant Process,
        updates that child's num_documents, and returns the index of
        the child in self.children.
        """
        r = rand() * self.num_documents_in_children + params["new_child_gamma"]

        for i in range(len(self.children)):
            child = self.children[i]
            # Should we use child #i?
            if r < child.num_documents:
                child.add_document()
                return i
            r -= child.num_documents

        # Add a new child.
        node = Topic_node(params)
        node.add_document()
        self.children.append(node)
        return len(self.children) - 1

    def add_document(self):
        self.num_documents += 1

    def dump(self, f):
        self.dump_indented(f, "", "")

    def dump_indented(self, f, first_prefix, rest_prefix):
        f.write(first_prefix)
        f.write(str(self.num_documents) + "; " + show_dist(self.word_dist) + 
                "\n")
        child_first_prefix = rest_prefix + "+"
        child_rest_prefix = rest_prefix + "|"
        for child in self.children:
            child.dump_indented(f, child_first_prefix, child_rest_prefix)
        f.write(rest_prefix + "\n")

def show_dist(dist):
    pairs = zip(dist, range(len(dist)))
    pairs.sort()
    pairs.reverse()
    return ", ".join(str(i) + ":" + str(p) for p, i in pairs)

def sample_topic_index(stay_probs, params):
    r = rand()
    level = 0
    while True:
        if len(stay_probs) <= level:
            new_stay_prob = params["parent_topic_bias_sample"]()
            stay_probs.append(new_stay_prob)
        stay_here_prob = stay_probs[level]
        if r < stay_here_prob:
            return level
        r = r * (1 - stay_here_prob)
        level += 1

def follow_path(path, path_indices, level, params):
    while len(path) <= level:
        tail = path[-1]
        child_index = tail.pick_child(params)
        path.append(tail.children[child_index])
        path_indices.append(child_index)
    return path[level]

def generate_one_doc_with_hlda(topic_root, params):
    # This document's path through the tree, generated as needed.
    # Each element is a Topic_node.
    path = [topic_root]
    topic_root.add_document()
    # path[i+1] is the path_indices[i]-th child of path[i].
    # We keep this around for debugging.
    path_indices = []
    # The probability at each node of using that node's topic,
    # conditioned on not using any of the parent nodes' topics.
    # Generated as needed.
    stay_probs = []
    num_words = params["words_per_doc_distribution"].sample()
    words = []
    for i in range(num_words):
        topic_level = sample_topic_index(stay_probs, params)
        topic_node = follow_path(path, path_indices, topic_level, params)
        word = util.sample(topic_node.word_cdf)
        words.append(word)
    return words, stay_probs, path_indices

def generate_docs_with_hlda(num_docs, words_per_doc, vocab_size, 
                            topic_to_word_beta, topic_dist_m, topic_dist_pi, 
                            new_child_gamma):
    params = {}
    params["topic_to_word_param"] = [topic_to_word_beta] * vocab_size
    params["words_per_doc_distribution"] = util.Poisson(words_per_doc)
    pta = topic_dist_m * topic_dist_pi
    ptb = topic_dist_pi - pta
    params["parent_topic_bias_sample"] = lambda: beta(pta, ptb)
    params["new_child_gamma"] = new_child_gamma
    topic_root = Topic_node(params)
    documents, topic_stay_probs, topic_paths = zip(*[generate_one_doc_with_hlda(topic_root, params) for i in range(num_docs)])
    return documents, topic_root, topic_stay_probs, topic_paths

def write(data, args):
    """writes the data generated by generate_docs to various files
    
    Writes three files, one containing the generated data, one containing the
    model used to generate the data, and one containing the options given at
    the command line. Also dumps to a pickle file for future reading.
    
    Returns:
        none, but writes two text files and one pickle file
    
    FILES
    -----
    documents-out:    
        file containing all of the generated data (raw data); each line is one
        document, and each entry is a word
    documents_options-out:
        file containing the exact command that was given to run this program 
        from the command line, ie "python documents.py [options...]"
        Note: does not end in a newline character
    results.pickle:
        file containing the documents and the topic tree
    """
    docs, tree, topic_stay_probs, topic_paths = data
    with open('output/hldaj-documents-out', 'w') as f:
        for doc in docs:
            for word in doc:
                f.write(str(word) + " ")
            f.write('\n')
    with open('output/hldaj-stay-probs-out', 'w') as f:
        for ps, indices in zip(topic_stay_probs, topic_paths):
            for p, i in zip(ps, [None] + indices):
                f.write(str(i) + ":" + str(p) + " ")
            f.write('\n')
    tree.dump(open('output/hldaj-tree-out', 'w'))
    with open('output/hldaj-documents_options-out', 'w') as f:
        f.write("python documents.py ")
        f.write("-n " + str(args.n) + " ")
        f.write("-l " + str(args.l) + " ")
        f.write("-m " + str(args.m) + " ")
        f.write("-b " + str(args.b) + " ")
        f.write("-z " + str(args.z) + " ")
        f.write("-p " + str(args.p) + " ")
        f.write("-g " + str(args.g) + " ")
    with open('output/hldaj-results.pickle', 'w') as f:
        pickle.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Document generator for \
    hierarchical LDA. Default parameters are noted in parentheses.")
    parser.add_argument('-w', action="store_true", default=False,
                        help="write flag (false)")
    parser.add_argument('-n', action="store", metavar='num_docs', type=int,
                        default=20, help="number of documents to generate (20)")
    parser.add_argument('-l', action="store", type=int, default=50, 
                        help="average number of words per document (50)")
    parser.add_argument('-m', action="store", type=int, default=30,
                        help="size of the vocabulary (30)")
    parser.add_argument('-b', action="store", metavar='beta', 
                        type=float, default=0.01, 
                        help="dirichlet parameter for words (0.01)")
    parser.add_argument('-z', action="store", type=float, default=0.25,
                        help="mean probability of not descending to a child")
    parser.add_argument('-p', action="store", type=float, default=1.0,
                        help="inflexibility in probability of not descending \
                        to a child")
    parser.add_argument('-g', action="store", type=float, default=1.0,
                        help="tendancy to create a new child \
                        (gamma in Chinese Restaurant Process)")
    
    args = parser.parse_args()
    
    print ""
    print "generating documents with parameters:"
    print "n    = ", args.n, "(number of documents)"
    print "l    = ", args.l, "(average number of words)"
    print "m    = ", args.m, "(size of vocabulary)"
    print "b    = ", args.b, "(word dirichlet)"
    print "z    = ", args.z, "(don't descend mean)"
    print "p    = ", args.p, "(don't descend inflexibility)"
    print "g    = ", args.g, "(new child gamma)"
    print ""
    
    data = generate_docs_with_hlda(
        num_docs = args.n, words_per_doc = args.l, vocab_size = args.m,
        topic_to_word_beta = args.b, topic_dist_m = args.z,
        topic_dist_pi = args.p, new_child_gamma = args.g)
    if args.w:
        print "writing data to file...",
        write(data, args)
        print "done"
    return data

if __name__ == '__main__':
    documents, topic_root, topic_stay_probs, topic_paths = main()
