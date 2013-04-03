# Goal: a simple example where methods that don't properly account for the fact
# that vocabulary words are held out will mess up.

# TODO:  Accept command line arguments: for example, random seed and output path.

import numpy as np

eps = 0.1
word_topic = np.matrix([[1.0 - eps, 0.5 * eps, 0.5 * eps], [0, 0.75, 0.25]]).transpose()
(vocab_size, num_topics) = word_topic.shape

def gen_docs(num_docs, doc_length):
    for i in xrange(num_docs):
        topics = np.random.dirichlet(np.ones(num_topics))
        word_probs = word_topic * np.matrix(topics).transpose()
        word_probs_array = word_probs.getA1()
        yield np.random.multinomial(doc_length, word_probs_array)

if "__main__" == __name__:
    for doc in gen_docs(1000, 100):
        for word in xrange(vocab_size):
            for i in xrange(doc[word]):
                print word,
        print
