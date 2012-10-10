import numpy as np
from numpy.random.mtrand import dirichlet

def word_dist(topic_word = None, observed_word_freqs = None, test_word_percent = None, l = None, num_iterations = 20):
    """ Estimate the distribution of new words in a document.

    fKeyword arguments:
    topic_word -- A (vocabulary size) x (# topics) matrix; entries should be
        nonnegative and each column should sum to 1.
    observed_word_freqs -- A vector of integers: the number of times each
        non-held-out word appeared in the document.
    test_word_percent -- The probability that each word was held out.
    l -- the rate of the Poisson distribution of document lengths
    num_iterations -- The number of iterations of Gibbs sampling to perform.
    """

    num_topics = topic_word.shape[0]
    vocab_size = topic_word.shape[1]
    assert vocab_size == observed_word_freqs.shape[0]

    # topic_dist is our current sample of the distribution of topics in the
    # document.  We initialize it to a uniform sample from the simplex.
    topic_dist = dirichlet(np.ones(num_topics))
    for iteration in range(num_iterations):
        # Estimate mask from topic_dist.
        TODO()

        # Estimate ntw.
        TODO()
 
        # Estimate topic_dist.
        TODO()

        # Generate words.
        TODO()
