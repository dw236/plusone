import numpy as np
from numpy.random.mtrand import dirichlet
from scipy.misc import factorial

def word_dist(topic_word, observed_word_freqs, test_word_prob, l, num_iterations = 20):
    """ Estimate the distribution of new words in a document.

    Arguments:
    topic_word -- A (vocabulary size) x (# topics) matrix; entries should be
        nonnegative and each column should sum to 1.
    observed_word_freqs -- A vector of integers: the number of times each
        non-held-out word appeared in the document.
    test_word_prob -- The probability that each word was held out.  (Same
        meaning as test_word_percent, but renamed to emphasize that it's a
        number between 0 and 1.)
    l -- the rate of the Poisson distribution of document lengths
    num_iterations -- The number of iterations of Gibbs sampling to perform.
    """

    num_topics = topic_word.shape[0]
    vocab_size = topic_word.shape[1]
    assert vocab_size == observed_word_freqs.shape[0]

    def sample_mask(n_observed, rate):
        """The mask indicates which words were held out: mask[w] is 1 if word w
        was included, and 0 if word w was held out.  This function samples the
        posterior mask distribution for a single word if words were held out
        with probability test_word_prob, this word would appear with Poisson
        rate rate, and it was observed to appear n_observed times.
        """
        if 0 < n_observed: return 1
        p_0 = 1 - test_word_prob
        p_1 = test_word_prob * rate ** n_observed * exp(- rate) / factorial(n_observed)
        if np.random.rand() * (p_0 + p_1) >= p_0:
            return 1
        else:
            return 0

    def sample_word_topic_counts(n_observed, topic_probs, mask):
        """Given that a word occurred n_observed times, it was held out (if
        mask is 0) or not held out (if mask is 1), and the row of the
        topic_word matrix is topic_probs (will not sum to 1 in general),
        samples from the posterior distribution on the number of times this
        word was generated from each topic.
        """
        return TODO

    # topic_dist is our current sample of the distribution of topics in the
    # document.  We initialize it to a uniform sample from the simplex.
    topic_dist = dirichlet(np.ones(num_topics))
    word_dist = np.dot(topic_word, topic_dist)
    for iteration in range(num_iterations):
        # Sample mask conditioned on topic_dist.
        mask = np.array(
            sample_mask(n_observed = observed_word_freqs[word],
                        rate = l * word_dist[word])
            for w in range(vocab_size)
        )

        # num_topic_word is a matrix of the same shape as topic_word, but with
        # a different interpretation.  Each entry (w, i) is a nonnegative
        # integer, and is the number of times word w was chosen from topic i.
        # The columns should sum to observed_word_freqs.
        #
        # Conditioned on topic_dist and mask, each column of num_topic_word is
        # multinomial (if mask is 1) or a collection of independent poissons
        # (if mask is 0).
        num_topic_word = np.transpose(np.array(tuple(
            sample_word_topic_counts(
                n_observed = observed_word_freqs[word],
                rates_by_topic = l * np.multiply(topic_word[word,:], topic_dist),
                mask = mask[word]
            )
            for word in range(vocab_size)
        )))
 
        # Sample topic_dist conditioned on num_topic_word.
        TODO()

        # Generate words.
        TODO()
