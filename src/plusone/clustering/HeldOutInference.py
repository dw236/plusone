import numpy as np
from numpy.random import multinomial
from numpy.random.mtrand import dirichlet
from scipy.misc import factorial

def word_dist(topic_strengths, topic_word, observed_word_freqs, test_word_prob, l, num_iterations = 20):
    """ Estimate the distribution of new words in a document.

    Arguments:
    topic_strengths -- The parameters to the Dirichlet distribution on topic
        distributions.  May be an array or a single number.  Set this to 1 for
        the uniform distribution.
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

    if 0 == np.ndim(topic_strengths):
        topic_strengths = topic_strengths * np.ones(num_topics)
    assert num_topics == len(topic_strengths)

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

    def sample_topic_freqs_by_word(n_observed, rates_by_topic, mask):
        """Given that a word occurred n_observed times, it was held out (if
        mask is 0) or not held out (if mask is 1), and the word was produced
        via topic i according to a Poisson with rate rates_by_topic[i], returns
        a sample from the posterior distribution on the number of times this
        word was generated from each topic.
        """
        if mask:
            # The word was not held out.
            return multinomial(n_observed, topic_probs / sum(topic_probs))
        else:
            # The word was held out.
            assert 0 == n_observed
            return np.array(tuple(poisson(r) for r in rates_by_topic))

    # topic_dist is our current sample of the distribution of topics in the
    # document.  We initialize it to a uniform sample from the simplex.
    topic_dist = dirichlet(np.ones(num_topics))
    word_dist = np.dot(topic_word, topic_dist)

    # The sum of all the word distributions we've sampled so far, excluding the
    # first, since it doesn't even depend on the observed word frequencies.
    word_dist_sum = np.zeroes(vocab_size)

    for iteration in range(num_iterations):
        # Sample mask conditioned on topic_dist.
        mask = np.array(
            sample_mask(n_observed = observed_word_freqs[word],
                        rate = l * word_dist[word])
            for w in range(vocab_size)
        )

        # topic_word_freqs is a matrix of the same shape as topic_word, but with
        # a different interpretation.  Each entry (w, i) is a nonnegative
        # integer, and is the number of times word w was chosen from topic i.
        # The columns should sum to observed_word_freqs.
        #
        # Conditioned on topic_dist and mask, each column of topic_word_freqs is
        # multinomial (if mask is 1) or a collection of independent poissons
        # (if mask is 0).
        topic_word_freqs = np.transpose(np.array(tuple(
            sample_topic_freqs_by_word(
                n_observed = observed_word_freqs[word],
                rates_by_topic = l * np.multiply(topic_word[word,:], topic_dist),
                mask = mask[word]
            )
            for word in range(vocab_size)
        )))
 
        # Sample topic_dist conditioned on topic_word_freqs.
        topic_dist = dirichlet(topic_strengths + np.sum(topic_word_freqs, 0))
        word_dist = np.dot(topic_word, topic_dist)

        # Generate words.
        word_dist_sum += word_dist

    return word_dist_sum / num_iterations
