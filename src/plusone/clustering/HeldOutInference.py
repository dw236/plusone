from math import exp
import numpy as np
from numpy.random import multinomial, poisson
from numpy.random.mtrand import dirichlet
from scipy.misc import factorial

def word_dist(topic_strengths, word_topic, observed_word_freqs, test_word_prob, l, num_iterations = 20):
    """ Estimate the distribution of new words in a document.

    Arguments:
    topic_strengths -- The parameters to the Dirichlet distribution on topic
        distributions.  May be an array or a single number.  Set this to 1 for
        the uniform distribution.
    word_topic -- A (vocabulary size) x (# topics) matrix; entries should be
        nonnegative and each column should sum to 1.
    observed_word_freqs -- A vector of integers: the number of times each
        non-held-out word appeared in the document.
    test_word_prob -- The probability that each word was held out.  (Same
        meaning as test_word_percent, but renamed to emphasize that it's a
        number between 0 and 1.)
    l -- the rate of the Poisson distribution of document lengths
    num_iterations -- The number of iterations of Gibbs sampling to perform.
    """

    vocab_size = word_topic.shape[0]
    num_topics = word_topic.shape[1]
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
            return multinomial(n_observed, rates_by_topic / sum(rates_by_topic))
        else:
            # The word was held out.
            assert 0 == n_observed
            return np.array(tuple(poisson(r) for r in rates_by_topic))

    # topic_dist is our current sample of the distribution of topics in the
    # document.  We initialize it to a uniform sample from the simplex.
    topic_dist = dirichlet(np.ones(num_topics))
    word_dist = np.dot(word_topic, topic_dist)

    # The sum of all the word distributions we've sampled so far, excluding the
    # first, since it doesn't even depend on the observed word frequencies.
    word_dist_sum = np.zeros(vocab_size)

    for iteration in range(num_iterations):
        # Sample mask conditioned on topic_dist.
        mask = np.array(tuple(
            sample_mask(n_observed = observed_word_freqs[word],
                        rate = l * word_dist[word])
            for word in range(vocab_size)
        ))

        # word_topic_freqs is a matrix of the same shape as word_topic, but with
        # a different interpretation.  Each entry (w, i) is a nonnegative
        # integer, and is the number of times word w was chosen from topic i.
        # The columns should sum to observed_word_freqs.
        #
        # Conditioned on topic_dist and mask, each column of word_topic_freqs is
        # multinomial (if mask is 1) or a collection of independent poissons
        # (if mask is 0).
        word_topic_freqs = np.array(tuple(
            sample_topic_freqs_by_word(
                n_observed = observed_word_freqs[word],
                rates_by_topic = l * np.multiply(word_topic[word,:], topic_dist),
                mask = mask[word]
            )
            for word in range(vocab_size)
        ))
 
        # Sample topic_dist conditioned on word_topic_freqs.
        topic_dist = dirichlet(topic_strengths + np.sum(word_topic_freqs, 0))
        word_dist = np.dot(word_topic, topic_dist)

        # Generate words.
        word_dist_sum += word_dist

    return word_dist_sum / num_iterations

if "__main__" == __name__:
    """Three examples.

    The first demonstration of the power of this inference method over ones not
    designed for held-out words.  We have the word-topic matrix:
        1/2 0 0
        1/2 0 0
        0   1 0
        0   0 1
    and a Poisson rate of 800.  We observe word counts of (100, 0, 0, 100).
    The explanation is that the middle two words were held out, the topic
    distribution is about (1/4, 5/8, 1/8), and so the word distributions are
    about (1/8, 1/8, 5/8, 1/8).  A method not aware of the Poisson rate, or
    not aware that words could be held out, wouldn't be able to figure out that
    the third word is present.

    The second example has word-topic matrix::
        1/6 0.3
        1/6   0
        1/6 0.2
    and a Poisson rate of 500.  We observe word counts of (200, 0, 200).

    The third is about untangling topics.  We have the word-topic matrix:
        1/4   0 1/6
        1/4 1/4 1/6
          0 1/4 1/6
        1/2   0   0
          0 1/2   0
          0   0 1/2
    and a Poisson rate of 9000.  We observe word counts of
    (1500, 1500, 1500, 0, 0, 0).
    This is consistent with the third topic having a much higher
    weight than the other two (and the last word being hold out).  So the topic
    distribution should be close to (0, 0, 1), and so the word distribution
    should be close to (1/6, 1/6, 1/6, 0, 0, 1/2).
    """
    wt0 = np.array([[0.5, 0, 0], [0.5, 0, 0], [0, 1, 0], [0, 0, 1]])
    print word_dist(topic_strengths = 1, word_topic = wt0, observed_word_freqs = np.array([100, 0, 0, 100]), test_word_prob = 0.2, l = 800)

    wt1 = np.array([[1.0/6, 0.3],
                    [1.0/6,   0],
                    [1.0/6, 0.2]])
    print word_dist(topic_strengths = 1, word_topic = wt1, observed_word_freqs = np.array([200, 0, 200]), test_word_prob = 0.2, l = 500, num_iterations=100)

    wt2 = np.array([[1.0/4,     0, 1.0/6],
                    [1.0/4, 1.0/4, 1.0/6],
                    [    0, 1.0/4, 1.0/6], 
                    [  0.5,     0,     0],
                    [    0,   0.5,     0],
                    [    0,     0,   0.5]])
    print word_dist(topic_strengths = 1, word_topic = wt2, observed_word_freqs = np.array([1500, 1500, 1500, 0, 0, 0]), test_word_prob = 0.2, l = 9000, num_iterations=100)
