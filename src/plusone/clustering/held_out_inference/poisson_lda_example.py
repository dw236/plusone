from poisson_lda import estimate_word_dist
import numpy as np

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
    print estimate_word_dist(topic_strengths = np.ones(3), word_topic = wt0, observed_word_freqs = np.array([100, 0, 0, 100]), test_word_prob = 0.2, l = 800)

    wt1 = np.array([[1.0/6, 0.3],
                    [1.0/6,   0],
                    [1.0/6, 0.2]])
    print estimate_word_dist(topic_strengths = np.ones(2), word_topic = wt1, observed_word_freqs = np.array([200, 0, 200]), test_word_prob = 0.2, l = 600, num_iterations=100)

    wt2 = np.array([[1.0/4,     0, 1.0/6],
                    [1.0/4, 1.0/4, 1.0/6],
                    [    0, 1.0/4, 1.0/6], 
                    [  0.5,     0,     0],
                    [    0,   0.5,     0],
                    [    0,     0,   0.5]])
    print estimate_word_dist(topic_strengths = np.ones(3), word_topic = wt2, observed_word_freqs = np.array([1500, 1500, 1500, 0, 0, 0]), test_word_prob = 0.2, l = 9000, num_iterations=100)
