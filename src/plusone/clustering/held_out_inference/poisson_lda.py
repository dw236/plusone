import  argparse
from    math                import exp
import  numpy               as np
from    numpy.random        import multinomial, poisson
from    numpy.random.mtrand import dirichlet
from    scipy.misc          import factorial
import  sys
from    sys                 import stdin

def estimate_word_dist(topic_strengths, word_topic, observed_word_freqs,
                       test_word_prob, l, num_iterations = 20):
    """ Estimate the distribution of new words in a document.

    Arguments:
    topic_strengths -- The parameters to the Dirichlet distribution on topic
        distributions.  Set this to an array of ones for the uniform
        distribution.
    word_topic -- A (vocabulary size) x (# topics) matrix; entries should be
        nonnegative and each column should sum to 1.
    observed_word_freqs -- A vector of integers: the number of times each
        non-held-out word appeared in the document.
    test_word_prob -- The probability that each word was held out.  (Same
        meaning as test_word_percent, but renamed to emphasize that it's a
        number between 0 and 1.)
    l -- the rate of the Poisson distribution of document lengths
    num_iterations -- The number of iterations of Gibbs sampling to perform.

    Returns:
    A numpy array containing the estimated word distribution.
    """

    vocab_size = word_topic.shape[0]
    num_topics = word_topic.shape[1]
    assert vocab_size == observed_word_freqs.shape[0]
    assert num_topics == len(topic_strengths)

    def sample_mask(n_observed, rate):
        """The mask indicates which words were held out: mask[w] is 1 if word w
        was included, and 0 if word w was held out.  This function samples the
        posterior mask distribution for a single word if words were held out
        with probability test_word_prob, this word would appear with Poisson
        rate rate, and it was observed to appear n_observed times.
        """
        if 0 < n_observed: return 1
        p_0 = test_word_prob
        p_1 = (1 - test_word_prob) * exp(- rate)
        if np.random.rand() * (p_0 + p_1) >= p_0:
            return 1
        else:
            return 0

    def sample_topic_freqs(word_topic_rates, mask):
        """Given observed_word_freqs, that each word was held out (if mask is
        0) or not held out (if mask is 1), and each word w was produced via
        topic i according to a Poisson with rate word_topic_rates[w, i],
        returns a sample from the posterior distribution on the number of times
        a word was generated from each topic.
        """

        # First, sample the contribution from words where mask was 0.  Each
        # such word is sampled from a poisson distribution.  The sum of poisson
        # distributions is poisson, so we only need to sample one poisson for
        # each topic.
        topic_freqs = \
            poisson(np.sum((1 - mask)[:, np.newaxis] * word_topic_rates, 0))

        # For each word where mask was 1, we sample from a multinomial
        # distribution with probabilities proportional to the word-topic rates
        # for that word.  For efficiency, we skip over words that didn't occur
        # at all: multinomial_words is an array of the indices of words for
        # which we do need to take a sample.
        multinomial_words = np.arange(vocab_size)[
            np.array(mask * observed_word_freqs, dtype = bool)]
        for word in multinomial_words:
            topic_rates = word_topic_rates[word, :]
            topic_freqs += multinomial(observed_word_freqs[word],
                                       topic_rates / sum(topic_rates))
        return topic_freqs

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

        word_topic_rates = l * np.multiply(word_topic, topic_dist)

        # word_topic_freqs is a matrix of the same shape as word_topic, but with
        # a different interpretation.  Each entry (w, i) is a nonnegative
        # integer, and is the number of times word w was chosen from topic i.
        # The columns should sum to observed_word_freqs.
        #
        # Conditioned on topic_dist and mask, each column of word_topic_freqs is
        # multinomial (if mask is 1) or a collection of independent poissons
        # (if mask is 0).
        topic_freqs = sample_topic_freqs(word_topic_rates = word_topic_rates, mask = mask)
 
        # Sample topic_dist conditioned on word_topic_freqs.
        topic_dist = dirichlet(topic_strengths + topic_freqs)
        word_dist = np.dot(word_topic, topic_dist)

        # Generate words.
        word_dist_sum += word_dist

    return word_dist_sum / num_iterations

def string_to_float_list(s):
    return map(float, s.split())

def read_array_line(line):
    return np.array(string_to_float_list(line))

def read_2d_array_lines(lines):
    return np.array(map(string_to_float_list, lines))

def show_array_line(a):
    return " ".join(map(str, a.tolist()))

if "__main__" == __name__:
    arg_parser = argparse.ArgumentParser(description = """
        Estimates word distributions using a model with Poisson-length documents
        with an LDA model for words, given the topic strengths and word-topic
        matrix.  Standard input should have 1+k+m lines, where k is the number
        of topics and m is the number of documents.  The first line should
        be the topic strength vector, as numbers separated by spaces.  k is
        inferred from this first line.  The next k lines should be the
        topic-word matrix (one line for each topic).  The next m lines should be
        the observed word frequencies of documents.  Every time this script reads
        one of those lines, it will output an estimated word distribution before
        reading the next line.""")
    arg_parser.add_argument("--test_word_prob", required = True, type = float,
        help = "How likely each word is to be held out as a test word.")
    arg_parser.add_argument("--lambda", dest = "l", required = True,
        type = float,
        help = "The parameter of the document length Poisson distribution.")
    arg_parser.add_argument("--num_iterations", required = True, type = int,
        help = "The number of iterations of Gibbs sampling to perform.")
    args = arg_parser.parse_args()
    topic_strengths = read_array_line(stdin.readline())
    num_topics = topic_strengths.size
    word_topic = np.transpose(read_2d_array_lines(
        stdin.readline() for i in range(num_topics)))
    while True:
        line = stdin.readline()
        if "" == line: break
        observed_word_freqs = read_array_line(line)
        word_dist = estimate_word_dist(
            topic_strengths = topic_strengths,
            word_topic = word_topic,
            observed_word_freqs = observed_word_freqs,
            test_word_prob = args.test_word_prob,
            l = args.l,
            num_iterations = args.num_iterations)
        print show_array_line(word_dist)
        sys.stdout.flush()
