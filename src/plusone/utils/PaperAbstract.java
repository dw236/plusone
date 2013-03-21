package plusone.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;

import plusone.Main;
import plusone.utils.TrainingPaper;
import plusone.utils.PredictionPaper;

/**
 * A <code>PaperAbstract</code> instance stores certain information about a
 * document in a text corpus.  It stores the number of times each word appears
 * in the document, and a list of incoming references from other documents and
 * outgoing references to other documents (for example, these could represent
 * citations).
 *
 * Each PaperAbstract must be given a unique index, passed to the constructor;
 * re-using indices results in undefined behavior.
 *
 * A <code>PaperAbstract</code> instance also stores information used for
 * training and evaluation tasks.  After being constructed, a
 * <code>PaperAbstract</code>'s <code>generateTf</code> (or
 * <code>generateTagTf</code>) method must be called, which is passed two
 * important parameters: a boolean indicating whether this
 * <code>PaperAbstract</code> instance should be part of the testing set or the
 * training set, and in the former case, a number <code>testWordpercent</code>
 * which is a fraction of vocabulary words (or tags, in the case of
 * <code>generateTagTf</code>) to hold out from the document for testing
 * purposes.
 *
 * Once <code>generateTf</code> has been called, the two methods
 * <code>getTrainingWords</code>, <code>getTrainingTf</code>,
 * <code>getTestingWords</code> and <code>getTestingTf</code> can be used to
 * access the non-held-out and held-out words.
 */
public class PaperAbstract implements TrainingPaper, PredictionPaper {
	private final Integer index;
	private final Integer[] inReferences;
	private final Integer[] outReferences;

	private Map<Integer, Integer> trainingTf;
	private Map<Integer, Integer> testingTf;
	private Map<Integer, Integer> tf;
	private double norm;
	private int group = 0; // Group # for cross-validation

	public PaperAbstract(int index, Integer[] inReferences,
			Integer[] outReferences, Integer[] abstractWords) {
		this.index = index;
		this.inReferences = inReferences;
		this.outReferences = outReferences;

		tf = new HashMap<Integer, Integer>();
		for (Integer word : abstractWords) {
			if (!tf.containsKey(word))
				tf.put(word, 0);
			tf.put(word, tf.get(word) + 1);
		}
	}

	/**
	 * Constructs a paper abstract.
	 *
	 * Before using a PaperAbstract in clustering methods, its generateTf
	 * (or generateTagTf) method must be called.
	 *
	 * @param index  The index of this paper within a corpus.  Note that
	 *            two paper abstracts are considered equal if they have the
	 *            same index.
	 * @param inReferences  An array of indices of papers that refer to this
	 *            paper.
	 * @param outReferences  An array of indices of papers that this paper
	 *            refers to.
	 * @param tf  The number of times every word occurs the document.  Keys
	 *            are words and values are numbers of occurrences.
	 */
	public PaperAbstract(int index, Integer[] inReferences,
			Integer[] outReferences, Map<Integer, Integer> tf) {
		this.index = index;
		this.inReferences = inReferences;
		this.outReferences = outReferences;
		this.tf = tf;
	}

	/**
	 * Set a group number; used for cross-validation.
	 */
	public void setGroup(int gp) {
		this.group = gp;
	}

	/**
	 * Get the group number most recently set with <code>setGroup()</code>;
	 * used for cross-validation.
	 */
	public int getGroup() {
		return group;
	}

	/**
	 * Generates the training term frequences for a document, and the testing
	 * term frequencies if the document is a test document.
	 *
	 * For a training document (<code>test == false</code>), the training term
	 * frequencies are the same frequencies passed to the constructor.
	 *
	 * For a testing document (<code>test == true</code>), each word in the
	 * vocabulary is independently randomly chosen to be either held out
	 * (probablity <code>testWordPercent</code>) or not.  If a word is held out,
	 * its training tf is zero and its testing tf is the tf.  Otherwise, the
	 * testing tf is zero and the training tf is the tf.
	 *
	 * This function must be called before we can use this paper in clustering
	 * methods.
	 *
	 * @param testWordPercent  The probability with which to hold out a word.
	 * @param terms  An array of Terms.Term objects.  If this is not
	 *            <code>null</code>, then this method updates terms' total
	 *            (training) count and list of testing or training documents,
	 *            where appropriate.
	 */
	public void generateTf(double testWordpercent, Terms.Term[] terms,
			boolean test) {
		Random randGen = Main.getRandomGenerator();
		trainingTf = new HashMap<Integer, Integer>();
		testingTf = test ? new HashMap<Integer, Integer>() : null;
		norm = 0.0;

		for (Integer word : tf.keySet()) {
			if (terms != null)
				terms[word].addDoc(this, test);
			
			int freq = tf.get(word);
			if (test && randGen.nextDouble() < testWordpercent) {
				testingTf.put(word, freq);
			} else {
				trainingTf.put(word, freq);
				norm += freq * freq;
				if (!test && terms != null)
					terms[word].totalCount += tf.get(word);
			}
		}
		if (test && testingTf.isEmpty()) {
			System.err.println("WARNING: Test document is empty");
		}
		for (Map.Entry<Integer, Integer> entry : trainingTf.entrySet()) {
			norm += entry.getValue() * entry.getValue();
		}
		norm = Math.sqrt(norm);
	}
	
	/**
	 * A specialized version of generateTf which should be called instead of
	 * generateTf for test documents with tagged data.  It ensures that words
	 * that are not tags are never held out; words that are tagged are still
	 * held out with probability <code>testWordPercent</code>.
	 *
	 * For a training document with tags, the <code>generateTf</code> method
	 * should be called instead of this.
	 * 
	 * @param myTags  The words which are tags and can be held out.
	 */
	public void generateTestingTagTf(List<Integer> myTags, double testWordpercent, Terms.Term[] terms) {
		trainingTf = new HashMap<Integer, Integer>();
		testingTf = new HashMap<Integer, Integer>();
		Random randGen = Main.getRandomGenerator();
		norm = 0.0;
		
		for (Integer word : tf.keySet()) {
			if (terms != null)
				terms[word].addDoc(this, true);
			
			int freq = tf.get(word);
			if (myTags.contains(word) && randGen.nextDouble() < testWordpercent) {
				testingTf.put(word, freq);
			} else {
				trainingTf.put(word, freq);
				norm += freq * freq;
			}
		}
		
		for (Map.Entry<Integer, Integer> entry : trainingTf.entrySet()) {
			norm += entry.getValue() * entry.getValue();
		}
		
		norm = Math.sqrt(norm);
	}
	
	/**
	 * Returns the number of occurrences of <code>word</code> in this
	 * documesnt, unless this is a testing document and the word was held
	 * out, in which case this returns <code>0</code>.
	 */
	public Integer getTrainingTf(Integer word) {
		return trainingTf == null ? 
				0 : (trainingTf.containsKey(word) ? trainingTf.get(word) : 0);
	}

	public SimpleMatrix getTrainingTfAsSimpleMatrixRow(int vocabSize) {
		SimpleMatrix ret = new SimpleMatrix(1, vocabSize);
		for (int word = 0; word < vocabSize; ++word)
			ret.set(0, word, getTrainingTf(word));
		return ret;
	}

	/**
	 * Returns the set of all words that occur at least once in the document
	 * and were not held out.  (For a training document, this means all
	 * words.)
	 */
	public Set<Integer> getTrainingWords() {
		return trainingTf.keySet();
	}

	/**
	 * If this is a testing document and <code>word</code> was held out,
	 * returns the number of times <code>word</code> occurrs in the
	 * document.  Otherwise returns <code>0</code>.
	 */
	public Integer getTestingTf(Integer word) {
		return testingTf == null ? 
				0 : (testingTf.containsKey(word) ? testingTf.get(word) : 0);
	}

	/**
	 * Returns the set of words that were held out from this document.  This
	 * should only be called if this is a testing document.
	 */
	public Set<Integer> getTestingWords() {
		return testingTf.keySet();
	}

	/**
	 * Returns this document's unique index (the <code>index</code>
	 * parameter passed to the constructor).
	 */
	public Integer getIndex() {
		return index;
	}

	/**
	 * Returns an array of indices of papers that refer to this paper (the
	 * <code>inReferences</code> parameter passed to the constructor).
	 */
	public Integer[] getInReferences() {
		return inReferences;
	}

	/**
	 * Returns an array of indices of papers this paper refers to (the
	 * <code>outReferences</code> parameter passed to the constructor).
	 */
	public Integer[] getOutReferences() {
		return outReferences;
	}

	/**
	 * The L2 norm of the vector of training word frequencies (those
	 * returned by <code>getTrainingTf</code>).
	 *
	 * <code>generateTf</code> or <code>generateTestingTagTf</code> must be
	 * called before this method.
	 */
	public double getNorm() {
		return norm;
	}

	/**
	 * Returns the value of the <code>test</code> parameter passed to
	 * <code>generateTf</code> or <code>generateTagTf</code>.  This
	 * indicates whether this is considered a testing document: that is,
	 * whether words are held out from the return values of
	 * <code>getTrainingWords</code> or <code>getTrainingTf</code>.
	 */
	public boolean isTest() {
		return testingTf != null;
	}

	/**
	 * Returns the dot product of the L2-normalized word frequency vectors of
	 * this <code>PaperAbstract</code> instance with another, not including
	 * held-out words.  (The L2 normalization is done after removing the
	 * held-out words.)
	 *
	 * <code>generateTf</code> or <code>generateTestingTagTf</code> must be
	 * called before this method.
	 */
	public double similarity(PaperAbstract a) {
		double sim = 0.0;

		for (Map.Entry<Integer, Integer> entry : trainingTf.entrySet()) {
			int wordId = entry.getKey();
			int count = entry.getValue();
			sim += count * a.getTrainingTf(wordId);
		}
		return sim / (a.getNorm() * norm);
	}

	/**
	 * Two paper abstracts are considered equal iff they have the same
	 * index.  (So it's a good idea to make indices unique.)
	 */
	public boolean equals(Object obj) {
		return obj instanceof PaperAbstract
				&& this.index == ((PaperAbstract) obj).index;
	}

	/**
	 * Given a list of documents, returns a map which maps a word to the number
	 * of training documents in which that word is present and not held out.
	 */
	public static Map<Integer, Integer> getCombinedTf(List<TrainingPaper> lst) {
		Map<Integer, Integer> tf = new HashMap<Integer, Integer>();
		for (TrainingPaper trainPaper : lst) {
			for (Integer wd : trainPaper.getTrainingWords()) {
				if (!tf.containsKey(wd))
					tf.put(wd, 0);
				tf.put(wd, tf.get(wd) + 1);
			}
		}
		return tf;
	}

	/**
	 * Given an array <code>freqs</code> of frequencies (or each frequency in a
	 * separate argument), returns a map mapping an index <code>i</code> to
	 * <code>freqs[i]</code>.
	 */
	public static Map<Integer, Integer> freqMap(int ... freqs) {
		Map<Integer, Integer> freqMap = new HashMap<Integer, Integer>();
		for (int i = 0; i < freqs.length; ++i) {
			if (freqs[i] > 0) freqMap.put(i, freqs[i]);
		}
		return freqMap;
	}
}
