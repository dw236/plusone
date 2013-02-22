package plusone.clustering;

import java.util.List;

import plusone.clustering.ClusteringTest;
import plusone.utils.PaperIF;
import plusone.utils.PredictionPaper;
import plusone.utils.RunInfo;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;

import cc.mallet.topics.HierarchicalLDAWithPrediction;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;

public class MalletHldaPrediction extends ClusteringTest {
	final int num_levels;
	final double level_smoothing;
	final double crp_smoothing;
	final double topic_word_smoothing;
	final int num_iterations;

	Alphabet alphabet;
	HierarchicalLDAWithPrediction hlda;

	long trainTimeNano = Long.MAX_VALUE;

	/**
	 * @param trainingSet      Documents to use to train the model.
	 * @param terms            The Terms instance corresponding to trainingSet.
	 * @param num_levels       The number of levels in the HLDA tree.
	 *                         Corresponds to the --num_levels option to mallet
	 *                         hlda.
	 * @param level_smoothing  The smoothing over level distributions.
	 *					 	   Corresponds to the --alpha option to mallet hlda.
	 * @param crp_smoothing    The Chinese Restaurant Problem smoothing
	 *                         parameter.  Corresponds to the --gamma option to
	 *                         mallet hlda.
	 * @param topic_word_smoothing
	 *                         Corresponds to the --eta option to mallet hlda.
	 * @param num_iterations   The number of iterations to run in training, and
	 *					       also the number of samples to take when doing
	 *					       prediction.
	 */
	public MalletHldaPrediction(
			List<TrainingPaper> trainingSet,
			Terms terms,
			int num_levels,
			double level_smoothing,
			double crp_smoothing,
			double topic_word_smoothing,
			int num_iterations) {
		super(String.format("mHldaPred-%dl-%fa-%fg",
						    num_levels, level_smoothing, crp_smoothing));
		this.num_levels = num_levels;
		this.level_smoothing = level_smoothing;
		this.crp_smoothing = crp_smoothing;
		this.topic_word_smoothing = topic_word_smoothing;
		this.num_iterations = num_iterations;

		alphabet = makeIdentityAlphabet(terms.size());

		train(trainingSet);
	}

	Alphabet makeIdentityAlphabet(int vocabSize) {
		Integer[] alphabet = new Integer[vocabSize];
		for (int i = 0; i < vocabSize; ++i)
			alphabet[i] = i;
		return new Alphabet(alphabet);
	}

	// Based on HierarchicalLDATUI.main().
	public void train(List<TrainingPaper> trainingSet) {
		InstanceList instances = makeInstanceList(trainingSet);
		hlda = new HierarchicalLDAWithPrediction();
		
		// Set hyperparameters

		hlda.setAlpha(level_smoothing);
		hlda.setGamma(crp_smoothing);
		hlda.setEta(topic_word_smoothing);

		// Initialize random number generator

		Randoms random = new Randoms();

		// Initialize and start the sampler

		long startNanoTime = System.nanoTime();
		hlda.initialize(instances, null, num_levels, random);
		hlda.estimate(num_iterations);
		trainTimeNano = System.nanoTime() - startNanoTime;
	}

	InstanceList makeInstanceList(List<?extends PaperIF> papers) {
		InstanceList il = new InstanceList(null);
		for (PaperIF paper : papers) {
			il.add(new Instance(makeFeatureSequence(paper), null, null, null));
		}
		return il;
	}

	FeatureSequence makeFeatureSequence(PaperIF paper) {
		FeatureSequence fs = new FeatureSequence(alphabet);
		for (int word : paper.getTrainingWords()) {
			int count = paper.getTrainingTf(word);
			for (int i = 0; i < count; ++i) {
				fs.add(word);
			}
		}
		return fs;
	}

	@Override
	public double getTrainTime() {
		return trainTimeNano / 1.0e9;
	}

	@Override
    public double[] predict(PredictionPaper testPaper, RunInfo testInfo) {
		long startNanoTime = System.nanoTime();
		double[] ret = hlda.predictNextWord(
				num_iterations, makeFeatureSequence(testPaper));
		testInfo.put("testTime", (System.nanoTime() - startNanoTime) / 1.0e9);
		return ret;
	}
}
