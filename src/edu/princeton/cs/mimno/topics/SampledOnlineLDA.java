package edu.princeton.cs.mimno.topics;

import edu.princeton.cs.mimno.UnicodeBarplot;

import cc.mallet.types.*;
import cc.mallet.util.*;

import java.util.logging.Logger;
import java.util.*;
import java.util.zip.*;
import java.io.*;

public class SampledOnlineLDA {

	protected static Logger logger = MalletLogger.getLogger(SampledOnlineLDA.class.getName());

	static cc.mallet.util.CommandOption.String instancesFile = new cc.mallet.util.CommandOption.String
		(SampledOnlineLDA.class, "instances", "FILENAME", true, null,
		 "File containing feature sequences", null);

	static cc.mallet.util.CommandOption.Double momentumOption = new cc.mallet.util.CommandOption.Double
		(SampledOnlineLDA.class, "momentum", "FILENAME", true, 0.0,
		 "Weighting of the previous gradient.", null);

	static cc.mallet.util.CommandOption.String outputPrefix = new cc.mallet.util.CommandOption.String
		(SampledOnlineLDA.class, "output-prefix", "STRING", true, "o-lda",
		 "The prefix for output files (sampling states, parameters, etc)", null);

	static cc.mallet.util.CommandOption.Integer numTopicsOption = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "num-topics", "INTEGER", true, 10,
		 "The number of topics", null);

	static cc.mallet.util.CommandOption.Integer numSamplesOption = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "num-samples", "INTEGER", true, 5,
		 "The number of topics", null);

	static cc.mallet.util.CommandOption.Integer sampleBurnInOption = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "sample-burn-in", "INTEGER", true, 2,
		 "The number of topics", null);

	static cc.mallet.util.CommandOption.Integer batchSizeOption = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "batch-size", "INTEGER", true, 100,
		 "The number of instances to examine before updating parameters", null);

	static cc.mallet.util.CommandOption.Double learningRateOption = new cc.mallet.util.CommandOption.Double
		(SampledOnlineLDA.class, "learning-rate", "INTEGER", true, 100.0,
		 "The learning rate will be 1.0 / ([this value] + t)", null);

	static cc.mallet.util.CommandOption.Double learningRateExponentOption = new cc.mallet.util.CommandOption.Double
		(SampledOnlineLDA.class, "learning-rate-exponent", "NUMBER", true, 0.6,
		 "The learning rate will be pow(1.0 / (offset + t), [this value]). Must be between 0.5 and 1.0", null);

	static cc.mallet.util.CommandOption.Double docTopicSmoothingOption = new cc.mallet.util.CommandOption.Double
		(SampledOnlineLDA.class, "alpha", "POS NUMBER", true, 0.1,
		 "Dirichlet parameter for symmetric prior over document-topic distributions. This is the value for each dimension.", null);

	static cc.mallet.util.CommandOption.Double topicWordSmoothingOption = new cc.mallet.util.CommandOption.Double
		(SampledOnlineLDA.class, "beta", "POS NUMBER", true, 0.1,
		 "Dirichlet parameter for symmetric prior over topic-word distributions. This is the value for each dimension.", null);

	static cc.mallet.util.CommandOption.Integer savedStatesCountOption = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "saved-wordtopics", "INTEGER", true, 100,
		 "The total number of instances to examine", null);

	static cc.mallet.util.CommandOption.Integer docsBetweenSavedStates = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "wordtopics-interval", "INTEGER", true, 500000,
		 "The number of instances to examine between saved states", null);

	static cc.mallet.util.CommandOption.Integer randomSeed = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "random-seed", "INTEGER", true, 1,
		 "An initial seed for the random number generator", null);

	static cc.mallet.util.CommandOption.Integer foldCountOption = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "total-folds", "INTEGER", true, 0,
		 "The number of equal-sized held-out cross validation folds. A value 0 will use all data.", null);

	static cc.mallet.util.CommandOption.Integer heldOutFoldOption = new cc.mallet.util.CommandOption.Integer
		(SampledOnlineLDA.class, "held-out-fold", "INTEGER", true, 0,
		 "The index of the cross validation fold to hold out, starting with 0.", null);

	public Alphabet vocabulary;

	InstanceList training;

	double[][] typeWeights;
	int[][] typeTopics;

	//double[][] typeTopicTokens;
	double[] topicTokenTotals;
	
	double[][] typeTopicGradients;

	int[] wordGradientQueueTopics;
	int[] wordGradientQueueWords;

	int wordGradientLimit = 0;

	int numDocs;
	int numTypes;
	int numTokens;
	int numTopics;

	int numSamples = 5;
	int sampleBurnIn = 2;
	double sampleWeight = 1.0 / (numSamples - sampleBurnIn);

	double documentTopicSmoothing;
	double topicWordSmoothing;

	int heldOutFold = 0;
	int numHeldOutFolds = 0;

	Randoms random;

	public SampledOnlineLDA(InstanceList training, int numTopics, double documentTopicSmoothing, double topicWordSmoothing) {

		this.training = training;
		this.numTopics = numTopics;

		vocabulary = training.getDataAlphabet();
		this.numTypes = vocabulary.size();

		this.numDocs = training.size();

		this.documentTopicSmoothing = documentTopicSmoothing;
		this.topicWordSmoothing = topicWordSmoothing;


        typeWeights = new double[numTypes][numTopics];
        typeTopics = new int[numTypes][numTopics];

		//typeTopicTokens = new double[numTypes][numTopics];
		topicTokenTotals = new double[numTopics];
		
		//typeTopicGradients = new double[numTypes][numTopics];
		wordGradientQueueTopics = new int[5000000];
		wordGradientQueueWords = new int[5000000];

		random = new Randoms(randomSeed.value);

	}
	
	public void setHeldOutFold(int fold, int totalFolds) {
		this.heldOutFold = fold;
		this.numHeldOutFolds = totalFolds;
	}

	public void setSampling(int n, int bi) {
		numSamples = n;
		sampleBurnIn = bi;
		sampleWeight = 1.0 / (numSamples - sampleBurnIn);
	}

	public double approximateExpDigamma(double x) {
		double correction = 0.0;
		while (x < 5.0) {
			correction += 1.0 / x;
			x++;
		}

		return (x - 0.5) * Math.exp(-correction);
	}

	public void train(int numOutputs, int docsBetweenOutput) throws IOException {
		
		long trainingStartTime = System.currentTimeMillis();
		long tokensProcessed = 0;
		long docsProcessed = 0;

		// To regularize output relative to batch sizes, output 
		//  topics after N documents have been processed.
		// This variable is numDocsProcessed / N (integer division)
		long currentOutputMultiple = 0;

		int[] wordOccurrences = new int[numTypes];

		double[] samplingWeights = new double[numTopics];

		double digammaTopicWordSmoothing = Dirichlet.digamma(topicWordSmoothing);
		double expDigammaTopicWordSmoothing = Math.exp(digammaTopicWordSmoothing);
		double[] topicCoefficients = new double[numTopics];
		double coefficientSum = 0.0;

		double[] topicNormalizers = new double[numTopics];

		// The main loop

		double scale = 1.0;

		int[] docTopicCounts = new int[numTopics];

		int iteration = 0;
		while (docsProcessed < numOutputs * docsBetweenOutput) {
			/*
			long initNanos = 0;
			long digammaNanos = 0;
			long expNanos = 0;
			long samplingNanos = 0;
			long setupNanos = 0;
			long sparseNanos = 0;
			long denseNanos = 0;
			long coefficientNanos = 0;
			long startNanos;
			*/

			long startTime = System.currentTimeMillis();
		
			wordGradientLimit = 0;

			// Process a minibatch

			int batchTokens = 0;
			int numSparseTopics = 0;

			for (int topic = 0; topic < numTopics; topic++) {
				topicNormalizers[topic] = 1.0 / (numTypes * topicWordSmoothing + scale * topicTokenTotals[topic] - 0.5);
			}

			int totalSamples = 0;
			int totalChanges = 0;

			for (int batchDoc = 0; batchDoc < batchSizeOption.value; batchDoc++) {
				
				int docIndex = random.nextInt(numDocs);

				// rejection sample for held-out validation
				if (numHeldOutFolds > 0) {
					while ((17 * docIndex) % numHeldOutFolds == heldOutFold) {
						//						logger.info("rejecting " + docIndex);
						docIndex = random.nextInt(numDocs);
					}
				}

				Instance instance = training.get(docIndex);
				FeatureSequence tokens = (FeatureSequence) instance.getData();

				int[] topics = new int[tokens.size()];
				//boolean[] hasLargeCounts = new boolean[tokens.size()];
				Arrays.fill(docTopicCounts, 0);

				if ((numSamples - sampleBurnIn) * tokens.size() + wordGradientLimit >= wordGradientQueueTopics.length) {
					int newSize = 2 * wordGradientQueueTopics.length;
					int[] tempWordGradientQueueTopics = new int[newSize];
					int[] tempWordGradientQueueWords = new int[newSize];
					System.arraycopy(wordGradientQueueTopics, 0, tempWordGradientQueueTopics, 0, wordGradientQueueTopics.length);
					System.arraycopy(wordGradientQueueWords, 0, tempWordGradientQueueWords, 0, wordGradientQueueWords.length);
					
					wordGradientQueueTopics = tempWordGradientQueueTopics;
					wordGradientQueueWords = tempWordGradientQueueWords;
				}

				tokensProcessed += tokens.size();

				/*
				// initialize from the word distributions
				for (int position = 0; position < tokens.size(); position++) {
					int type = tokens.getIndexAtPosition(position);
					//double[] currentTypeTopicTokens = typeTopicTokens[type];
					double[] currentTypeWeights = typeWeights[type];
					int[] currentTypeTopics = typeTopics[type];

					wordOccurrences[type] ++;

					double sum = 0.0;
					int index = 0;
					while (index < currentTypeWeights.length && currentTypeWeights[index] > 0.0) {
						//if (topicWordSmoothing < 0.1 && scale * currentTypeWeights[index] > 1000.0) {
						//	hasLargeCounts[position] = true;
						//}
						sum += currentTypeWeights[index];
						index++;
					}
					
					int newTopic = 0;

					if (sum == 0.0) {
						newTopic = random.nextInt(numTopics);
					}
					else {
						double sample = random.nextUniform() * sum;
						index = 0;
						while (sample > currentTypeWeights[index] && index < currentTypeWeights.length && currentTypeWeights[index] > 0.0) {
							sample -= currentTypeWeights[index];
							index++;
						}
						newTopic = currentTypeTopics[index];
					}
					
					topics[position] = newTopic;
					docTopicCounts[newTopic]++;
				}
				*/
				coefficientSum = 0.0;
				for (int topic = 0; topic < numTopics; topic++) {
					topicCoefficients[topic] = (documentTopicSmoothing + docTopicCounts[topic]) * topicNormalizers[topic]; // / (numTypes * topicWordSmoothing + scale * topicTokenTotals[topic] - 0.5);
					coefficientSum += topicCoefficients[topic];
				}
				
				for (int sweep = 0; sweep < numSamples; sweep++) {
					//System.out.println(sweep + "\t" + UnicodeBarplot.getBars(docTopicCounts));

					for (int position = 0; position < tokens.size(); position++) {

						int type = tokens.getIndexAtPosition(position);
						//double[] currentTypeTopicTokens = typeTopicTokens[type];
						double[] currentTypeWeights = typeWeights[type];
						int[] currentTypeTopics = typeTopics[type];
											
						int oldTopic = topics[position];

						if (sweep > 0) {
							docTopicCounts[oldTopic]--;
							coefficientSum -= topicCoefficients[oldTopic];
							topicCoefficients[oldTopic] = (documentTopicSmoothing + docTopicCounts[oldTopic]) * topicNormalizers[oldTopic];
							coefficientSum += topicCoefficients[oldTopic];
						}


						int newTopic = 0;
						
						// At this level, exp(digamma(beta)) is large enough that we don't need
						// to sample in log space.
						int samplingLimit = 0;
						double sparseSamplingSum = 0;
						while (samplingLimit < currentTypeWeights.length && currentTypeWeights[samplingLimit] > 0.0) {
							if (scale * currentTypeWeights[samplingLimit] > 5.0) {
								samplingWeights[samplingLimit] =
									(topicWordSmoothing + scale * currentTypeWeights[samplingLimit] - 0.5 - expDigammaTopicWordSmoothing) *
									topicCoefficients[ currentTypeTopics[samplingLimit] ];
							}
							else {
								samplingWeights[samplingLimit] =
									(approximateExpDigamma(topicWordSmoothing + scale * currentTypeWeights[samplingLimit]) - expDigammaTopicWordSmoothing) *
									topicCoefficients[ currentTypeTopics[samplingLimit] ];
							}
							sparseSamplingSum += samplingWeights[samplingLimit];
							samplingLimit++;
						}
						
						double sample = (sparseSamplingSum + expDigammaTopicWordSmoothing * coefficientSum) * random.nextUniform();
						
						if (sample < sparseSamplingSum) {
							int index = 0;
							while (sample > samplingWeights[index] && index < samplingLimit) {
								sample -= samplingWeights[index];
								index++;
							}
							newTopic = currentTypeTopics[index];
						}
						else {
							sample = (sample - sparseSamplingSum) / expDigammaTopicWordSmoothing;
							newTopic = 0;
							while (sample > topicCoefficients[ newTopic ]) {
								sample -= topicCoefficients[ newTopic ];
								newTopic++;
							}
						}
						
						
						if (sweep >= sampleBurnIn) {
							totalSamples++;
							if (newTopic != oldTopic) { totalChanges++; }
						}							

						topics[position] = newTopic;
						docTopicCounts[newTopic]++;
						coefficientSum -= topicCoefficients[newTopic];
						topicCoefficients[newTopic] = (documentTopicSmoothing + docTopicCounts[newTopic]) * topicNormalizers[newTopic];
						coefficientSum += topicCoefficients[newTopic];
						wordGradientQueueWords[wordGradientLimit] = type;
						wordGradientQueueTopics[wordGradientLimit] = newTopic;
						wordGradientLimit++;
						//						currentTypeTopicGradient[newTopic] += sampleWeight;
						
						

					}
				}
			}

			docsProcessed += batchSizeOption.value;

			//System.out.format("numSparse %d init %d digamma %d exp %d sampling %d setup %d sparse %d dense %d coeff %d total %d\n", numSparseTopics, initNanos/1000, digammaNanos/1000, expNanos/1000, samplingNanos/1000, setupNanos/1000, sparseNanos/1000, denseNanos/1000, coefficientNanos/1000, (System.nanoTime() - overallStartNanos)/1000);

			// Step in the direction of the gradient

			double learningRate = Math.pow(learningRateOption.value + iteration, -learningRateExponentOption.value);
			double oneMinusLearningRate = 1.0 - learningRate;

			scale *= oneMinusLearningRate;


			//System.out.println((System.currentTimeMillis() - startTime) + "\tsampling");
			// Now add the gradient

			double wordWeight = learningRate * numDocs / (scale * numSamples * batchSizeOption.value);

			for (int i = 0; i < wordGradientLimit; i++) {
				int type = wordGradientQueueWords[i];
				int topic = wordGradientQueueTopics[i];
				double[] weights = typeWeights[type];
				int[] topics = typeTopics[type];
				
				// Search through the sparse list to find position of the topic
				int index = 0;
				boolean found = false;
				while (! found && topics[index] != topic && weights[index] > 0.0) {
					index++;
				}
				// At this point either we've found the topic and stopped or found an empty position
				if (! found) { topics[index] = topic; }
				weights[index] += wordWeight;

				//typeTopicTokens[ wordGradientQueueWords[i] ][ wordGradientQueueTopics[i] ] += wordWeight;
				topicTokenTotals[ wordGradientQueueTopics[i] ] += wordWeight;
			}
			wordGradientLimit = 0;

			if (scale < 0.01) {
				logger.info("rescaling " + iteration);
				rescale(scale);
				scale = 1.0;
				sortAndPrune(0.1);
			}

			if ((iteration + 1) % (10000 / batchSizeOption.value) == 0) {
				System.out.format("iteration %d: %dms/iter %dms %f %d tokens\n", iteration + 1, System.currentTimeMillis() - startTime, System.currentTimeMillis() - trainingStartTime, scale, tokensProcessed);
				System.out.format("%d / %d = %f\n", totalChanges, totalSamples, (double) totalChanges / totalSamples);
				System.out.println(UnicodeBarplot.getBars(topicTokenTotals));
				totalSamples = 0;
			}

			if ((iteration + 1) % (50000 / batchSizeOption.value) == 0) {
				logger.info(topWords(30, scale));
			}

			if (docsProcessed / docsBetweenOutput > currentOutputMultiple) {
				currentOutputMultiple = docsProcessed / docsBetweenOutput; // note the int division
				
				try {
					writeWordTopicParameters(outputPrefix.value + ".wordtopics." + currentOutputMultiple, scale);
				} catch (Exception e) {
					logger.info("can't write to file: " + e.getMessage());
				}
				//logger.info(topWords(30));
			}

			iteration++;
		}
	}
	
	/**
	 *  Return an array of sorted sets (one set per topic). Each set 
	 *   contains IDSorter objects with integer keys into the alphabet.
	 *   To get direct access to the Strings, use getTopWords().
	 */
	public ArrayList<TreeSet<IDSorter>> getSortedWords () {
        
		ArrayList<TreeSet<IDSorter>> topicSortedWords = new ArrayList<TreeSet<IDSorter>>(numTopics);

		// Initialize the tree sets
		for (int topic = 0; topic < numTopics; topic++) {
			topicSortedWords.add(new TreeSet<IDSorter>());
		}

		// Collect counts
		for (int type = 0; type < numTypes; type++) {

			int[] topics = typeTopics[type];
			double[] weights = typeWeights[type];

			int index = 0;
			while (index < topics.length && weights[index] > 0.0) {
				topicSortedWords.get(topics[index]).add(new IDSorter(type, weights[index]));
				index++;
			}
		}

		return topicSortedWords;
	}

	public String topWords (int numWords, double scale) {

		StringBuilder output = new StringBuilder();
		ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords();

		for (int topic = 0; topic < numTopics; topic++) {

			TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);
			Iterator<IDSorter> iterator = sortedWords.iterator();
                        
			output.append(topic + "\t" + (scale * topicTokenTotals[topic]) + "\t");

			int i = 0;
			while (i < numWords && iterator.hasNext()) {
				IDSorter sorter = iterator.next();
				//output.append(vocabulary.lookupObject(sorter.getID()) + " (" + wordOccurrences[sorter.getID()] + ") ");
				output.append(vocabulary.lookupObject(sorter.getID()) + " ");
				i++;
			}
			output.append("\n");
		}

		return output.toString();
	}

	public void rescale(double scale) {
		for (int type = 0; type < numTypes; type++) {
			/*
			for (int topic = 0; topic < numTopics; topic++) {
				typeTopicTokens[type][topic] *= scale;
			}
			*/

			double[] weights = typeWeights[type];
			int index = 0;
			while (index < weights.length && weights[index] > 0.0) {
				weights[index] *= scale;
				index++;
			}
		}

		for (int topic = 0; topic < numTopics; topic++) {
			topicTokenTotals[topic] *= scale;
		}
	}

	public void sortAndPrune(double cutoff) {
		for (int type = 0; type < numTypes; type++) {
			double[] weights = typeWeights[type];
			int[] topics = typeTopics[type];

			// Do a simple bubble sort, clearing low values as we go
			int sortedLimit = 0;
			while (sortedLimit < weights.length && weights[sortedLimit] > 0.0) {

				if (weights[sortedLimit] < cutoff) { 
					// Zero out low weights
					weights[sortedLimit] = 0.0;
					topics[sortedLimit] = 0;
				}
				else {
					// Make sure the current value is less than any previous values
					int i = sortedLimit - 1;
					while (i >= 0 && weights[i+1] > weights[i]) {
						int tempTopic = topics[i];
						double tempWeight = weights[i];
						topics[i] = topics[i+1];
						weights[i] = weights[i+1];
						topics[i+1] = tempTopic;
						weights[i+1] = tempWeight;
						i--;
					}
				}
				sortedLimit++;
			}
		}
		
	}

	public void writeWordTopicParameters(String filename, double scale) throws Exception {
		PrintStream out = new PrintStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(filename))));

		for (int type = 0; type < numTypes; type++) {
			Formatter formatter = new Formatter(new StringBuilder(), Locale.US);

			double[] weights = typeWeights[type];
			int index = 0;
			while (index < weights.length && weights[index] > 0.0) {
				formatter.format("%d:%.2f\t", typeTopics[type][index], (scale * weights[index]));
				index++;
			}
			out.println(formatter);
		}
		
		out.close();
	}

 	public static void main (String[] args) throws Exception {
		CommandOption.setSummary (SampledOnlineLDA.class,
								  "Gibbs sampling within variational inference.");
		CommandOption.process (SampledOnlineLDA.class, args);

		int numTopics = numTopicsOption.value;
		
		InstanceList instances = null;
		try {
			instances = InstanceList.load (new File(instancesFile.value));
		} catch (Exception e) {
			System.err.println("Unable to restore instance list " +
							   instancesFile.value + ": " + e);
			System.exit(1);
		}

		logger.info("docs loaded " + instances.size());

		SampledOnlineLDA trainer = new SampledOnlineLDA(instances, numTopicsOption.value, docTopicSmoothingOption.value, topicWordSmoothingOption.value);
		
		trainer.setHeldOutFold(heldOutFoldOption.value, foldCountOption.value);
		trainer.setSampling(numSamplesOption.value, sampleBurnInOption.value);
	
		trainer.train(savedStatesCountOption.value, docsBetweenSavedStates.value);
	}

}