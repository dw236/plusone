package plusone.clustering;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

import plusone.utils.Indexer;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.PredictionPaper;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;
import plusone.utils.Utils;

import org.ejml.simple.SimpleMatrix;


/**
 * Runs Mallet with the specified algorithm
 */
public class Mallet extends ClusteringTest {

	private Algorithm algorithm;
	private enum Algorithm {lda, hlda};

	private List<TrainingPaper> trainingSet;
	private List<PredictionPaper> testSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	private int numTopics;
	
	private SimpleMatrix topicWord;
	private SimpleMatrix docTopic;
	
	/* We replace all of the numbers with strings, so we have to
	 * remember what their indices were (outputs of Mallet reference
	 * the new strings)
	 */
	private HashMap<String, Integer> fakeWordIndexer = new HashMap<String, Integer>();
	
	public Mallet(String name) {
		super(name);
	}
	
	public Mallet(String algorithmName, List<TrainingPaper> trainingSet, 
			Indexer<String> wordIndexer,
			Terms terms, 
			int numTopics) {
		this("mallet" + algorithmName + "-" + numTopics);
		if (algorithmName.equals("lda")) {
			this.algorithm = Algorithm.lda;
		} else if (algorithmName.equals("hlda")) {
			this.algorithm = Algorithm.hlda;
		}
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
		train();
	}

	/**
	 * Runs Mallet with the algorithm given in the constructor
	 */
	private void train() {
		try {
			new File("Mallet").mkdir();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		String trainingData = "Mallet/train.txt";

		makeMalletInput(trainingData, trainingSet);
		
		Utils.runCommand("lib/mallet-2.0.7/bin/mallet import-file --keep-sequence "
				+ "--input " + trainingData + " --output Mallet/train.mallet", false);
		
		switch (algorithm) {
			case lda:
				System.out.println("Running Mallet LDA");
				Utils.runCommand("lib/mallet-2.0.7/bin/mallet train-topics"
						+ " --input Mallet/train.mallet --num-topics " + numTopics
						+ " --inferencer-filename Mallet/train.inferencer"
						//+ " --evaluator-filename Mallet/train.evaluator"
						//+ " --output-state Mallet/topic-state.gz"
						+ " --optimize-interval 10 --num-iterations 575"
						+ " --word-topic-counts-file Mallet/word-topics", false);
				break;
			case hlda:
				System.out.println("Running Mallet HLDA");
				Utils.runCommand("lib/mallet-2.0.7/bin/mallet hlda"
						+ " --input Mallet/train.mallet", false);
				break;
		}
		

		
		topicWord = new SimpleMatrix(readTopicWordMatrix("Mallet/word-topics"));
		
	}

	
	@Override
	public double[][] predict(List<PredictionPaper> testDocs) {
		this.testSet = testDocs;
		
		String testingData = "Mallet/test.txt";

		makeMalletInputTest(testingData, testDocs);
		Utils.runCommand("lib/mallet-2.0.7/bin/mallet import-file --keep-sequence"
				+ " --input " + testingData + " --output Mallet/test.mallet"
				+ " --use-pipe-from Mallet/train.mallet", false);
		
		
		Utils.runCommand("lib/mallet-2.0.7/bin/mallet infer-topics --input Mallet/test.mallet"
				+ " --output-doc-topics Mallet/doc-topics"
				+ " --inferencer Mallet/train.inferencer", false);
		
		docTopic = new SimpleMatrix(readDocTopicMatrix("Mallet/doc-topics"));
		
		/*Utils.runCommand("lib/mallet-2.0.7/bin/mallet evaluate-topics --input Mallet/test.mallet"
				+ " --evaluator Mallet/train.evaluator"
				+ " --output-doc-probs Mallet/doc-probs", true);*/

		SimpleMatrix probabilities = docTopic.mult(topicWord);
		double[][] result = new double[probabilities.numRows()]
                [probabilities.numCols()];
		for (int row=0; row<probabilities.numRows(); row++) {
			for (int col=0; col<probabilities.numCols(); col++) {
				result[row][col] = probabilities.get(row, col);
			}
		}

		return result;
	}
	
	/**
	 * Turns the training documents into Mallet's input format. Uses
	 * a placeholder value (X) for each document's label.
	 * 
	 * If any words are made entirely out of numbers, replaces the word with
	 * a fake word made entirely out of characters. Deletes the numbers for
	 * words which are composed of numbers and characters

	 * 
	 * @param filename name of the file
	 * @param trainingSet the list of training docs
	 */
	public void makeMalletInput(String filename, List<TrainingPaper> trainingSet) {
		System.out.print("creating Mallet input in file: " + filename + " ... ");

		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);

		for (TrainingPaper p : trainingSet) {
			fileWriter.write(p.getIndex() + " X ");
			for (int word : p.getTrainingWords()) {
				for (int i = 0; i < p.getTrainingTf(word); i++) {
					String wordAsString = wordIndexer.get(word);
					try {
						StringBuffer wordWithChars = new StringBuffer();
						for (int j = 0; j < wordAsString.length(); j++) {
							int jthCharAsInt = Integer.parseInt("" + wordAsString.charAt(j));
							wordWithChars.append((char)(jthCharAsInt + (int)'a'));
						}
						fakeWordIndexer.put(wordWithChars.toString(), word);
						fileWriter.write(wordWithChars.toString() + " ");
					} catch (NumberFormatException e) {
						//Don't have numbers after all, just write them
						fileWriter.write(wordAsString + " ");
					}
				}
			}
			fileWriter.write("\n");
		}
		fileWriter.close();
		
		System.out.println("done");
	}
	
	/**
	 * Turns the testing documents into Mallet's input format. Uses
	 * a placeholder value (X) for each document's label.
	 * 
	 * If any words are made entirely out of numbers, replaces the word with
	 * a fake word made entirely out of characters. Deletes the numbers for
	 * words which are composed of numbers and characters
	 * 
	 * @param filename name of the file
	 * @param testDocs the list of testing docs
	 */
	public void makeMalletInputTest(String filename, List<PredictionPaper> testDocs) {
		System.out.print("creating Mallet input in file: " + filename + " ... ");

		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);

		for (PredictionPaper p : testDocs) {
			fileWriter.write(p.getIndex() + " X ");
			for (int word : p.getTrainingWords()) {
				for (int i = 0; i < p.getTrainingTf(word); i++) {
					String wordAsString = wordIndexer.get(word);
					try {
						StringBuffer wordWithChars = new StringBuffer();
						for (int j = 0; j < wordAsString.length(); j++) {
							int jthCharAsInt = Integer.parseInt("" + wordAsString.charAt(j));
							wordWithChars.append((char)(jthCharAsInt + (int)'a'));
						}
						fakeWordIndexer.put(wordWithChars.toString(), word);
						fileWriter.write(wordWithChars.toString() + " ");
					} catch (NumberFormatException e) {
						//Don't have numbers after all, just write them
						fileWriter.write(wordAsString + " ");
					}
				}
			}
			fileWriter.write("\n");
		}
		fileWriter.close();
		
		System.out.println("done");
	}
	
	/**
	 * Reads in the doc-topic matrix as given by the --output-doc-topics in
	 * Mallet's inference and gives back a 2D Array with the same information
	 * 
	 * @param filename the output of --output-doc-topics
	 * @return 2D Array with Doc-Topic probabilities
	 */
	public double[][] readDocTopicMatrix(String filename) {
		double[][] ret = new double[testSet.size()][numTopics];
		Scanner lines;
		try {
			lines = new Scanner(new File(filename));
			while (lines.hasNextLine()) {
				String line = lines.nextLine();
				if (line.charAt(0) == '#') {
					continue;
				}
				String[] parsedLine = line.split(" ");
				int docNumber = Integer.parseInt(parsedLine[0]);
				for (int i = 2; i < parsedLine.length; i += 2) {
					int topicNumber = Integer.parseInt(parsedLine[i]);
					double topicProportion = Double.parseDouble(parsedLine[i+1]);
					ret[docNumber][topicNumber] = topicProportion;
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		return ret;
	}

	/**
	 * Reads in the topic-word matrix as given by the --word-topic-counts-file in
	 * Mallet's training and gives back a 2D Array with the same information
	 * 
	 * @param filename the output of --word-topic-counts-file
	 * @return 2D Array with Topic-Word probabilities
	 */
	public double[][] readTopicWordMatrix(String filename) {
		double[][] ret = new double[numTopics][wordIndexer.size()];
		Scanner lines;
		try {
			lines = new Scanner(new File(filename));
			while (lines.hasNextLine()) {
				String line = lines.nextLine();
				if (line.charAt(0) == '#') {
					continue;
				}
				String[] parsedLine = line.split(" ");
				int wordIndex = fakeWordIndexer.get(parsedLine[1]);
				for (int i = 2; i < parsedLine.length; i++) {
					String topicAndCount = parsedLine[i];
					int topic = Integer.parseInt(topicAndCount.split(":")[0]);
					int count = Integer.parseInt(topicAndCount.split(":")[1]);
					ret[topic][wordIndex] = count;
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		return ret;
	}
}