package plusone.clustering;

import java.io.File;
import java.util.List;

import plusone.utils.Indexer;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.PredictionPaper;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;
import plusone.utils.Utils;


/**
 * Runs Mallet with the specified algorithm
 */
public class Mallet extends ClusteringTest {

	private String algorithm;
	private List<TrainingPaper> trainingSet;
	private List<PredictionPaper> testSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	private int numTopics;

	public Mallet(String name) {
		super(name);
	}
	
	public Mallet(String algorithm, List<TrainingPaper> trainingSet, 
			Indexer<String> wordIndexer,
			Terms terms, 
			int numTopics) {
		this("mallet" + algorithm + "-" + numTopics);
		this.algorithm = algorithm;
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
		//Note: Mallet topic modeling requires the "--keep-sequence option
		//(no exchangeability assumption)
		Utils.runCommand("lib/mallet-2.0.7/bin/mallet import-file --keep-sequence "
				+ "--input " + trainingData + " --output Mallet/train.mallet", false);
		
		System.out.println("Running Mallet LDA");
		Utils.runCommand("lib/mallet-2.0.7/bin/mallet train-topics "
				+ "--input Mallet/train.mallet --num-topics " + numTopics
				+ " --output-state Mallet/topic-state.gz", false);
	}

	
	@Override
	public double[][] predict(List<PredictionPaper> testDocs) {
		this.testSet = testDocs;
		
		String testingData = "Mallet/test.txt";

		makeMalletInputTest(testingData, testDocs);
		//Anything below this line isn't yet finished
		Utils.runCommand("lib/mallet-2.0.7/bin/mallet import-file --keep-sequence "
				+ "--input " + testingData + " --output Mallet/test.mallet"
				+ "--use-pipe-from Mallet/train.mallet", false);
		
		Utils.runCommand("lib/mallet-2.0.7/bin/mallet infer-topics "
				+ "--input test.mallet --output-doc-topics Mallet/doc-topic.gz"
				+ "", false);

		
		return new double[][]{};
	}
	
	/**
	 * Turns the training documents into Mallet's input format. Uses
	 * a placeholder value (X) for each document's label.
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
					fileWriter.write(wordIndexer.get(word) + " ");
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
					fileWriter.write(wordIndexer.get(word) + " ");
				}
			}
			fileWriter.write("\n");
		}
		fileWriter.close();
		
		System.out.println("done");
	}

}