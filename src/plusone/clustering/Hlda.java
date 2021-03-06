package plusone.clustering;

import plusone.utils.Indexer;
import plusone.utils.TrainingPaper;
import plusone.utils.PredictionPaper;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.Terms;
import plusone.utils.Utils;
import java.io.File;
import java.util.List;

public class Hlda extends ClusteringTest {
	
	private List<TrainingPaper> trainingSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	
	public Hlda(List<TrainingPaper> trainingSet, Indexer<String> wordIndexer,
			Terms terms) {
		super("Hlda");
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		train();
	}
	
	/**
	 * Runs hlda-c on the training set
	 */
	private void train() {
		try {
			new File("hlda").mkdir();
		} catch (Exception e) {
			e.printStackTrace();
		}

		String trainingData = "hlda/train.dat";

		createHldaInput(trainingData, trainingSet);
		Utils.runCommand("lib/hlda-c/main gibbs " + trainingData
				+ " lib/hlda-c/settings-d4.txt hlda", false);

	}
	
	@Override
	//TODO: Figure out hlda output and fill this in correctly
	public double[][] predict(List<PredictionPaper> testDocs){
		
		String trainingData = "hlda/train.dat";
		String testData = "hlda/test.dat";
		createHldaInputTest(testData, testDocs);
		
		Utils.runCommand("lib/hlda-c/main heldout " + trainingData
				+ " " + testData + " lib/hlda-c/settings-d5.txt hlda", false);

		return new double[terms.size()][terms.size()];
	}
	
	private void createHldaInput(String filename, List<TrainingPaper> papers){
		System.out.print("creating hlda input in file: " + filename + " ... ");

		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);

		for (TrainingPaper paper : papers) {
			fileWriter.write(paper.getTrainingWords().size() + " ");
			
			for (int word : paper.getTrainingWords()) {
				fileWriter.write(word + ":" + paper.getTrainingTf(word) + " ");
			}
			fileWriter.write("\n");
		}

		fileWriter.close();
		
		System.out.println("done.");
	}
	
	/**
	 * Takes a list of PaperAbstract documents and writes them to file according
	 * to the format specified by hlda-c (same format as lda-c-dist)
	 * 
	 * @param filename	name of the file to be created (will be overwritten
	 * 					if it already exists)
	 * @param papers	list of papers to be written to file 
	 */
	private void createHldaInputTest(String filename, List<PredictionPaper> papers) {

		System.out.print("creating hlda input in file: " + filename + " ... ");

		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);

		for (PredictionPaper paper : papers) {
			fileWriter.write(paper.getTrainingWords().size() + " ");
			
			for (int word : paper.getTrainingWords()) {
				fileWriter.write(word + ":" + paper.getTrainingTf(word) + " ");
			}
			fileWriter.write("\n");
		}

		fileWriter.close();
		
		System.out.println("done.");
	}
}