package plusone.clustering;

import plusone.utils.Indexer;
import plusone.utils.PaperAbstract;
import plusone.utils.TrainingPaper;
import plusone.utils.PredictionPaper;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.Terms;
import plusone.utils.Utils;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

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
}