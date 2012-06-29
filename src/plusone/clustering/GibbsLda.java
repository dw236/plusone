package plusone.clustering;

import plusone.utils.Indexer;
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
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

public class GibbsLda extends ClusteringTest {
	
	private List<TrainingPaper> trainingSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	private SimpleMatrix beta;
	private SimpleMatrix gammas;
	private int numTopics;
	
	public GibbsLda(List<TrainingPaper> trainingSet, Indexer<String> wordIndexer,
			Terms terms, int numTopics) {
		super("GibbsLda");
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics = numTopics;
		train();
	}
	
	/**
	 * Runs GibbsLDA on the training set
	 */
	private void train() {
		try {
			new File("GibbsLDA").mkdir();
		} catch (Exception e) {
			e.printStackTrace();
		}

		String trainingData = "GibbsLDA/train.dat";

		createGibbsLdaInput(trainingData, trainingSet);
		Utils.runCommand("lib/GibbsLDA/src/lda -est -alpha 0.001 -ntopics "
				+ numTopics + " -dfile " + trainingData, true);
		double[][] betaMatrix = readLdaResultFile("GibbsLDA/model-final.phi", 0, false);
		beta = new SimpleMatrix(betaMatrix);
	}
	
	@Override
	public double[][] predict(List<PredictionPaper> testDocs){
		HashMap<Integer,String> translation = readWordMap("GibbsLDA/wordmap.txt");
		double[][] result;
		String testData = "test.dat";
		
		createGibbsLdaInputTest("GibbsLDA/" + testData, testDocs);
		Utils.runCommand("lib/GibbsLDA/src/lda -inf -dir "
				+ "GibbsLDA -model model-final -dfile " + testData, false);
		
		double[][] gammasMatrix = readLdaResultFile("GibbsLDA/"+testData+".theta", 0, false);
		double alpha = readAlpha("GibbsLDA/"+testData+".others");
		
		for (int i=0; i<gammasMatrix.length; i++) {
			for (int j=0; j<gammasMatrix[i].length; j++) {
				gammasMatrix[i][j] -= alpha;
			}
		}
		gammas = new SimpleMatrix(gammasMatrix);
		SimpleMatrix probabilities = gammas.mult(beta);

		result = new double[probabilities.numRows()]
		                    [terms.size()];
		for (int row=0; row<probabilities.numRows(); row++) {
			for (int col=0; col<probabilities.numCols(); col++) {
				int realCol = wordIndexer.indexOf(translation.get(col));
				result[row][realCol] = probabilities.get(row, col);
			}
		}
		return result;
	}
	
	/**
	 * Takes a file output by GibbsLDA and stores it in a matrix.
	 * 
	 * @param filename	file to be read
	 * @param start		TODO use for start (typically 0)
	 * @param exp		whether to exponentiate the read entries
	 * @return	a double[][] (matrix) with the contents of filename 
	 */
	private double[][] readLdaResultFile(String filename, int start, 
			boolean exp) {
		List<String[]> gammas = new ArrayList<String[]>();
		double[][] results = null;
		
		try {
			FileInputStream fstream = new FileInputStream(filename);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;

			int c = 0;
			while ((strLine = br.readLine()) != null) {
				if (c >= start) {
					gammas.add(strLine.trim().split(" "));
				}
				c++;
			}

			results = new double[gammas.size()][];
			for (int i = 0; i < gammas.size(); i++) {
				results[i] = new double[gammas.get(i).length];
				for (int j = 0; j < gammas.get(i).length; j++) {
					results[i][j] = new Double(gammas.get(i)[j]);
					if (exp)
						results[i][j] = Math.exp(results[i][j]);
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return results;
	}
	
	private void createGibbsLdaInput(String filename, List<TrainingPaper> papers){
		System.out.print("creating GibbsLDA input in file: " + filename + " ... ");

		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);
		
		fileWriter.write(papers.size()+"\n");

		for (TrainingPaper paper : papers) {
			for (int word : paper.getTrainingWords()) {
				for (int i = 0; i < paper.getTrainingTf(word); i++) {
					fileWriter.write(wordIndexer.get(word) + " ");
				}
			}
			fileWriter.write("\n");
		}

		fileWriter.close();
		
		System.out.println("done.");
	}
	
	private void createGibbsLdaInputTest(String filename, List<PredictionPaper> papers){
		System.out.print("creating GibbsLDA input in file: " + filename + " ... ");

		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);
		
		fileWriter.write(papers.size()+"\n");

		for (PredictionPaper paper : papers) {
			for (int word : paper.getTrainingWords()) {
				for (int i = 0; i < paper.getTrainingTf(word); i++) {
					fileWriter.write(wordIndexer.get(word) + " ");
				}
			}
			fileWriter.write("\n");
		}

		fileWriter.close();
		
		System.out.println("done.");
	}
	
	/**
	 * Reads in the value of alpha from a *.others file, contained in the GibbsLDA output
	 * 
	 * @param filename the path to a *.others file
	 * @return the numerical value of alpha
	 */
	private double readAlpha(String filename) {
		FileInputStream filecontents = null;
		try {
			filecontents = new FileInputStream(filename);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		Scanner lines = new Scanner(filecontents);
		String alphaLine = lines.nextLine();
		String[] splitLine = alphaLine.split("=");
		return Double.parseDouble(splitLine[1]);
	}
	
	/** 
	 * Maps each index of the returned array to the word it represents.
	 * wordIndexer.indexOf(wordMap.get(#)) changes from their index to ours
	 * 
	 * @param filename path to wordmap.txt
	 * @return a map from each index as given by the GibbsLda code to each word as
	 * a string in its original dataset
	 */
	private HashMap<Integer,String> readWordMap(String filename) {
		HashMap<Integer,String> result = new HashMap<Integer,String>();
		FileInputStream filecontents = null;
		try {
			filecontents = new FileInputStream(filename);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		Scanner lines = new Scanner(filecontents);
		lines.nextLine();
		while(lines.hasNextLine()) {
			String[] index = lines.nextLine().split(" ");
			result.put(Integer.parseInt(index[1]), index[0]);
		}
		return result;
	}
}