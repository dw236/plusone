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

public class Ctm extends ClusteringTest {

	private List<TrainingPaper> trainingSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	private int numTopics;
	private SimpleMatrix beta;
	private SimpleMatrix gammas;
	private List<PredictionPaper> testDocs;
	

	public Ctm(List<TrainingPaper> trainingSet, Indexer<String> wordIndexer,
			Terms terms, int numTopics) {
		super("Ctm");
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
		train();
	}

	/**
	 * Runs ctm-dist on the training set to learn the beta matrix and alpha
	 * parameter (in this case, all alphas to the dirichlet are equal)
	 */
	private void train() {
		try {
			new File("ctm").mkdir();
		} catch (Exception e) {
			e.printStackTrace();
		}

		String trainingData = "ctm/train.dat";

		createCtmInput(trainingData, trainingSet);
		Utils.runCommand("lib/ctm-dist/ctm est " + trainingData
				+ " " + numTopics + " rand ctm "
				+ "lib/ctm-dist/settings.txt", true);
		double[][] betaMatrix = readCtmFile("ctm/final-log-beta.dat",
				numTopics, terms.size(), true);
		beta = new SimpleMatrix(betaMatrix);
	}

	/**
	 * Given a set of test documents, runs lda-c-dist inference to learn the
	 * final gammas. Then, subtracts alpha from each gamma to find the expected
	 * number of times each word appears per topic. Finally, multiplies each
	 * gamma by beta to find the expected number of times a word appears for
	 * each document.
	 * 
	 * @param testDocs	the list of documents to run prediction on
	 * @return	the expected number of times each word appears per document
	 */
	@Override
	public double[][] predict(List<PredictionPaper> testDocs){
		String testData = "ctm/test.dat";

		createCtmInputTest(testData, testDocs);
		Utils.runCommand("lib/ctm-dist/ctm inf "
				+ "ctm/test.dat ctm/final ctm/holdout lib/ctm-dist/inf-settings.txt"
				, false);
		
		double[][] gammasMatrix = readCtmFile("ctm/holdout-lambda.dat",
				testDocs.size(), numTopics, false);
		gammas = new SimpleMatrix(gammasMatrix);
		SimpleMatrix probabilities = gammas.mult(beta);
		
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
	 * Takes a file output by lda-c-dist and stores it in a matrix.
	 * 
	 * @param filename	file to be read
	 * @param start		TODO use for start (typically 0)
	 * @param exp		whether to exponentiate the read entries
	 * @return	a double[][] (matrix) with the contents of filename 
	 */
	private double[][] readCtmFile(String filename, int rows, 
			int cols, boolean exp) {
		List<String[]> gammas = new ArrayList<String[]>();
		double[][] results = null;
		
		try {
			FileInputStream fstream = new FileInputStream(filename);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;

			int c = 0;
			String[] thisRow = new String[cols];
			while ((strLine = br.readLine()) != null) {
				if ((c % cols) == 0 && (c != 0)) {
					gammas.add(thisRow);
					thisRow = new String[cols];
				}
				thisRow[c % cols] = strLine;
				c++;
			}
			gammas.add(thisRow);

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

	/**
	 * Identical to createLdaInput
	 * @param filename
	 * @param papers
	 */
	private void createCtmInput(String filename, List<TrainingPaper> papers){
		System.out.print("creating ctm input in file: " + filename + " ... ");

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
	 * Identical to createLdaInputTest
	 * @param filename
	 * @param papers
	 */
	private void createCtmInputTest(String filename, List<PredictionPaper> papers) {

		System.out.print("creating ctm input in file: " + filename + " ... ");

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