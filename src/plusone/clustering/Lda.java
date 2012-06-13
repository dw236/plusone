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

public class Lda extends ClusteringTest {

	private List<TrainingPaper> trainingSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	private int numTopics;
	private SimpleMatrix beta;
	private SimpleMatrix gammas;
	private Map<PaperAbstract, Integer> trainingIndices;
	private Map<PaperAbstract, Integer> testIndices;
	//flag to take true parameters (only for synthesized data)
	private static boolean CHEAT;
	private List<PredictionPaper> testDocs;

	public Lda(List<TrainingPaper> trainingSet, Indexer<String> wordIndexer,
			Terms terms, int numTopics) {
		super("Lda");
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
	}

	public Lda(List<TrainingPaper> trainingSet, 
			Indexer<String> wordIndexer,
			Terms terms, 
			int numTopics, 
			Map<PaperAbstract, Integer> trainingIndices,
			Map<PaperAbstract, Integer> testIndices) {
		this(trainingSet, wordIndexer, terms, numTopics);
		this.trainingIndices = trainingIndices;
		this.testIndices = testIndices;
		train();
	}

	/**
	 * Runs lda-c-dist on the training set to learn the beta matrix and alpha
	 * parameter (in this case, all alphas to the dirichlet are equal)
	 */
	private void train() {
		CHEAT = false; 	//CHANGE TO false WHEN TRAINING ON REAL DATA
		double[][] betaMatrix = null;
		if (CHEAT) {
			System.out.println("We are cheating and using the true beta");
			betaMatrix = getRealBeta("src/datageneration/output/" + 
					"documents_model-out");
			cheat();
		} else {
			try {
				new File("lda").mkdir();
			} catch (Exception e) {
				e.printStackTrace();
			}
	
			String trainingData = "lda/train.dat";
	
			createLdaInput(trainingData, trainingSet);
			Utils.runCommand("lib/lda-c-dist/lda est 1 " + numTopics
					+ " lib/lda-c-dist/settings.txt " + trainingData
					+ " random lda", false);
		
			betaMatrix = readLdaResultFile("lda/final.beta", 0, true);
		}
		
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
		this.testDocs = testDocs;
		double[][] result = null;
		CHEAT = false;	//CHANGE TO false WHEN TESTING ON REAL DATA
		if (CHEAT) {
			System.out.println("we are cheating and using true parameters " +
					"for prediction");	
			double[][] gammasMatrix = getRealGamma("src/datageneration/output/"
					+ "documents_model-out");
			gammas = new SimpleMatrix(gammasMatrix);
			SimpleMatrix probabilities = gammas.mult(beta);
			result = new double[testDocs.size()][probabilities.numCols()];
			
			int row = 0;
			for (PredictionPaper doc : testDocs) {
				int paperIndex = testIndices.get(doc);
				int col = 0;
				for (col = 0; col < probabilities.numCols(); col++) {
					result[row][col] = probabilities.get(paperIndex, col);
				}
				row++;
			}
		} else {
			
			String testData = "lda/test.dat";
	
			createLdaInputTest(testData, testDocs);
			Utils.runCommand("lib/lda-c-dist/lda inf " + 
					" lib/lda-c-dist/settings.txt " + "lda/final " + 
					testData + " lda/output", false);
			
			double[][] gammasMatrix = readLdaResultFile("lda/output-gamma.dat",
					0, false);
			double alpha = readAlpha("lda/final.other");
			for (int i=0; i<gammasMatrix.length; i++) {
				for (int j=0; j<gammasMatrix[i].length; j++) {
					gammasMatrix[i][j] -= alpha;
				}
			}
			gammas = new SimpleMatrix(gammasMatrix);
			SimpleMatrix probabilities = gammas.mult(beta);
			
			result = new double[probabilities.numRows()]
			                    [probabilities.numCols()];
			for (int row=0; row<probabilities.numRows(); row++) {
				for (int col=0; col<probabilities.numCols(); col++) {
					result[row][col] = probabilities.get(row, col);
				}
			}
		}
		return result;
	}

	private void createLdaInput(String filename, List<TrainingPaper> papers){
		System.out.print("creating lda input in file: " + filename + " ... ");

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
	 * to the format specified by lda-c-dist
	 * 
	 * @param filename	name of the file to be created (will be overwritten
	 * 					if it already exists)
	 * @param papers	list of papers to be written to file 
	 */
	private void createLdaInputTest(String filename, List<PredictionPaper> papers) {

		System.out.print("creating lda input in file: " + filename + " ... ");

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
	
	private void cheat() {
		System.out.print("replacing trained beta with real beta...");
		Utils.runCommand("cp src/datageneration/output/final.beta " +
				"lda", false);
		System.out.println("done.");
		
		System.out.print("replacing trained gammas with real gammas...");
		try {
			FileInputStream fstream = new FileInputStream("src/" +
					"datageneration/output/final.gamma");
			PlusoneFileWriter fileWriter = new PlusoneFileWriter("lda/" +
					"final.gamma");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			
			int index = 0;
			while ((strLine = br.readLine()) != null) {
				if (trainingIndices.containsKey(index)) {
					fileWriter.write(strLine);
				}
			}
		} catch(Exception e) {
			e.printStackTrace();
		}
		System.out.println("done.");
		
		System.out.print("replacing trained alpha with real alpha...");
		Utils.runCommand("cp src/datageneration/output/final.other " +
				"lda", false);
		System.out.println("done.");
	}
	
	/**
	 * Only used for synthesized data. Reads in the distribution matrix that was
	 * used to generate the data.
	 * @param filename	location of stored matrix
	 * @return 			the beta matrix from which the documents were generated
	 */
	private double[][] getRealBeta(String filename) {
		double[][] res = null;
		List<String[]> topics = new ArrayList<String[]>();
		try {
			FileInputStream fstream = new FileInputStream(filename);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			
			while (!(strLine = br.readLine()).equals("V")) {
				topics.add(strLine.trim().split(" "));
			}
			
			res = new double[topics.size()][];
			for (int i = 0; i < topics.size(); i++) {
				res[i] = new double[topics.get(i).length];
				for (int j = 0; j < topics.get(i).length; j++) {
					res[i][j] = new Double(topics.get(i)[j]);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return res;
	}
	
	/**
	 * Only used for synthesized data. Reads in the distribution matrix that was
	 * used to generate the data.
	 * @param filename	location of stored matrix
	 * @return 			the gamma matrix from which the documents were generated
	 */
	private double[][] getRealGamma(String filename) {
		double[][] res = null;
		List<String[]> topics = new ArrayList<String[]>();
		FileInputStream filecontents = null;
		try {
			filecontents = new FileInputStream(filename);
		} catch (FileNotFoundException e) {
			System.out.println("Check your filepath");
			System.exit(1);
		}
		Scanner gammas = new Scanner(filecontents);
		String gammaRow;
		while(!(gammaRow = gammas.nextLine()).equals("V")) {
		}
		while (gammas.hasNextLine()) {
			gammaRow = gammas.nextLine();
			topics.add(gammaRow.trim().split(" "));
		}
		
		res = new double[topics.size()][];
		for (int i = 0; i < topics.size(); i++) {
			res[i] = new double[topics.get(i).length];
			for (int j = 0; j < topics.get(i).length; j++) {
				res[i][j] = new Double(topics.get(i)[j]);
			}
		}

		return res;
	}
	
	/**
	 * Takes a file output by lda-c-dist and stores it in a matrix.
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
	
	/**
	 * Reads in the value of alpha from a *.other file, contained in the LDA output
	 * 
	 * @param filename the path to a *.other file
	 * @return the numerical value of alpha
	 */
	private double readAlpha(String filename) {
		FileInputStream filecontents = null;
		try {
			filecontents = new FileInputStream(filename);
		} catch (FileNotFoundException e) {
			System.out.println("Check your filepath");
			System.exit(1);
		}
		Scanner lines = new Scanner(filecontents);
		String alphaLine = lines.nextLine();
		alphaLine = lines.nextLine();
		alphaLine = lines.nextLine();
		String[] splitLine = alphaLine.split(" ");
		return Double.parseDouble(splitLine[1]);
	}
	
	/**
	 * Returns the perplexity for the test set. Can only be run after the predict method.
	 * 
	 * @param testDocs the testing documents
	 * @return the perplexity for testDocs
	 */
	public double getPerplexity() {
		if (CHEAT) {
			double[][] betaMatrix = getRealBeta("src/datageneration/output/documents_model-out");
			double[][] gammaMatrix = getRealGamma("src/datageneration/output/documents_model-out");
			SimpleMatrix realBetas = new SimpleMatrix(betaMatrix);
			SimpleMatrix realGammas = new SimpleMatrix(gammaMatrix);
			SimpleMatrix probMatrix = realGammas.mult(realBetas);
			
			double numerator = 0, denominator = 0;
			for (int i=0; i<testDocs.size(); i++) {
				double docProb = 0;
				int row = testIndices.get(testDocs.get(i));
				for (Integer j : ((PaperAbstract)testDocs.get(i)).getTrainingWords()) {
					int tf = ((PaperAbstract)testDocs.get(i)).getTrainingTf(j);
					docProb += tf*Math.log(probMatrix.get(row, j));
				}
				numerator += docProb;
				for (int j=0; j<terms.size(); j++) {
					denominator += ((PaperAbstract)testDocs.get(i)).getTrainingTf(j);
				}
			}
			System.out.println(Math.exp(-1*numerator/denominator));
			return Math.exp(-1*numerator/denominator);
		} else {
			FileInputStream filecontents = null;
			try {
				filecontents = new FileInputStream("lda/output-lda-lhood.dat");
			} catch (FileNotFoundException e) {
				System.out.println("Could not locate output-lda-lhood");
				System.exit(1);
			}
			Scanner logLhoods = new Scanner(filecontents);

			double numerator = 0, denominator = 0;
			for (int i=0; i<testDocs.size(); i++) {
				numerator += Double.parseDouble(logLhoods.nextLine());
				for (int j=0; j<terms.size(); j++) {
					denominator += ((PaperAbstract)testDocs.get(i)).getTestingTf(j);
				}
			}

			return Math.exp(-1*numerator/denominator);
		}
	}
}