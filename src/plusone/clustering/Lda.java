package plusone.clustering;

import plusone.utils.Indexer;
import plusone.utils.PaperAbstract;
import plusone.utils.TrainingPaper;
import plusone.utils.PredictionPaper;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.RunInfo;
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

	private String name;
	private List<TrainingPaper> trainingSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	private int numTopics;
	private SimpleMatrix beta;
	private SimpleMatrix gammas;
	private Map<PaperAbstract, Integer> trainingIndices;
	private Map<PaperAbstract, Integer> testIndices;
	//flag to take true parameters (only for synthesized data)
	private boolean trainCheat;
	private boolean testCheat;
	private boolean project;
	private List<PredictionPaper> testDocs;
	private String[] hoverText;
	private long trainTimeNano = Long.MAX_VALUE;
	
	/** 
	 * Changes flags in Lda based on which algorithm is being run 
	 * @param variant specific algorithm being tested (lda, LdaT, ldaC, proj)
	 */
	public Lda(String variant) {
		super(variant);
		if (variant.substring(0, 4).equals("ldaT")) {
			this.trainCheat = true;
			this.testCheat = false;
		} else if (variant.substring(0,4).equals("ldaC")) {
			this.trainCheat = true;
			this.testCheat = true;
		} else if (variant.substring(0, 4).equals("proj")) {
			this.project = true;
		} else {
			this.trainCheat = false;
			this.testCheat = false;
		}
	}

	public Lda(String variant, List<TrainingPaper> trainingSet, Indexer<String> wordIndexer,
			Terms terms, int numTopics) {
		this(variant + "-" + numTopics);
		this.name = variant;
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
	}

	public Lda(String variant, List<TrainingPaper> trainingSet, 
			Indexer<String> wordIndexer,
			Terms terms, 
			int numTopics, 
			Map<PaperAbstract, Integer> trainingIndices,
			Map<PaperAbstract, Integer> testIndices) {
		this(variant, trainingSet, wordIndexer, terms, numTopics);
		this.trainingIndices = trainingIndices;
		this.testIndices = testIndices;
		train();
	}

	@Override
	public double getTrainTime() {
	    return trainTimeNano / 1.0e9;
	}

    /**
     * Returns the estimated word-topic matrix, or null if e.g. the model has
     * not been trained.
     */
    public SimpleMatrix getWordTopicMatrix() {
        // beta is (# topics) x (# words); we want (# words) x (# topics).
        return beta.transpose();
    }

	/**
	 * Lda:
	 * Runs lda-c-dist on the training set to learn the beta matrix and alpha
	 * parameter (in this case, all alphas to the dirichlet are equal)
	 * LdaT/LdaC:
	 * Copies over the true gamma and beta matrix
	 * projector:
	 * Gets beta from projector
	 */
	private void train() {
		System.out.print("cleaning out lda folder for training...");
		Utils.runCommand("./clear-folder lda", true);
		double[][] betaMatrix = null;
		if (trainCheat) {
			System.out.println("We are cheating and using the true beta");
			betaMatrix = getRealBeta("src/datageneration/output/" + 
					"documents_model-out");
			cheat();
			trainTimeNano = Long.MAX_VALUE;
		} else if (project) { 
            if (!new File("projector/data").exists()) {
                new File("projector/data").mkdir();
            }
			System.out.println("We are getting beta from the projector");
			System.out.println("cleaning out projector folder for training...");
			Utils.runCommand("rm projector/data/documents", true);
			Utils.runCommand("rm projector/data/final.beta", true);
			createProjectorInput("projector/data/documents", trainingSet);
			long startNanoTime = System.nanoTime();
			while(!Utils.runCommand("./run-projector " + numTopics + " " 
					+ trainingSet.size() + " " + terms.size(), true));
			trainTimeNano = System.nanoTime() - startNanoTime;
			betaMatrix = readLdaResultFile("projector/data/final.beta", 0 , true);
			System.out.print("replacing trained beta with projector beta...");
			Utils.runCommand("cp projector/data/final.beta lda", false);
			createProjectorInfo("lda/final.other");
			System.out.println("done.");
		} else {
			try {
				new File("lda").mkdir();
			} catch (Exception e) {
				e.printStackTrace();
			}
	
			String trainingData = "lda/train.dat";
	
			createLdaInput(trainingData, trainingSet);
			long startNanoTime = System.nanoTime();
			Utils.runCommand("lib/lda-c-dist/lda est 1 " + numTopics
					+ " lib/lda-c-dist/settings.txt " + trainingData
					+ " random lda", false);
			trainTimeNano = System.nanoTime() - startNanoTime;
			new File ("lda/trained").mkdir();
			System.out.print("copying trained beta to 'trained' folder...");
			Utils.runCommand("cp lda/final.beta lda/trained", false);
			System.out.print("copying trained gammas to 'trained' folder...");
			Utils.runCommand("cp lda/final.gamma lda/trained", false);
		
			betaMatrix = readLdaResultFile("lda/final.beta", 0, true);
		}
		

		beta = new SimpleMatrix(betaMatrix);
	}

	//TODO move this
	private void createProjectorInput(String filename, List<TrainingPaper> papers) {
		System.out.print("creating projector input: " + filename + " ... ");

		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);

		for (TrainingPaper paper : papers) {
			for (int word : paper.getTrainingWords()) {
				for (int i=0; i<paper.getTrainingTf(word); i++) {
					fileWriter.write(word + " ");
				}
			}
			fileWriter.write("\n");
		}

		fileWriter.close();
		
		System.out.println("done.");
	}
	
	//TODO move this
	private void createProjectorInputTest(String filename,
			List<PredictionPaper> papers) {
		System.out.print("creating projector test input: " 
				+ filename + " ... ");
		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);

		for (PredictionPaper paper : papers) {
			for (int word : paper.getTrainingWords()) {
				for (int i=0; i<paper.getTrainingTf(word); i++) {
					fileWriter.write(word + " ");
				}
			}
			fileWriter.write("\n");
		}

		fileWriter.close();
		
		System.out.println("done.");
	}
	
	//TODO move this
	/**
	 * creates the final.other file required for lda inference
	 */
	private void createProjectorInfo(String filename) {
		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);
		fileWriter.write("num_topics " + numTopics + " \n");
		fileWriter.write("num_terms " + terms.size() + " \n");
		fileWriter.write("alpha " + 
						 readAlpha("src/datageneration/output/final.other") 
						 + " \n");
		fileWriter.close();
	}
	
	/**
	 * LdaC:
	 * Multiplies the true beta and gamma and then returns that doc-word
	 * matrix as its result
	 * Other algorithms:
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
	public double[][] predict(List<PredictionPaper> testDocs, RunInfo testInfo){
		this.testDocs = testDocs;
		System.out.print("writing test indices to file in lda/trained...");
		Utils.writeIndices("lda/trained/testIndices", testDocs, testIndices);
		System.out.println("done.");
		double[][] result = null;
		long startNanoTime = System.nanoTime();
		if (testCheat) {
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
		System.out.println(name + " perplexity is " + getPerplexity());
		testInfo.put("testTime",
			     (System.nanoTime() - startNanoTime) / 1.0e9);
		long finishTime = System.nanoTime();
		
		//write result to hoverText so it can be displayed
		int howMany = 20;
		hoverText = new String[howMany];
		int papers = 0;
		for (PredictionPaper doc : testDocs) {
			if (papers > (howMany - 1)) {
				break;
			}
			int paperIndex = testIndices.get(doc);
			String prediction = paperIndex + " ";
			for (int col=0; col<result[papers].length; col++) {
				prediction += result[papers][col] + " ";
			}
			hoverText[papers] = prediction.trim();
			papers++;
		}
		//end hover
		
		return result;
	}

	/**
	 * Currently unused - hover is being used to display average scores of
	 * multiple experiments
	 */
	public String[] getHover() {
		return hoverText;
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

		System.out.print("creating lda test input in file: " 
				+ filename + " ... ");

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
	
	/**
	 * Replaces trained beta/gamma with real beta/gamma
	 */
	private void cheat() {
		System.out.print("replacing trained beta with real beta...");
		Utils.runCommand("cp src/datageneration/output/final.beta lda", false);
		System.out.println("done.");
		
		System.out.print("replacing trained gammas with real gammas...");
		try {
			FileInputStream fstream = new FileInputStream("src/" +
					"datageneration/output/final.gamma");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			PlusoneFileWriter fileWriter = new PlusoneFileWriter("lda/" +
					"final.gamma");
			
			int index = 0;
			while ((strLine = br.readLine()) != null) {
				if (trainingIndices.containsValue(index)) {
					fileWriter.write(strLine);
					fileWriter.write("\n");
				}
				index++;
			}
			
			fileWriter.close();
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
	 * @return the perplexity for testDocs
	 */
	public double getPerplexity() {
		if (testCheat) {
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
