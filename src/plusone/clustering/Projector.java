package plusone.clustering;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import plusone.utils.Indexer;
import plusone.utils.PaperAbstract;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.PredictionPaper;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;
import plusone.utils.Utils;


/**
 * @author Victor
 *
 * Runs the projector algorithm (found in projector/predictTopics.m) for
 * training, followed by the projector inference algorithm (found in
 * projector/inference.m). Both use system calls to MATLAB.
 *
 */
public class Projector extends ClusteringTest {

	private String name;
	private List<TrainingPaper> trainingSet;
	private List<PredictionPaper> testSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	private int numTopics;
	private Map<PaperAbstract, Integer> trainingIndices;
	private Map<PaperAbstract, Integer> testIndices;
	private double[][] betaMatrix;

	public Projector(String name) {
		super(name);
	}
	
	public Projector(String name, List<TrainingPaper> trainingSet, 
			Indexer<String> wordIndexer,
			Terms terms, 
			int numTopics, 
			Map<PaperAbstract, Integer> trainingIndices,
			Map<PaperAbstract, Integer> testIndices) {
		this(name + "-" + numTopics);
		this.name = name;
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
		this.trainingIndices = trainingIndices;
		this.testIndices = testIndices;
		train();
	}

	/**
	 * Runs the MATLAB program found in projector/predictTopics.m
	 */
	private void train() {
		if (!new File("projector/data").exists()) {
            new File("projector/data").mkdir();
        }
		System.out.println("We are getting beta from the projector");
		System.out.println("cleaning out projector folder for training...");
		Utils.runCommand("rm projector/data/documents", false);
		Utils.runCommand("rm projector/data/test_documents", false);
		Utils.runCommand("rm projector/data/final.beta", false);
		Utils.runCommand("rm projector/data/predictions", false);
		createProjectorInput("projector/data/documents", trainingSet);
		while(!Utils.runCommand("./run-projector " + numTopics + " " 
				+ trainingSet.size() + " " + terms.size(), true));
		betaMatrix = readMatrix("projector/data/final.beta", true);
		System.out.print("replacing trained beta with projector beta...");
		Utils.runCommand("cp projector/data/final.beta lda", false);
		System.out.println("done.");
	}

	/**
	 * Creates a file containing the documents to be used for training
	 * 
	 * @param filename
	 * 		The name of the file to be created
	 * @param papers
	 * 		The list of documents to be used for training
	 */
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
	
	/**
	 * Creates a file containing the documents to be used for testing
	 * 
	 * @param filename
	 * 		The name of the file to be created
	 * @param papers
	 * 		The list of documents to be used for testing
	 */
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
	
	/**
	 * Reads in a file and interprets it as a matrix (each line is a row,
	 * each entry is a column)
	 * 
	 * @param filename
	 * 		Name of the file to be read
	 * @param exp
	 * 		Flag to exponentiate entries
	 * @return
	 * 		double[][] array containing the read matrix
	 */
	private double[][] readMatrix(String filename, boolean exp) {
		List<String[]> gammas = new ArrayList<String[]>();
		double[][] results = null;
		
		try {
			FileInputStream fstream = new FileInputStream(filename);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;

			while ((strLine = br.readLine()) != null) {
				gammas.add(strLine.trim().split(" "));
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
	
	@Override
	public double[][] predict(List<PredictionPaper> testDocs) {
		this.testSet = testDocs;
		System.out.print("writing test indices to file in projector/data...");
		Utils.writeIndices("projector/data/testIndices", testDocs, testIndices);
		System.out.println("done.");
		createProjectorInputTest("projector/data/test_documents", testDocs);
		
		while(!Utils.runCommand("./run-projector-inference " + numTopics + " " 
				+ testSet.size() + " " + terms.size(), true));
		
		return readMatrix("projector/data/predictions", false);
	}
	
}