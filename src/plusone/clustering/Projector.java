package plusone.clustering;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.util.*;

import org.ejml.simple.SimpleMatrix;

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
	private double learnedAlpha;
	private boolean synthetic;

	public Projector(String name) {
		super(name);
	}
	
	public Projector(String name, List<TrainingPaper> trainingSet, 
			Indexer<String> wordIndexer,
			Terms terms, 
			int numTopics, 
			Map<PaperAbstract, Integer> trainingIndices,
			Map<PaperAbstract, Integer> testIndices,
			double learnedAlpha,
			boolean synthetic) {
		this(name + "-" + numTopics);
		this.name = name;
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
		this.trainingIndices = trainingIndices;
		this.testIndices = testIndices;
		this.learnedAlpha = learnedAlpha;
		this.synthetic = synthetic;
		train();
	}

	/**
	 * Runs the MATLAB program found in projector/predictTopics.m
	 */
	private void train() {
		if (!new File("projector/data").exists()) {
            new File("projector/data").mkdir();
        }
		//Use the commented block for the custom inference: 
        /*System.out.println("We are getting beta from the projector");
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
		System.out.println("done.");*/
        System.out.println("We are getting beta from the projector");
        System.out.println("cleaning out projector folder for training...");
        Utils.runCommand("rm projector/data/documents", true);
        Utils.runCommand("rm projector/data/final.beta", true);
        Utils.runCommand("rm projector/data/centroid", true);
        Utils.runCommand("rm projector/data/labels", true);
        Utils.runCommand("rm projector/data/projected", true);
        Utils.runCommand("rm projector/data/V", true);
        createProjectorInput("projector/data/documents", trainingSet);
        //uncomment these next three lines to use kmeans.py
        Utils.runCommand("./run-projector-prepare " + numTopics + " "
        		+ trainingSet.size() + " " + terms.size(), true);
        Utils.runCommand("python src/plusone/clustering/kmeans.py " +
        		"projector/data/projected -k " + 
        		numTopics + " -m cosine -w projector/data " +
        		"-i 50 -q", true);
        Utils.runCommand("./run-projector-train " + numTopics +  " " + 
        		trainingSet.size() + " " + terms.size(), true);
        //uncomment to use projector kmeans
//        while(!Utils.runCommand("./run-projector " + numTopics + " " 
//            + trainingSet.size() + " " + terms.size(), true));
        //uncomment to use rawProjector
//        Utils.runCommand("./run-rawProjector " + numTopics + " " + 
//        			     trainingSet.size() + " " + terms.size(), true);
        betaMatrix = readMatrix("projector/data/final.beta", true);
        System.out.print("replacing trained beta with projector beta...");
        Utils.runCommand("cp projector/data/final.beta lda", false);
        createProjectorInfo("lda/final.other");
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
    if (synthetic) {
	    fileWriter.write("alpha " + 
	                        readAlpha("src/datageneration/output/final.other") 
	                         + " \n");
    } else {
	    fileWriter.write("alpha " + learnedAlpha + " \n");
    }
    fileWriter.close();
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
                System.out.println("Couldn't read LDA alpha");
                e.printStackTrace();
        }
        Scanner lines = new Scanner(filecontents);
        String alphaLine = lines.nextLine();
        alphaLine = lines.nextLine();
        alphaLine = lines.nextLine();
        String[] splitLine = alphaLine.split(" ");
        return Double.parseDouble(splitLine[1]);
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
    /**
    * Takes a list of PaperAbstract documents and writes them to file according
    * to the format specified by lda-c-dist
    * 
    * @param filename  name of the file to be created (will be overwritten
    *                  if it already exists)
    * @param papers    list of papers to be written to file 
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

    @Override
	public double[][] predict(List<PredictionPaper> testDocs) {
		/*Projector with custom inference:
        this.testSet = testDocs;
		System.out.print("writing test indices to file in projector/data...");
		Utils.writeIndices("projector/data/testIndices", testDocs, testIndices);
		System.out.println("done.");
		createProjectorInputTest("projector/data/test_documents", testDocs);
		
		while(!Utils.runCommand("./run-projector-inference " + numTopics + " " 
				+ testSet.size() + " " + terms.size(), true));
		
		return readMatrix("projector/data/predictions", false);*/

        //LDA inference:
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
        SimpleMatrix gammas = new SimpleMatrix(gammasMatrix);
        SimpleMatrix beta = new SimpleMatrix(betaMatrix);
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
	
}
