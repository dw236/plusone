package plusone.clustering;

import java.io.File;
import java.util.List;
import java.util.Map;

import org.ejml.simple.SimpleMatrix;

import plusone.utils.Indexer;
import plusone.utils.PaperAbstract;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.PredictionPaper;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;
import plusone.utils.Utils;

public class Kmeans extends ClusteringTest {

	private String name;
	private List<TrainingPaper> trainingSet;
	private List<PredictionPaper> testSet;
	private Indexer<String> wordIndexer;
	private Terms terms;
	private int numTopics;
	private Map<PaperAbstract, Integer> trainingIndices;
	private Map<PaperAbstract, Integer> testIndices;
	private double[][] betaMatrix;
	private boolean synthetic;

	public Kmeans(String name) {
		super(name);
	}
	
	public Kmeans(String name, List<TrainingPaper> trainingSet, 
			Indexer<String> wordIndexer,
			Terms terms, 
			int numTopics, 
			Map<PaperAbstract, Integer> trainingIndices,
			Map<PaperAbstract, Integer> testIndices,
			boolean synthetic) {
		this(name + "-" + numTopics);
		this.name = name;
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
		this.trainingIndices = trainingIndices;
		this.testIndices = testIndices;
		this.synthetic = synthetic;
		train();
	}
	
	private void train() {
		if (!new File("kmeans").exists()) {
            new File("kmeans").mkdir();
        }
		createKmeansInput("kmeans/documents", trainingSet);
		Utils.runCommand("python src/plusone/clustering/kmeans.py " +
        		"kmeans/documents -k " + numTopics + " -m cosine " + 
				"-w kmeans -i 50 -q", true);
		betaMatrix = Utils.readMatrix("kmeans/centers", false);
		System.out.print("moving kmeans beta to lda folder...");
        Utils.runCommand("cp kmeans/centers lda", false);
        Utils.runCommand("mv lda/centers lda/final.beta", true);
        Utils.createLdaInfo("kmeans/final.other", numTopics, terms.size(), 
        		synthetic);
        Utils.runCommand("cp kmeans/final.other lda", true);
        System.out.println("done.");
	}
	
	@Override
	public double[][] predict(List<PredictionPaper> testDocs) {
		String testData = "lda/test.dat";

        createLdaInputTest(testData, testDocs);
        Utils.runCommand("lib/lda-c-dist/lda inf " + 
                        " lib/lda-c-dist/settings.txt " + "lda/final " + 
                        testData + " lda/output", false);

        double[][] gammasMatrix = Utils.readMatrix("lda/output-gamma.dat", 
        		false);
        double alpha = Utils.readAlpha("lda/final.other");
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
	
    /**TODO move this to Utils.java because other classes have a similar method
	 * Creates a file containing the documents to be used for training
	 * Documents are represented as an n x m matrix, where each row is 
	 * normalized (ie each entry i, j is the percentage of document i that 
	 * is word j)
	 * 
	 * @param filename
	 * 		The name of the file to be created
	 * @param papers
	 * 		The list of documents to be used for training
	 */
	private void createKmeansInput(String filename, List<TrainingPaper> papers) {
		System.out.print("creating kmeans input: " + filename + " ... ");
		double[][] fullRepresentation = new double[papers.size()][terms.size()];
		int paperIndex = 0;
		for (TrainingPaper paper : papers) {
			double sum = 0;
			for (int word : paper.getTrainingWords()) {
				double wordFrequency = paper.getTrainingTf(word);
				fullRepresentation[paperIndex][word] = wordFrequency; 
				sum += wordFrequency;
			}
			for (int col=0; col<fullRepresentation[paperIndex].length; col++) {
				fullRepresentation[paperIndex][col] /= sum;
			}
			paperIndex++;
		}
		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);
		for (int row=0; row<fullRepresentation.length; row++) {
			for (int col=0; col<fullRepresentation[row].length; col++) {
				fileWriter.write(fullRepresentation[row][col] + " ");
			}
			fileWriter.write("\n");
		}
		fileWriter.close();
		System.out.println("done.");
	}
	
	/** TODO This method exists in several classes--move to Utils?
    * Takes a list of PaperAbstract documents and writes them to file according
    * to the format specified by lda-c-dist
    * 
    * @param filename  name of the file to be created (will be overwritten
    *                  if it already exists)
    * @param papers    list of papers to be written to file 
    */
    private void createLdaInputTest(String filename, List<PredictionPaper> papers) {

        System.out.print("creating lda test input in file: " + filename + 
        		" ... ");
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