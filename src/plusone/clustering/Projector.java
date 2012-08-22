package plusone.clustering;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import plusone.utils.Indexer;
import plusone.utils.PaperAbstract;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;
import plusone.utils.Utils;
import plusone.clustering.Lda.*;

public class Projector extends ClusteringTest {

	private String name;
	private List<TrainingPaper> trainingSet;
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
		this(name);
		this.name = name;
		this.trainingSet = trainingSet;		
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
		this.trainingIndices = trainingIndices;
		this.testIndices = testIndices;
		train();
	}

	private void train() {
		createProjectorInput("projector/documents", trainingSet);
		System.out.println("We are getting beta from the projector");
		Utils.runCommand("./run-projector", false);
		betaMatrix = readBeta("projector/final.beta", true);
		System.out.print("replacing trained beta with projector beta...");
		Utils.runCommand("cp projector/final.beta lda", false);
		Utils.runCommand("cp src/datageneration/output/final.other lda", false);
		System.out.println("done.");
	}

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
	
	private double[][] readBeta(String filename, boolean exp) {
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
	
}