package plusone.clustering;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import org.ejml.simple.SimpleMatrix;

import Jama.Matrix;
import Jama.QRDecomposition;

import plusone.utils.Indexer;
import plusone.utils.PaperAbstract;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.PredictionPaper;
import plusone.utils.RunInfo;
import plusone.utils.SVD;
import plusone.utils.Kmeans;
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
	private int numTerms;
	private Matrix betaMatrix;
	private double[][] beta;

	private int numDoc;
	private Random rand=new Random();
    protected SVD svd;
    private Kmeans kmeans;
	private boolean synthetic;
	private double trainSeconds = Double.POSITIVE_INFINITY;
	private long projectorTime;
	private final double ALPHA=0.9;
	

	public Projector(String name) {
		super(name);
	}
	
	public Projector(String name, List<TrainingPaper> trainingSet, 
			Indexer<String> wordIndexer,
			Terms terms, 
			int numTopics, 
			boolean synthetic) {
		this(name + "-" + numTopics);
		this.name = name;
		this.trainingSet = trainingSet;
		numDoc=trainingSet.size();
		this.wordIndexer = wordIndexer;
		this.terms = terms;
		this.numTopics=numTopics;
		this.synthetic = synthetic;
		numTerms=terms.size();
		train();
	}
	
	private double normalize(double[] x) {
		double lengthx = 0;
		for (int i = 0; i < x.length; i ++)
			lengthx += x[i] * x[i];
		lengthx = Math.sqrt(lengthx);

		for (int i = 0; i < x.length; i ++)
			x[i] /= lengthx;

		return lengthx;
	}

	/**
	 * Runs the MATLAB program found in projector/predictTopics.m
	 */
	private void train() {
		long startNanoTime = System.nanoTime();
		System.out.println("Projector: svd step");
		svd = new SVD(numTopics-1, trainingSet, numTerms,true);
		Matrix projMatrix=(new Matrix(svd.getV())).transpose();
		double[][] docProj=projMatrix.getArray();
		double[] sigma=svd.getSingularValues();
		
		
		for (int i=0;i<sigma.length;i++)
			for (int j=0;j<numDoc;j++){
				docProj[j][i]*=sigma[i];
			}
		
		System.out.println("Projector: kmeans step");
		kmeans = new Kmeans(docProj,numTopics, numTopics-1, 
				"cluster", "cosine", 20);
		kmeans.runKmeans();
		
		System.out.println("Projector: scaling step");
		
		double[][] centers=kmeans.getCenters();
		double[][] hyperplane=new double[numTopics][numTopics-1];
		double[] bounds = new double[numTopics];
		
		double[][] X=new double[numTopics-1][numTopics-2];
		
		for (int i=0;i<numTopics;i++){
			for (int j=0;j<numTopics-2;j++)
				for (int l=0;l<numTopics-1;l++)
					X[l][j]=centers[j+((i<=j+1)?2:1)][l]-centers[(i==0)?1:0][l];
			Matrix XMatrix=new Matrix(X);
			QRDecomposition QR=new QRDecomposition(XMatrix);
			double[][] normal= new double[1][numTopics-1];
			for (int j=0;j<numTopics-1;j++)
				normal[0][j]=rand.nextDouble();
			Matrix nrml=new Matrix(normal);
			normal=nrml.minusEquals(
					nrml.times(QR.getQ()).times(QR.getQ().transpose())).getArray();
			normalize(normal[0]);
			for (int j=0;j<numTopics-1;j++){
				hyperplane[i][j]=normal[0][j];
				bounds[i]+=hyperplane[i][j]*centers[(i==0)?1:0][j];
			}
			if (bounds[i]<0){
				bounds[i]*=-1;
				for (int j=0;j<numTopics-1;j++)
					hyperplane[i][j]*=-1;
			}
			double test=0;
			for (int j=0;j<numTopics-1;j++)
				test+=hyperplane[i][j]*centers[i][j];
			if (test>0)
				System.out.println("not a well defined convex hull!");			
		}
		double[][] scaleMatrix=projMatrix.times((new Matrix(hyperplane)).transpose()).getArray();
		int omit=(int)(this.numDoc*(1-ALPHA));
		PriorityQueue<Double> scales =new PriorityQueue<Double>(omit+1);
		for (int i=0;i<numDoc;i++){
			double nearestHP=scaleMatrix[i][0]/bounds[0];
			for (int j=1;j<numTopics;j++)
				nearestHP=Math.max(nearestHP, scaleMatrix[i][j]/bounds[j]);
			if (scales.size()<(omit+1))
				scales.add(nearestHP);
			else if (nearestHP>scales.peek()){
				scales.poll();
				scales.add(nearestHP);
			}
		}
		double scale=Math.max(scales.poll(),1);
		Matrix centersMatrix=(new Matrix(centers)).timesEquals(scale);
		
		System.out.println("Projector: recover topics");
		betaMatrix=centersMatrix.times(new Matrix(svd.getU()));
		beta=betaMatrix.getArray();
		double[] centroid = svd.getCentroid();
		for (int i=0;i<numTopics;i++){
			double norm1=0;
			for (int j=0;j<this.numTerms;j++){
				beta[i][j]=Math.max(0, beta[i][j]+centroid[j]);
				norm1+=beta[i][j];
			}
			
			for (int j=0;j<this.numTerms;j++)				
				beta[i][j]=beta[i][j]/norm1;
			
		}

		System.out.println("cleaning out projector folder for training...");
		
		if (!new File("projector/data").exists()) {
            new File("projector/data").mkdir();
        }
		Utils.runCommand("rm projector/data/final.beta", false);
		writeBeta("projector/data/final.beta",true);
		
		this.projectorTime=System.nanoTime()- startNanoTime;
		System.out.format("[Projector] took %.3f seconds.\n",
				  projectorTime/1.0e9);
		
		

        System.out.println("done.");
        
	}

    @Override
    public double getTrainTime() {
        return trainSeconds;
    }


    private void writeBeta(String filename,boolean takingLog){
        System.out.print("Output Projector learned topics in file: " 
        + filename + " ... ");

        PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);

        for (int i=0;i<numTopics;i++) {
            for (int j=0;j<numTerms;j++) {
            	if (takingLog)
                	fileWriter.write((Math.log(beta[i][j]+Double.MIN_VALUE))+(j==numTerms-1?"":" "));
            	else
            		fileWriter.write(beta[i][j]+(j==numTerms-1?"":" "));
            }
            fileWriter.write("\n");
        }

    fileWriter.close();

    System.out.println("done.");
    }
    @Override
	public double[][] predict(List<PredictionPaper> testDocs, RunInfo testInfo){
		double[][] testDocsArray=new double[testDocs.size()]
                [terms.size()];
		int i=0;
		for (PredictionPaper paper:testDocs){
			
			for (Integer word : paper.getTrainingWords())
				testDocsArray[i][word] = paper.getTrainingTf(word);
			i++;
			}

    	double[][] scores=(betaMatrix.solveTranspose(new Matrix(testDocsArray))).transpose()
    			.times(betaMatrix).getArray();
    			
    	return scores;
    	
    	
    
    }
	
}
