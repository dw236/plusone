package plusone.utils;

import plusone.Main;
import plusone.utils.TrainingPaper;
import plusone.utils.PredictionPaper;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;


public class SVD {

	class Entry{
		public int docID;
		public int termID;
		public double value;

		public Entry(int docID, int termID, double value) {
			this.docID = docID;
			this.termID = termID;
			this.value = value;
		}
	}

	protected List<TrainingPaper> trainingSet;
	protected LinkedList<Entry>[] DocTerm;
	protected LinkedList<Entry>[] TermDoc;
	protected int DIMENSION;
	protected double[][] mu;
	protected double[][] beta;
	protected double[] sigma;
	protected boolean subtractMean;
	double[] centroid;
	private int numTerms;
	private int numDocs;
	private Random rand=new Random();
	boolean shift=false;
	final long svdTimeNano;

	/* Set to true to bring back an old bug. */
	protected boolean oldBehavior0 = false;

	public SVD(int DIMENSION, List<TrainingPaper> trainingSet, int numTerms, boolean shift) {

		this.DIMENSION = DIMENSION;
		this.trainingSet = trainingSet;
		this.numTerms = numTerms;
		this.numDocs=trainingSet.size();
		this.shift=shift;

		long startNanoTime = System.nanoTime();
		System.out.println("[SVD] training with " + DIMENSION + 
				" dimension.");

		mu = new double[DIMENSION][numDocs];
		beta = new double[DIMENSION][numTerms];
		sigma = new double[DIMENSION];
		centroid=new double[numTerms];
		DocTerm = new LinkedList[numDocs];
		TermDoc = new LinkedList[numTerms];
		for (int i = 0; i < numDocs; i ++) {
			TrainingPaper doc = trainingSet.get(i);
			int length=doc.getTrainingLength();
			DocTerm[i] = new LinkedList<Entry>();

			for (Integer word : doc.getTrainingWords()) {
				Entry temp = new Entry(i, word, ((double)doc.getTrainingTf(word))/length);
				DocTerm[i].add(temp);
				centroid[word]+=((double)doc.getTrainingTf(word))/length/numDocs;
				if (TermDoc[word] == null){
					TermDoc[word] = new LinkedList<Entry>();
				}
				TermDoc[word].add(temp);
			}
		}

		this.train();

		svdTimeNano = System.nanoTime() - startNanoTime;
		System.out.format("[SVD] took %.3f seconds.\n",
				  svdTimeNano/1.0e9);
		
	}
	public double[] getCentroid(){
		return this.centroid;
	}
	public long getSVDTimeNano() {
	    return svdTimeNano;
	}

    public void setOldBehavior0(boolean b) {
        oldBehavior0 = b;
    }

/*	private Map<Integer, Double> getReducedDocument(int index) {
		Map<Integer, Double> result = new HashMap<Integer, Double>();
		for (int i = 0; i < DIMENSION; i ++) {
			double tf = mu[i][index];
			if (tf != 0.0) {
				result.put(i, tf);
			}
		}
		return result;
	}*/

	public double dotProduct(double[] a, double[] b){
		double result = 0.0;
		for (int i = 0; i < a.length; i ++){
			result += a[i] * b[i];
		}
		return result;
	}
	public double sum(double[] a){
		double result=0;
		for (double v : a)
			result+=v;
		return result;
	}

	public void powerMethod(double[] x, double[] y, int k){
		for (int j = 0; j < y.length; j ++)
			y[j] = 1.0/Math.sqrt(y.length);

		double xnorm;
		double ynorm;
		double diff = dotProduct(x, x) * dotProduct(y, y);
		double xsum=0;
		boolean converge = false;
		while (!converge){
			double[] subtract = new double[k+1];
			xnorm = dotProduct(x, x);
			if (shift)
				xsum=sum(x);
			if (xnorm <= 0.0001)
				break;
			
			for (int i = 0; i < k; i ++)
				subtract[i] = dotProduct(mu[i], x);

			for (int i = 0; i < y.length; i ++){
				double value = 0;
				if (TermDoc[i] != null) {
					for (Entry t : TermDoc[i]) {
						value += t.value * x[t.docID];
					}
				}
				if (shift)
					value-=centroid[i]*xsum;
				for (int j = 0; j < k; j ++)
					value -= beta[j][i] * sigma[j] * subtract[j];

				y[i] = value / xnorm;
			}
			ynorm = dotProduct(y, y);

			if (ynorm <= 0.0001)
				break;

			
			for (int i = 0; i < k; i ++){
				subtract[i] = dotProduct(beta[i], y);
			}
			double xshift=0;
			if (shift)
				xshift=dotProduct(y,centroid);
				
			for (int i = 0; i < x.length; i ++){
				double value = 0;
				for (Entry t : DocTerm[i]) {
					value += t.value * y[t.termID];
				}
				for (int j = 0; j < k; j ++)
					value -= mu[j][i] * sigma[j] * subtract[j];
				if (shift)
					value-=xshift;
				x[i] = value / ynorm;
			}

			double temp = dotProduct(x, x) * dotProduct(y, y);
			if (Math.abs(diff - temp) < .00001 * diff)
				converge = true;
			diff = temp;
		}

	}

	public void orthog(double[] x1, double[] x2) {
		double length = 0;
		for (int i = 0; i < x1.length; i ++)
			length += x1[i] * x2[i];

		for (int i = 0; i < x2.length; i ++)
			x2[i] -= length * x1[i];
	}

	public double normalize(double[] x) {
		double lengthx = 0;
		for (int i = 0; i < x.length; i ++)
			lengthx += x[i] * x[i];
		lengthx = Math.sqrt(lengthx);

		for (int i = 0; i < x.length; i ++)
			x[i] /= lengthx;

		return lengthx;
	}

	public void train(){
		for (int k = 0; k < DIMENSION; k ++){
			//start with random vector
			for(int i=0;i<mu[k].length;i++)
				mu[k][i]=rand.nextDouble();
			//make it orthogonal to previous vectors
			double sum=0;
			if (shift){
				sum=this.sum(mu[k]);
				for (int i=0;i<mu[k].length;i++)
					mu[k][i]-=sum/mu[k].length;
				}
			for (int i = 0; i < k; i ++) {
				orthog(mu[i],  mu[k]);
			}

			powerMethod(mu[k], beta[k], k);

			sigma[k] = 1;
			sigma[k]*=normalize(mu[k]);
			sigma[k]*=normalize(beta[k]); 

		}
	
	}


	private double similarity(int docId, int termId) {
		double result = 0;
		for (int i = 0; i < DIMENSION; i ++)
			result += mu[i][docId] * sigma[i] * beta[i][termId];
		return result;
	}
	
    /**
     * Projects a vector of word frequencies to topic space, and returns a
     * topic vector.
     *
     * In other words, given word frequencies w, returns t such that beta*t is
     * as close as possible to w in l2 distance.  Note that the columns of beta
     * are normalized.
     */
	public double[] projection(PaperAbstract paper){
		double[] doct = new double[numTerms];
		for (Integer word : paper.getTrainingWords()) {
			doct[word] = paper.getTrainingTf(word);
		}

		double[] dock = new double[DIMENSION];
		for (int i = 0; i < dock.length; i ++) {
            if (oldBehavior0) {
                /* This is what used to happen here. */
                dock[i] = dotProduct(doct, beta[i]) / sigma[i];
            } else {
                dock[i] = dotProduct(doct, beta[i]);	    
            }
		}
		return dock;
	}
	
	public double[] predict(PredictionPaper testPaper) {
	
		double[] ret = new double[numTerms];
		
		double[] dock=projection((PaperAbstract)testPaper);

		for (int i = 0; i < numTerms; i ++) {
			if (testPaper.getTrainingTf(i) > 0)
				continue;
			double score = 0.0;
			for (int j = 0; j < DIMENSION; j ++) {
				score += dock[j] * beta[j][i];
			}
			ret[i]=score;

		}

		return ret;
	}
	
	public double[] getSingularValues() {
		return sigma;
	}
	
	public double[][] getU() {
		return beta;
	}
	public double[][] getV(){
		return mu;
	}

    /**
     * Returns an array, each element of which is the word-frequency vector of
     * one topic, normalized to have l_2 norm equal to 1.
     */
    public double[][] getNormalizedTopics() {
        return beta;
    }
}
