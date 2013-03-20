package plusone.clustering;

import plusone.Main;
import plusone.utils.ItemAndScore;
import plusone.utils.PredictionPaper;
import plusone.utils.RunInfo;
import plusone.utils.SVD;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;

import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

public class LSIHO extends ClusteringTest {

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
    protected Terms terms;
    protected SVD svd;
    public int numTerms;

    public LSIHO(int DIMENSION, List<TrainingPaper> trainingSet, Terms terms, double testWordPercent) {
        this(DIMENSION, trainingSet, terms, testWordPercent, "LSIHO");
    }

    public LSIHO(int DIMENSION, List<TrainingPaper> trainingSet, Terms terms, double testWordPercent, String nameBase) {
	super(nameBase + "-" + DIMENSION);
	this.DIMENSION = DIMENSION;
	numTerms=terms.size();

	svd = new SVD(DIMENSION, trainingSet, numTerms);
	svd.renormalizeHeldOut(testWordPercent);
    }
    @Override
    public double[] predict(PredictionPaper testPaper, RunInfo testInfo) {
        long startNanoTime = System.nanoTime();
        double[] ret = svd.predict(testPaper);
        testInfo.put("testTime", (System.nanoTime() - startNanoTime) / 1.0e9);
        return ret;
    }
 
    public double[] getSingularValues() {
        return svd.getSingularValues();
    }

    @Override
    public double getTrainTime() {
        return svd.getSVDTimeNano() / 1.0e9;
    }
}
