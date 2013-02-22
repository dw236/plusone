package plusone.clustering;

import plusone.utils.PredictionPaper;
import plusone.utils.RunInfo;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;

import java.util.List;

public class Baseline extends ClusteringTest {
    
    private List<TrainingPaper> trainingSet;
    private Terms terms;
    double[] ret;
    long trainTimeNano;

    public Baseline(List<TrainingPaper> trainingSet, Terms terms) {
	super("Baseline");
	this.trainingSet = trainingSet;
	this.terms = terms;
	long startNanoTime = System.nanoTime();
	ret = new double[terms.size()];
	for (int i=0;i<terms.size();i++) {
		int id=terms.get(i).id;
		if (id!=i)
			System.out.println("in Baseline: the id of term is not the same as the index " +
					"of it in the terms array!!");
		double freq = terms.get(i).totalCount;
		/*
	    if (testPaper.getTrainingTf(id) != 0.0)
	    	freq=0.0;*/
	    ret[id]=freq;
		}
	trainTimeNano = System.nanoTime() - startNanoTime;
    }

    @Override
    public double getTrainTime() {
	return trainTimeNano / 1.0e9;
    }

    @Override
    public double[] predict(PredictionPaper testPaper, RunInfo testInfo) {
	testInfo.put("testTime", 0.0);
	return ret;
    }
}
