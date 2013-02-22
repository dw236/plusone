package plusone.clustering;

import plusone.utils.Indexer;
import plusone.utils.KNNSimilarityCache;
import plusone.utils.PaperAbstract;
import plusone.utils.PredictionPaper;
import plusone.utils.RunInfo;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;

import java.util.List;

public class KNN extends ClusteringTest {

    protected List<TrainingPaper> trainingSet;
    protected int K_CLOSEST;
    protected Indexer<PaperAbstract> paperIndexer;
    protected Terms terms;
    protected KNNSimilarityCache similarityCache;
    
    public KNN(int K_CLOSEST, List<TrainingPaper> trainingSet, 
	       Indexer<PaperAbstract> paperIndexer, Terms terms,
	       KNNSimilarityCache similarityCache) {
	super("knn-" + K_CLOSEST);
	this.K_CLOSEST = K_CLOSEST;
	this.trainingSet = trainingSet;
	this.terms = terms;
	this.similarityCache = similarityCache;
	this.paperIndexer = paperIndexer;
    }

    @Override
    public double getTrainTime() {
	return similarityCache.getRuntime();
    }
    
    @Override
    public double[] predict(PredictionPaper testPaper, RunInfo testInfo) {
	long startNanoTime = System.nanoTime();

	Integer[] kList = kNbr(testPaper, K_CLOSEST);
	
	double[] ret = new double[terms.size()];
	
	for (int i = 0; i < kList.length; i++){
	    Integer paperIndex = kList[i];
	    TrainingPaper a = paperIndexer.get(paperIndex);

	    for (Integer word : a.getTrainingWords()) {
		ret[word] += a.getTrainingTf(word);
	    }
	}

	testInfo.put("testTime", (System.nanoTime() - startNanoTime) / 1.0e9);

	return ret;
    }
   /* 
    protected List<Integer> predictTopKWordsWithKList
	(Integer[] kList, PredictionPaper testDoc, int k) {
	
	int[] count = new int[terms.size()];
	List<Integer> wordSet = new ArrayList<Integer>();
	
	for (int i = 0; i < kList.length; i++){
	    Integer paperIndex = kList[i];
	    TrainingPaper a = paperIndexer.get(paperIndex);

	    for (Integer word : a.getTrainingWords()) {
		if (count[word] == 0)
		    wordSet.add(word);
		count[word] += a.getTrainingTf(word);
	    }
	}
	
	Queue<ItemAndScore> queue = new PriorityQueue<ItemAndScore>(k + 1);
	for (Integer word : wordSet) {
	    if (testDoc.getTrainingTf(word) > 0.0)
	    	continue;

	    if (queue.size() < k || 
		(double)count[word] > queue.peek().score) {
		if (queue.size() >= k)
		    queue.poll();
		queue.add(new ItemAndScore(word, count[word], true));
	    }
	}
	
	List<Integer> results = new ArrayList<Integer>();
	while (!queue.isEmpty()) {
	    results.add((Integer)queue.poll().item);
	}
	
	return results;
    }	
     */   
    /**
     * Gets the k closest neighbors using the similarity function
     * defined in PaperAbstract.
     */
    /*
    public Integer[] kNbr1(PredictionPaper doc, int K_CLOSEST){
	PriorityQueue<ItemAndScore> queue = 
	    new PriorityQueue<ItemAndScore>(K_CLOSEST + 1);
	
	for (int i = 0; i < trainingSet.size(); i++) {
	    TrainingPaper a = trainingSet.get(i);
	    double sim = ((PaperAbstract)doc).similarity((PaperAbstract)a);
	    
	    if (queue.size() < K_CLOSEST || sim > queue.peek().score) {
		if (queue.size() >= K_CLOSEST) {
		    queue.poll();
		}
		queue.add(new ItemAndScore(paperIndexer.fastIndexOf(a), 
					   sim, true));
	    }
	}

	Integer[] results = new Integer[Math.min(K_CLOSEST, queue.size())];
	for (int i = 0; i < K_CLOSEST && !queue.isEmpty(); i ++) {
	    results[i] = ((Integer) queue.poll().item);
	}
	
	return results;
    }*/

    public Integer[] kNbr(PredictionPaper doc, int K_CLOSEST) {
	Integer[] allRank = similarityCache.getDistance(doc);
	Integer[] rank = new Integer[Math.min(K_CLOSEST + 1, allRank.length)];

	for (int i = 0; i < rank.length; i ++) {
	    rank[i] = allRank[i];
	}

	return rank;
    }
}
