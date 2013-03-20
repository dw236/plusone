package plusone.utils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

import plusone.Main;

public class KNNGraphDistanceCache {
    
    public static class KeyPair {
	Integer key1, key2, value;
	public KeyPair(Integer key1, Integer key2, Integer value) {
	    this.key1 = key1;
	    this.key2 = key2;
	    this.value = value;
	}

	public boolean equals(Object obj) {
	    if (!(obj instanceof KeyPair)) {
		return false;
	    }
	    KeyPair kp = (KeyPair)obj;
	    return (kp.key1 == key1 && kp.key2 == key2);
	}
    }

    private int[] trainPaperKeys;
    private int[] testPaperKeys;
    private int[] testPaperNotFound;
    private List<KeyPair>[] distances;
    private Indexer<PaperAbstract> paperIndexer;
    private List<KeyPair>[] commonNeighbors;

    private Indexer<Integer> trainingIndexer;
    private Indexer<Integer> testingIndexer;

    public final int BUCKETS = 10000;

    public KNNGraphDistanceCache(List<TrainingPaper> trainingPapers,
				 List<PredictionPaper> testingPapers,
				 Indexer<PaperAbstract> paperIndexer) {
	trainPaperKeys = new int[trainingPapers.size()];
	testPaperKeys = new int[testingPapers.size()];
	testPaperNotFound = new int[testingPapers.size()];
	distances = new List[BUCKETS];
	commonNeighbors = new List[BUCKETS];
	this.paperIndexer = paperIndexer;
	trainingIndexer = new Indexer<Integer>();
	testingIndexer = new Indexer<Integer>();

	for (int i = 0; i < trainingPapers.size(); i ++) {
	    trainingIndexer.add(trainingPapers.get(i).getIndex());
	    trainPaperKeys[i] = -1;
	}

	for (int i = 0; i < testingPapers.size(); i ++) {
	    testingIndexer.add(testingPapers.get(i).getIndex());
	    testPaperKeys[i] = -1;
	}

	long t1 = System.currentTimeMillis();
	System.out.println("[GraphDistanceCache] filling cache" );
	precompute(trainingPapers, testingPapers, paperIndexer);
	System.out.format("[GraphDistanceCache] took %.3f seconds.\n",
			  (System.currentTimeMillis() - t1) / 1000.0); 
    }

    private void precompute(List<TrainingPaper> trainingPapers,
			    List<PredictionPaper> testingPapers,
			    Indexer<PaperAbstract> paperIndexer) {
	for (PredictionPaper testpaper : testingPapers) {
	    precomputeSingleTestpaper(trainingPapers, testpaper, 
				      paperIndexer);
	}
    }

    private void precomputeSingleTestpaper
	(List<TrainingPaper> trainPapers, PredictionPaper testPaper,
	 Indexer<PaperAbstract> paperIndexer) {
	Queue<Integer> currentQueue = new LinkedList<Integer>();
	Queue<Integer> nextQueue = new LinkedList<Integer>();
	Set<Integer> kSet = new HashSet<Integer>();
	Set<Integer> doneSet = new HashSet<Integer>();

	for (Integer currentPaper : testPaper.getInReferences()) {
	    currentQueue.add(currentPaper);
	}

	for (Integer currentPaper : testPaper.getOutReferences()) {
	    currentQueue.add(currentPaper);
	}

	for (TrainingPaper trainPaper : trainPapers) {
	    kSet.add(trainPaper.getIndex());
	}

	int d = 1, maxD = 1, testPaperIndex = testPaper.getIndex();

	doneSet.add(testPaper.getIndex());
	
	while (!currentQueue.isEmpty()) {
	    for (Integer currentPaper : currentQueue) {
		if (doneSet.contains(currentPaper))
		    continue;

		if (kSet.contains(currentPaper))
		    add(currentPaper, testPaperIndex, d);

		maxD = Math.max(maxD, d);

		doneSet.add(currentPaper);
		 
		PaperAbstract paper = paperIndexer.get(currentPaper);
		for (Integer neighbors : paper.getInReferences()) {
		    nextQueue.add(neighbors);
		}
		for (Integer neighbors : paper.getOutReferences()) {
		    nextQueue.add(neighbors);
		}
	    }
	    d ++;
	    currentQueue = nextQueue;
	    nextQueue = new LinkedList<Integer>();
	}

	kSet.removeAll(doneSet);
	testPaperNotFound[testingIndexer.indexOf(testPaperIndex)] = maxD + 1;
    }

    private void add(int key1, int key2, int value) {

	key1 = trainingIndexer.indexOf(key1);
	key2 = testingIndexer.indexOf(key2);
	
	while (trainPaperKeys[key1] == -1) {
	    trainPaperKeys[key1] = Main.getRandomGenerator().nextInt();
	}
	
	while (testPaperKeys[key2] == -1) {
	    testPaperKeys[key2] = Main.getRandomGenerator().nextInt();
	}

	int index = hashcode(trainPaperKeys[key1], 
			     testPaperKeys[key2]) % BUCKETS;
	if (distances[index] == null)
	    distances[index] = new ArrayList<KeyPair>();

	distances[index].add(new KeyPair(key1, key2, value));
       
	if (value == 2) {
	    if (commonNeighbors[index] == null)
		commonNeighbors[index] = new ArrayList<KeyPair>();

	    //PaperAbstract paper1 = paperIndexer.get(key1);
	    //PaperAbstract paper2 = paperIndexer.get(key2 + trainPaperKeys.length);
	    PaperAbstract paper1 = paperIndexer.get(trainingIndexer.get(key1));
	    PaperAbstract paper2 = paperIndexer.get(testingIndexer.get(key2));

	    Set<Integer> s1 = new HashSet<Integer>();
	    for (Integer r : paper1.getInReferences()) {
		s1.add(r);
	    }
	    for (Integer r : paper1.getOutReferences()) {
		s1.add(r);
	    }

	    Set<Integer> s2 = new HashSet<Integer>();
	    for (Integer r : paper2.getInReferences()) {
		s2.add(r);
	    }
	    for (Integer r : paper2.getOutReferences()) {
		s2.add(r);
	    }

	    s1.retainAll(s2);
	    commonNeighbors[index].add(new KeyPair(key1, key2, s1.size()));
	}
    }

    public int getCommonNeighbors(int key1, int key2) {
	key2 -= trainPaperKeys.length;
	List<KeyPair> lst = 
	    commonNeighbors[hashcode(trainPaperKeys[key1], 
				     testPaperKeys[key2]) % BUCKETS];

	if (lst == null)
	    return 0;

	for (KeyPair kp : lst) {
	    if (kp.key1 == key1 && kp.key2 == key2)
		return kp.value;
	}
	return 0;
    }

    public int get(int key1, int key2) {
	return getValue(trainingIndexer.indexOf(key1),
			testingIndexer.indexOf(key2));
    }

    private int getValue(int key1, int key2) {
	List<KeyPair> lst = 
	    distances[hashcode(trainPaperKeys[key1], 
			       testPaperKeys[key2]) % BUCKETS];

	if (lst == null)
	    return 1;

	for (KeyPair kp : lst) {
	    if (kp.key1 == key1 && kp.key2 == key2)
		return kp.value;
	}

	return testPaperNotFound[key2];
    }

    private int hashcode(int key1, int key2) {
	return Math.abs(key1 ^ key2);
    }
}
