package plusone;

import plusone.utils.DatasetJSON;
import plusone.utils.Indexer;
import plusone.utils.ItemAndScore;
import plusone.utils.KNNGraphDistanceCache;
import plusone.utils.KNNSimilarityCache;
import plusone.utils.KNNSimilarityCacheLocalSVDish;
import plusone.utils.MetadataLogger;
import plusone.utils.PaperAbstract;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.PredictionPaper;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;
import plusone.utils.LocalSVDish;
import plusone.utils.Results;
import plusone.utils.LocalCOSample;
import plusone.utils.Utils;

import plusone.clustering.*;
import plusone.clustering.held_out_inference.PoissonLDAPredictor;

import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Date;
import java.util.List;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import org.ejml.simple.SimpleMatrix;
import org.json.*;


public class Main {

	private static int nTrials;
	private static Indexer<String> wordIndexer;
	private static Indexer<PaperAbstract> paperIndexer;

	private static Terms terms;
	
	private static String dataFile;

	private static MetadataLogger metadataLogger;
	private static Random randGen;

	private HashMap<PaperAbstract, Integer> trainingIndices;
	private Map<PaperAbstract, Integer> testIndices;
	// Document sets
	public List<TrainingPaper> trainingSet;
	public List<PredictionPaper> testingSet;
	
	private static int numTopics;
	
	private static String generator;

	private static int FOLD; // cross validation parameter
	private static DatasetJSON dataset;
	
	private static boolean tagged;
	private static HashMap<Integer, ArrayList<Integer>> tagMap;

	static final String[] allResultFieldNames = {
	    "predictionRate", "tfScore", "tfidfScore",
	};
	private Map<String,Results>[] allResults;

	public static void load_data(String filename) {
		dataset = DatasetJSON.loadDatasetFromPath(filename);
		tagged = dataset.isTagged(filename);
		tagMap = dataset.getTagMap();
		wordIndexer = dataset.getWordIndexer();
		PlusoneFileWriter wordMap = new PlusoneFileWriter("data/wordMap.txt");
		for (int i = 0; i < wordIndexer.size(); i++) {
			wordMap.write(wordIndexer.get(i) + "\n");
		}
		wordMap.close();
		paperIndexer = dataset.getPaperIndexer();
	}

	private void setupData(int testGroup, double testWordPercent){
		System.out.println("Preparing data...");
		List<PaperAbstract> documents = dataset.getDocuments();
		// split into training documents and testing documents
		trainingSet=new ArrayList<TrainingPaper>();
		testingSet = new ArrayList<PredictionPaper>();

//		for (int i = 0; i < documents.size(); i ++) {
//			if (tagged) {
//				//Because documents.get(i).getIndex() == i
//				if (tagMap.keySet().contains(i)) {
//					testingSet.add((PredictionPaper)documents.get(i));
//					testIndices.put(documents.get(i), i);
//				} else {
//					trainingSet.add((TrainingPaper)documents.get(i));
//					trainingIndices.put(documents.get(i), i);
//				}
//			} else {
//				if (documents.get(i).getGroup()==testGroup) {
//					testingSet.add((PredictionPaper)documents.get(i));
//					testIndices.put(documents.get(i), i);
//				} else {
//					trainingSet.add((TrainingPaper)documents.get(i));
//					trainingIndices.put(documents.get(i), i);
//				}
//			}
//		}
//		
//		System.out.println("Training size:" + trainingSet.size());
//		System.out.println("Testing size:" + testingSet.size());

		double[] trainPercents = 
			parseDoubleList(System.getProperty("plusone.trainPercents", 
					"0.3,0.5"));
		System.out.println("train percent: " + trainPercents[0]);
		splitByTrainPercent(trainPercents[0], documents);
		
		// Held out words
		Terms.Term[] terms = new Terms.Term[wordIndexer.size()];
		for (int i = 0; i < wordIndexer.size(); i++) {
			terms[i] = new Terms.Term(i);
		}

		if (tagged) {
			for (TrainingPaper a : trainingSet){
				((PaperAbstract)a).generateTf(testWordPercent, terms, false);
			}
	
			for (PredictionPaper a : testingSet){
				((PaperAbstract)a).generateTagTf(tagMap.get(a.getIndex()),
					testWordPercent, terms);
			}
		} else {
			for (TrainingPaper a : trainingSet){
				((PaperAbstract)a).generateTf(testWordPercent, terms, false);
			}
	
			for (PredictionPaper a : testingSet){
				((PaperAbstract)a).generateTf(testWordPercent, null, true);
			}
		}
		
		//Find average number of non-heldout words in each of the test docs
		double avgWordsPerTestDoc = 0.0;
		for (PredictionPaper p : testingSet) {
			for (Integer i : p.getTrainingWords()) {
				avgWordsPerTestDoc += p.getTrainingTf(i);
			}
		}
		
		avgWordsPerTestDoc /= testingSet.size();
		System.out.println("Average number of words per test doc:" + avgWordsPerTestDoc);

		this.terms = new Terms(terms);
		System.out.println("Data ready for experiment");
	}

	private void runExperiments(String path) {
		Boolean crossValid = Boolean.getBoolean("plusone.crossValidation.run");

		if (crossValid)
			System.out.println("We will do "+FOLD+"-fold cross validation");
		double[] testWordPercents = 
				parseDoubleList(System.getProperty("plusone.testWordPercents", 
						"0.3,0.5"));
		int[] ks = 
				parseIntList(System.getProperty("plusone.kValues", 
						"1,3,5,10"));
		Arrays.sort(ks);
		allResults = new Map[ks.length];
		for (int twp = 0; twp < testWordPercents.length; twp++) {
			double testWordPercent = testWordPercents[twp];
			System.out.println("processing testwordpercent: " + testWordPercent);
			for (int i=0;i<ks.length;i++)
				allResults[i]=new HashMap<String,Results>();

			for (int testGroup=(crossValid ?0:FOLD-1);testGroup<FOLD;testGroup++){	    
				setupData(testGroup,testWordPercent);

				runClusteringMethods(ks, testWordPercent);

			}
		}
		outputResults(ks, testWordPercents);
	}

	private String getLegacyResultFieldName(String fieldName) {
	    if (fieldName.equals("predictionRate"))
		return "Predicted";
	    else if (fieldName.equals("tfScore"))
	        // Preserving the old field, even though it looks like the name is a mistake...
		return "idf score";
	    else if (fieldName.equals("tfidfScore"))
		return "tfidf score";
	    else
		return null;
	}

	private void addJSONMeansOrVariances(
		Map<String, Double> values, String suffix, JSONObject testResults)
	    throws JSONException {
	    for (Map.Entry<String, Double> fieldValue : values.entrySet()) {
		String fieldName = fieldValue.getKey();
		testResults.put(fieldName + suffix, fieldValue.getValue());
		String legacyResultFieldName = getLegacyResultFieldName(fieldName);
		if (null != legacyResultFieldName)
		    testResults.put(legacyResultFieldName + suffix, fieldValue.getValue());
	    }
	}

	private JSONObject genOneTestJSON(int k, double twpName, Results results) throws JSONException {
		JSONObject testResults = new JSONObject();
		testResults.put("numPredictions", k);
		testResults.put("trainPercent", twpName);
		addJSONMeansOrVariances(results.getResultsMean(), "_Mean", testResults);
		addJSONMeansOrVariances(results.getResultsVariance(), "_Var", testResults);
		return testResults;
	}

	/* Makes a fake experiment with cosine similarities.
	 */
	private void AddCosineSimilarityFakeExperiments(JSONObject allTests) throws JSONException {
		if (!generator.equals("") && testIsEnabled("lda")
				&& testIsEnabled("projector")) {
			Scanner in = null;
			Utils.runCommand("python parse_betas.py data/normfile -s", false);
			try {
				in = new Scanner( new File( "data/normfile" ) );
			} catch (Exception e) {
				System.out.println("Couldnt find cosine similarity file");
			}
			String[] cosineSimilarities = in.nextLine().split(" ");
			double cosineSimilarityMean = 0;
			for (String sim : cosineSimilarities) {
				cosineSimilarityMean += Double.parseDouble(sim);
			}
			cosineSimilarityMean /= cosineSimilarities.length; 
			JSONObject fakeExperiment = new JSONObject();
			fakeExperiment.put("Predicted_Mean", cosineSimilarityMean);
			allTests.put("~projector-cosine", fakeExperiment);
			
			String[] cosineSimilaritiesLDA = in.nextLine().split(" ");
			double cosineSimilarityMeanLDA = 0;
			for (String sim : cosineSimilaritiesLDA) {
				cosineSimilarityMeanLDA += Double.parseDouble(sim);
			}
			cosineSimilarityMeanLDA /= cosineSimilaritiesLDA.length; 
			JSONObject fakeExperimentLDA = new JSONObject();
			fakeExperimentLDA.put("Predicted_Mean", cosineSimilarityMeanLDA);
			allTests.put("~lda-cosine", fakeExperimentLDA);
			
			String[] cosineSimilaritiesMallet = in.nextLine().split(" ");
			double cosineSimilarityMeanMallet = 0;
			for (String sim : cosineSimilaritiesMallet) {
				cosineSimilarityMeanMallet += Double.parseDouble(sim);
			}
			cosineSimilarityMeanMallet /= cosineSimilaritiesMallet.length; 
			JSONObject fakeExperimentMallet = new JSONObject();
			fakeExperimentMallet.put("Predicted_Mean", cosineSimilarityMeanMallet);
			allTests.put("~mallet-cosine", fakeExperimentMallet);
		}
	}

	private JSONObject genTestsJSONForTWPAndK(int ki, int k, double twpName) throws JSONException {
		JSONObject allTests = new JSONObject();
		Map<String,Results> resultK=allResults[ki];
		for (Map.Entry<String,Results> entry : resultK.entrySet()){
			allTests.put(entry.getKey(), genOneTestJSON(k, twpName, entry.getValue()));
		}

		AddCosineSimilarityFakeExperiments(allTests);

		return allTests;
	}

    private JSONArray genTestsJSON(int[] ks, double[] twpNames) throws JSONException {
		JSONArray tests = new JSONArray();
		for (int i = 0; i < twpNames.length; i++) {
			for(int ki=0;ki<ks.length;ki++){
				tests.put(genTestsJSONForTWPAndK(ki, ks[ki], twpNames[i]));
			}
		}
        return tests;
    }

    private JSONObject genResultsJSON(int[] ks, double[] twpNames) throws JSONException {
		JSONObject json = new JSONObject();

		JSONObject dataList = new JSONObject();
		JSONObject parameters = new JSONObject();
		if (generator.equals("")) {
			//Real data
			parameters.put("k", numTopics);
		} else {
			//Synthetic data
			putInfo(parameters, dataList);
		}
		json.put("parameters", parameters);
		json.put("data", dataList);
		
		json.put("tests", genTestsJSON(ks, twpNames));

		return json;
    }

	/** 
	 * Outputs the results of the tests into the data folder.
	 * 
	 * @param ks an array containing how many words each test should predict
	 * @param twpNames an array containing the percentage of held out words for each test
	 */
	private void outputResults(int[] ks, double[] twpNames) {
		String fileName = ""; String dirName = "";
		if (generator.equals("")) {
			//Real data
			String shortFile = dataFile.split("/")[1];
			fileName = "k." + numTopics + "."
					+ shortFile.substring(0,shortFile.length()-4);
		} else {
			//Find where alpha is so we can correctly split dir/file name
			int alphaLoc = 0;
			String tmpOutName = getOutputFileName();
			for (int i = 0; i < tmpOutName.length(); i++) {
				if (tmpOutName.charAt(i) == 'a') {
					alphaLoc = i;
				}
			}
			dirName = tmpOutName.substring(0, alphaLoc - 1);
			fileName = tmpOutName.substring(alphaLoc);
		}
		new File("data/" + dirName).mkdir();
		File out = new File("data/" + dirName, "experiment." + fileName + "json");
		if (out.exists()) {
			//Keep trying to append 0, 1, 2... until we find an unused file name
			int newFileEnd = 0;
			String oldFileName = fileName;
			while (out.exists()) {
				fileName = new String(oldFileName.concat(newFileEnd + "."));
				newFileEnd++;
				out = new File("data/" + dirName, "experiment." + fileName + "json");
			}
		}
		System.out.println("Wrote to data/" + dirName + "/experiment."
				+ fileName + "json");

		PlusoneFileWriter writer = new PlusoneFileWriter(out);
		try {
			writer.write(genResultsJSON(ks, twpNames).toString());
		} catch (JSONException e) {
			System.out.println("Error generating results JSON.");
		}
		writer.close();
	}
	
	/**
	 * Puts useful information into the output JSON, such as the parameters
	 * used to generate a synthetic dataset and statistics about the dataset
	 * 
	 * @param parameters empty when called, afterwards has alpha, beta, k, etc
	 * @param dataList empty when called, afterwards has sig_words, sig_topics, etc
	 */
	public void putInfo(JSONObject parameters, JSONObject dataList) {
		try {
			ArrayList<Double> params = new ArrayList<Double>();
			ArrayList<String> paramNames = new ArrayList<String>();
			ArrayList<Double> data = new ArrayList<Double>();
			ArrayList<String> dataNames = new ArrayList<String>();
	
			//Synthetic data
			File documentsOptionsOut = null, documentsOtherOut = null;
			documentsOptionsOut = new File(
					"src/datageneration/output/documents_options-out");
			if (!generator.equals("hlda")) {
				documentsOtherOut = new File(
						"src/datageneration/output/documents_other-out");				
			}
			//Put the information from documents_options-out into params
			Scanner line = null;
			try {
				line = new Scanner(new FileInputStream(documentsOptionsOut));
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			String[] parsedLine = line.nextLine().split(" ");
			for (int i = 0; i < (parsedLine.length-2)/2; i++) {
				params.add(Double.parseDouble(parsedLine[3+2*i]));
				paramNames.add(parsedLine[2+2*i].substring(1));
			}
			StringBuffer tmpOutName = new StringBuffer();
			for (int i = 0; i < params.size(); i++) {
				if (paramNames.get(i).equals("a") || paramNames.get(i).equals("b")) {
					tmpOutName.append(paramNames.get(i) + params.get(i) + ".");
				} else {
					tmpOutName.append(paramNames.get(i)
							+ (int)Math.floor(params.get(i)) + ".");
				}
			}
			
			if (documentsOtherOut != null) {
				//Put the information from documents_other-out into data
				Scanner lines = null;
				try {
					lines = new Scanner(new FileInputStream(documentsOtherOut));
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
				while (lines.hasNextLine()) {
					String[] nameAndValue = lines.nextLine().split(" ");
					dataNames.add(nameAndValue[0]);
					data.add(Double.parseDouble(nameAndValue[1]));
				}
				for (int i = 0; i < data.size(); i++) {
					dataList.put(dataNames.get(i), data.get(i));
				}
			}
			for (int i = 0; i < params.size(); i++) {
				if (paramNames.get(i).equals("a") || paramNames.get(i).equals("b")) {
					parameters.put(paramNames.get(i), params.get(i));
				} else {
					parameters.put(paramNames.get(i), (int)Math.floor(params.get(i)));
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Based on the parameters used to generate synthetic data, get the name
	 * of the output folder/file 
	 * 
	 * @return a string formatted like k15.n1000.l75.m1000.a0.01.b0.25
	 */
	public String getOutputFileName() {
		StringBuffer tmpOutName = new StringBuffer();
		try {
			ArrayList<Double> params = new ArrayList<Double>();
			ArrayList<String> paramNames = new ArrayList<String>();

			File documentsOptionsOut = new File(
					"src/datageneration/output/documents_options-out");
			Scanner line = new Scanner(new FileInputStream(documentsOptionsOut));
			String[] parsedLine = line.nextLine().split(" ");
			for (int i = 0; i < (parsedLine.length-2)/2; i++) {
				params.add(Double.parseDouble(parsedLine[3+2*i]));
				paramNames.add(parsedLine[2+2*i].substring(1));
			}
			for (int i = 0; i < params.size(); i++) {
				if (paramNames.get(i).equals("a") || paramNames.get(i).equals("b")) {
					tmpOutName.append(paramNames.get(i) + params.get(i) + ".");
				} else {
					tmpOutName.append(paramNames.get(i)
							+ (int)Math.floor(params.get(i)) + ".");
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return tmpOutName.toString();
	}

	void handleHeldOutInferenceTests(
			String nameBase, SimpleMatrix wordTopicMatrix, int[] ks, int size,
			double testWordPercent) {
		final String rateName =
				"plusone.documentLengthRate";
		final String alphaName =
				"plusone.topicAlpha";
		final String iterationsName =
				"plusone.poissonLda.numIterations";

		if (!testIsEnabled("heldOutPoissonLda"))
			return;

		double lambda;
		double alpha;
		int[] numsIterations;
		try {
			lambda = Double.parseDouble(System.getProperty(rateName));
			alpha = Double.parseDouble(System.getProperty(alphaName));
			numsIterations = parseIntList(System.getProperty(iterationsName));
		} catch (NullPointerException e) {
			throw new IllegalArgumentException(
				rateName + " and " + alphaName +
				" must be set if Poisson-LDA held-out inference is enabled.");
		}

		// Set all topic strengths to alpha.
		SimpleMatrix topicStrengths =
			new SimpleMatrix(wordTopicMatrix.numCols(), 1);
		topicStrengths.set(alpha);

		for (int numIterations : numsIterations) {
			ClusteringTest pldap = new PoissonLDAPredictor(
					nameBase, testWordPercent, lambda, numIterations,
					topicStrengths, wordTopicMatrix,
					PoissonLDAPredictor.PredictionMethod.WORD_DIST);
			runClusteringMethod(pldap, ks, size, true);
		}
	}

	public void runClusteringMethods(int[] ks, double testWordPercent) {
		int size = trainingSet.size() + testingSet.size();
		// Baseline
		if (testIsEnabled("baseline")) {
			Baseline baseline = new Baseline(trainingSet, terms);
			runClusteringMethod(baseline, ks, size, false);
		}

		// KNN
		KNNSimilarityCache knnSimilarityCache = null;
		if (testIsEnabled("knn") || testIsEnabled("knnc"))
			knnSimilarityCache = 
			new KNNSimilarityCache(trainingSet, testingSet);
		KNN knn;
		int[] closest_k =   parseIntList(System.getProperty("plusone.closestKValues", 
				"1,3,5,10,25,50,100,250,500,1000"));
		for (int ck = 0; ck < closest_k.length; ck ++) {
			if (testIsEnabled("knn")) {
				knn = new KNN(closest_k[ck], trainingSet, paperIndexer, 
						terms, knnSimilarityCache);
				runClusteringMethod(knn, ks, size, false);
			}
		}

		// Local Co-Occurance
		CO co;
		if (testIsEnabled("localCO")){
			co=new CO(Integer.getInteger("plusone.localCO.docEnzs"),
					Integer.getInteger("plusone.localCO.termEnzs"),
					Integer.getInteger("plusone.localCO.dtNs"),
					Integer.getInteger("plusone.localCO.tdNs"),
					trainingSet, terms);
			runClusteringMethod(co,ks,size,false);
		}
		// LSI
		LSI lsi;
		if (testIsEnabled("lsi")){
			int[] dimensions = parseIntList(System.getProperty("plusone.svdDimensions", 
					"10,30,50"));
			for (int dk = 0; dk < dimensions.length; dk ++) {

				lsi = new LSI(dimensions[dk], trainingSet, terms);

				runClusteringMethod(lsi, ks, size,false);

				LSIOld0 lsio0 = new LSIOld0(dimensions[dk], trainingSet, terms);

				runClusteringMethod(lsio0, ks, size,false);
			}
		}
		//PLSI
		PLSI plsi;
		if (testIsEnabled("plsi")){
			int[] dimensions = parseIntList(System.getProperty("plusone.plsi.dimensions", 
					"10,30,50"));
			plsi = new PLSI(trainingSet, terms.size());
			for (int dk = 0; dk < dimensions.length; dk ++) {
				long t1 = System.currentTimeMillis();
				System.out.println("PLSI with " + dimensions[dk]+" topics starts model training");
				plsi.train(dimensions[dk]);
				System.out.println("model training took " +
						(System.currentTimeMillis() - t1) / 1000.0 
						+ " seconds.");
				runClusteringMethod(plsi, ks, size, false);
			}
		}
		//lda
		if (testIsEnabled("lda")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
					"10,30,50"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				Lda lda = new Lda("lda", trainingSet, wordIndexer, terms, dimensions[dk],
						trainingIndices, testIndices);
				runClusteringMethod(lda, ks, size, true);
				handleHeldOutInferenceTests(
					lda.getName(), lda.getWordTopicMatrix(), ks, size,
					testWordPercent);
			}
		}
		
		//lda (Mallet)
		Mallet malletLda = null;
		if (testIsEnabled("malletLda")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
					"10,30,50"));
			int gibbsIterations = Integer.parseInt(System.getProperty("plusone.mallet.gibbsIterations", 
					"500"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				malletLda = new Mallet("lda", trainingSet, wordIndexer, terms, 
						dimensions[dk], gibbsIterations, !generator.equals(""));
				runClusteringMethod(malletLda, ks, size, true);
			}
		}
		
		//hlda (Mallet)
		Mallet malletHlda = null;
		if (testIsEnabled("malletHlda")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
					"10,30,50"));
			int gibbsIterations = Integer.parseInt(System.getProperty("plusone.mallet.gibbsIterations", 
					"500"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				malletHlda = new Mallet("hlda", trainingSet, wordIndexer, terms, 
						dimensions[dk], gibbsIterations, !generator.equals(""));
				runClusteringMethod(malletHlda, ks, size, true);
			}
		}
		
		//projector, uses a projection algorithm
		Projector projector = null;
		if (testIsEnabled("projector")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
			"20"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				projector = new Projector("projector", trainingSet, wordIndexer, 
								terms, dimensions[dk], trainingIndices, 
								testIndices);
				runClusteringMethod(projector, ks, size, true);
			}
		}
		
		//ldaT, cheats on training but not testing
		if (testIsEnabled("ldaTrained")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
					"20"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				Lda ldaTrained = new Lda("ldaT", trainingSet, wordIndexer, terms, dimensions[dk],
						trainingIndices, testIndices);
				runClusteringMethod(ldaTrained, ks, size, true);
				handleHeldOutInferenceTests(
					ldaTrained.getName(), ldaTrained.getWordTopicMatrix(),
					ks, size, testWordPercent);
			}
		}
		
		//ldaC, cheats in both training and testing
		if (testIsEnabled("ldaCheat")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
					"20"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				Lda ldaCheat = new Lda("ldaC", trainingSet, wordIndexer, terms, dimensions[dk],
						trainingIndices, testIndices);
				runClusteringMethod(ldaCheat, ks, size, true);
			}
		}
		
		//HLDA
		Hlda hlda = null;
		if (testIsEnabled("hlda")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
					"20"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				hlda = new Hlda(trainingSet, wordIndexer, terms);
				runClusteringMethod(hlda, ks, size, true);
			}
		}
		//GibbsLda
		if (testIsEnabled("gibbsLda")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
					"20"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				GibbsLda gibbs = new GibbsLda(trainingSet, wordIndexer, terms, dimensions[dk]);
				runClusteringMethod(gibbs, ks, size, true);
			}
		}
		//CTM
		Ctm ctm = null;
		if (testIsEnabled("ctm")){
			int[] dimensions = parseIntList(System.getProperty("plusone.lda.dimensions", 
					"20"));
			for (int dk = 0; dk < dimensions.length; dk ++) {
				ctm = new Ctm(trainingSet, wordIndexer, terms, dimensions[dk]);
				runClusteringMethod(ctm, ks, size, true);
			}
		}
		// KNNSVDish
		int[] closest_k_svdish = parseIntList(System.getProperty("plusone.closestKSVDishValues", 
				"1,3,5,10,25,50,100,250,500,1000"));
		KNNSimilarityCacheLocalSVDish KNNSVDcache = null;
		LocalSVDish localSVD;
		KNNLocalSVDish knnSVD;
		if (testIsEnabled("svdishknn")){
			localSVD=new LocalSVDish(Integer.getInteger("plusone.svdishknn.nLevels"),
					parseIntList(System.getProperty("plusone.svdishknn.docEnzs")),
					parseIntList(System.getProperty("plusone.svdishknn.termEnzs")),
					parseIntList(System.getProperty("plusone.svdishknn.dtNs")),
					parseIntList(System.getProperty("plusone.svdishknn.tdNs")),
					parseIntList(System.getProperty("plusone.svdishknn.numLVecs")),
					trainingSet,terms.size(),
					Integer.getInteger("plusone.svdishknn.walkLength"));
			KNNSVDcache = new KNNSimilarityCacheLocalSVDish(trainingSet,testingSet,localSVD);
		}
		for (int ck = 0; ck < closest_k_svdish.length; ck ++) {
			if (testIsEnabled("svdishknn")){
				knnSVD= new KNNLocalSVDish(closest_k_svdish[ck],trainingSet, paperIndexer,
						terms, KNNSVDcache);
				runClusteringMethod(knnSVD, ks,size,false);
			}
		}


		/*CommonNeighbors cn;
	  DTRandomWalkPredictor dtRWPredictor;
	  KNN knn;
	  KNNWithCitation knnc;
	  LSI lsi;



	  KNNGraphDistanceCache knnGraphDistanceCache;


	  if (testIsEnabled("knnc"))
	  knnGraphDistanceCache = 
	  new KNNGraphDistanceCache(trainingSet, testingSet, 
	  paperIndexer);



	  if (testIsEnabled("dtrw")) {
	  int rwLength =
	  Integer.getInteger("plusone.dtrw.walkLength", 4);
	  boolean stoch = Boolean.getBoolean("plusone.dtrw.stochastic");
	  int nSampleWalks = Integer.getInteger("plusone.dtrw.nSampleWalks");
	  System.out.println("Random walk length: " + rwLength);
	  if (stoch)
	  System.out.println("Stochastic random walk: " + nSampleWalks + " samples.");
	  boolean finalIdf = Boolean.getBoolean("plusone.dtrw.finalIdf");
	  boolean ndiw = Boolean.getBoolean("plusone.dtrw.normalizeDocsInWord");
	  Boolean nwid = Boolean.getBoolean("plusone.dtrw.normalizeWordsInDoc");
	  dtRWPredictor =
	  new DTRandomWalkPredictor(trainingSet, terms, rwLength, stoch, nSampleWalks, finalIdf, nwid, ndiw);
	  runClusteringMethod(testingSet, dtRWPredictor, 
	  outputDir, ks, size);
	  }









	  for (int ck = 0; ck < closest_k.length; ck ++) {
	  if (testIsEnabled("knn")) {
	  knn = new KNN(closest_k[ck], trainingSet, paperIndexer, 
	  terms, knnSimilarityCache);
	  runClusteringMethod(testingSet, knn, outputDir, ks, size);
	  }
	  if (testIsEnabled("knnc")) {
	  knnc = new KNNWithCitation(closest_k[ck], trainingSet,
	  paperIndexer, knnSimilarityCache,
	  knnGraphDistanceCache, terms);
	  runClusteringMethod(testingSet, knnc, outputDir, ks, size);
	  }*/

		/*	    if (testIsEnabled("cn")) {
		    cn = new CommonNeighbors(closest_k[ck], trainingSet, paperIndexer,
		    knnSimilarityCache, 
		    knnGraphDistanceCache, terms);
		    runClusteringMethod(testingSet, cn, outputDir, ks, size);
		    }



		    if (testIsEnabled("svdknn")) {
		    svdKnn = new SVDAndKNN(closest_k[ck], trainingSet);
		    runClusteringMethod(testingSet, knnc, outputDir, ks, size);
		    }


		    if (testIsEnabled("knnrw")) {
		    knnRWPredictor =
		    new KNNRandomWalkPredictor(closest_k[ck], trainingSet,
		    wordIndexer, paperIndexer,
		    1, 0.5, 1);
		    runClusteringMethod(trainingSet, testingSet,
		    knnRWPredictor, outputDir, ks, usedWord);
		    }*/

		/*	}




		 */


	}
	private void logResult(int ki,String expName, Map<String, Double> result){
		if (!allResults[ki].containsKey(expName))
			allResults[ki].put(expName,
				new Results(Arrays.asList(allResultFieldNames), expName));
		allResults[ki].get(expName).addResult(result);
	}

	public void runClusteringMethod(ClusteringTest test, int[] ks, int size, boolean bulk) {
		long t1 = System.currentTimeMillis();
		System.out.println("[" + test.testName + "] starting test" );
		double[][] allScores=new double[testingSet.size()][terms.size()];
		if (bulk){
			allScores = test.predict(testingSet);
		}

		double[][] results = new double[ks.length][4];

		for (int id=0;id<testingSet.size();id++){
			PredictionPaper testingPaper=testingSet.get(id);
			double[] itemScores;
			if (!bulk)
				itemScores= test.predict(testingPaper);
			else
				itemScores=allScores[id];


			int largestK=ks[ks.length-1];
			Queue<ItemAndScore> queue = new PriorityQueue<ItemAndScore>(largestK + 1);
			for (int i=0;i<itemScores.length;i++) {
				if (testingPaper.getTrainingTf(i) > 0.0)
					continue;

				if (queue.size() < largestK || 
						(double)itemScores[i] > queue.peek().score) {
					if (tagged) {
						if (wordIndexer.get(i).length() >= 4 &&
								wordIndexer.get(i).substring(0, 4).equals("tag ")) {
							if (queue.size() >= largestK)
								queue.poll();
							queue.add(new ItemAndScore(i, itemScores[i], true));
						}
					} else {
						if (queue.size() >= largestK)
							queue.poll();
						
						queue.add(new ItemAndScore(i, itemScores[i], true));
					}
				}
			}

			List<Integer> topPrdcts = new ArrayList<Integer>();
			while (!queue.isEmpty()) {
				topPrdcts.add(0,(Integer)queue.poll().item);
			}

			for (int ki = 0; ki < ks.length; ki ++) {
				int k = ks[ki];

//				MetadataLogger.TestMetadata meta = getMetadataLogger().getTestMetadata("k=" + k + test.testName);
//				test.addMetadata(meta);
//				List<Double> predictionScores = new ArrayList<Double>();
				
				Integer[] predict = topPrdcts.subList(0, Math.min(topPrdcts.size(), k)).toArray(new Integer[k]);

				double[] result = evaluate(testingPaper, predict, size, k);
				for (int j = 0; j < 4; ++j) results[ki][j] += result[j];
			}
		}
				//				predictionScores.add(result[0]);

//				meta.createListValueEntry("predictionScore", predictionScores.toArray());
//				meta.createSingleValueEntry("numPredict", k);
		for (int ki=0;ki<ks.length;ki++){
				//File out = new File(kDir, test.testName + ".out");
				int k=ks[ki];
				Map<String, Double> result = new HashMap<String, Double>();
				result.put("predictionRate", results[ki][0]/k/testingSet.size());
				result.put("tfScore", results[ki][1]/k/testingSet.size());
				result.put("tfidfScore", results[ki][2]/k/testingSet.size());
				this.logResult(ki, test.getName(), result);
		}
		System.out.println("[" + test.testName + "] took " +
				(System.currentTimeMillis() - t1) / 1000.0 
				+ " seconds.");
	}




	static double[] parseDoubleList(String s) {
		String[] tokens = s.split(",");
		double[] ret = new double[tokens.length];
		for (int i = 0; i < tokens.length; ++ i) {
			ret[i] = Double.valueOf(tokens[i]);
		}
		return ret;
	}

	static int[] parseIntList(String s) {
		String[] tokens = s.split(",");
		int[] ret = new int[tokens.length];
		for (int i = 0; i < tokens.length; ++ i) {
			ret[i] = Integer.valueOf(tokens[i]);
		}
		return ret;
	}

	static Boolean testIsEnabled(String testName) {
		return Boolean.getBoolean("plusone.enableTest." + testName);
	}

	/* FIXME: We probably should divide by k here, rather than the total
	 * number of predictions made; otherwise we reward methods that make
	 * less predictions.  -James */
	public double[] evaluate(PredictionPaper testingPaper,
			Integer[] prediction, int size, int k) {
		int predicted = 0, total = 0;
		double tfidfScore = 0.0, idfScore = 0.0, idf_top =  Math.log(size);

		Set<Integer> givenWords = testingPaper.getTrainingWords();
		Set<Integer> predictionWords = ((PaperAbstract)testingPaper).
				getTestingWords();
		for (int j = 0; j < prediction.length && j < k; j ++) {
			Integer word = prediction[j];
			if (predictionWords.contains(word)) {
				predicted ++;
				double logVal = Math.log(terms.get(word).idfRaw() + 1.0);

				tfidfScore += ((PaperAbstract)testingPaper).
						getTestingTf(word) * 
						(idf_top - logVal);
				idfScore += (idf_top - logVal);
			}
		}

		return new double[]{(double)predicted, idfScore, 
				tfidfScore, (double)prediction.length}; 
	}


	public static Random getRandomGenerator() { return randGen; }

	public static MetadataLogger getMetadataLogger() { return metadataLogger; }

	// -------------------------------------------------------------------------------------------------------------
	/**
	 * Splits all the documents into training and testing papers.
	 * This function must be called before we can do execute any
	 * clustering methods.
	 */
	private void splitByTrainPercent(double trainPercent, 
			List<PaperAbstract> documents) {
		//    Random randGen = Main.getRandomGenerator();
		trainingSet = new ArrayList<TrainingPaper>();
		testingSet = new ArrayList<PredictionPaper>();
		for (int i = 0; i < documents.size(); i ++) {
			if (tagged) {
				//Because documents.get(i).getIndex() == i
				if (tagMap.keySet().contains(i) && randGen.nextDouble() < trainPercent) {
					testingSet.add((PredictionPaper)documents.get(i));
					testIndices.put(documents.get(i), i);
				} else {
					trainingSet.add((TrainingPaper)documents.get(i));
					trainingIndices.put(documents.get(i), i);
				}
			} else {
				if (randGen.nextDouble() < trainPercent) {
					trainingSet.add((TrainingPaper)documents.get(i));
					trainingIndices.put(documents.get(i), i);
				} else {
					testingSet.add((PredictionPaper)documents.get(i));
					testIndices.put(documents.get(i), i);
				}
			}
		}
		System.out.println("trainingSet size: " + trainingSet.size());
		System.out.println("testingSet size: " + testingSet.size());
	}

	public void splitHeldoutWords(double testWordPercent) {
		Terms.Term[] terms = new Terms.Term[wordIndexer.size()];
		for (int i = 0; i < wordIndexer.size(); i++) {
			terms[i] = new Terms.Term(i);
		}

		for (TrainingPaper a : trainingSet){
			((PaperAbstract)a).generateTf(testWordPercent, terms, false);
		}

		for (PredictionPaper a : testingSet){
			((PaperAbstract)a).generateTf(testWordPercent, null, true);
		}

		/*
	  terms_sorted = new Term[terms.length];

	  for (int c = 0; c < terms.length; c ++) {
	  terms_sorted[c] = terms[c];
	  }
	  Arrays.sort(terms_sorted);
		 */
		this.terms = new Terms(terms);



	}

	/*
	 * data - args[0]
	 * train percent - args[1]
	 * test word percent - args[2] (currently ignored)
	 */
	public static void main(String[] args) {

		dataFile = System.getProperty("plusone.dataFile", "med.out");

		if (!new File(dataFile).exists()) {
			System.out.println("Data file does not exist.");
			System.exit(0);
		}

		long randSeed = 
				new Long(System.getProperty("plusone.randomSeed", "0"));
		generator = System.getProperty("plusone.generator");
		numTopics = Integer.parseInt(System.getProperty("plusone.lda.dimensions"));

		randGen = new Random(randSeed);
		metadataLogger = new MetadataLogger();

		Main main = new Main();

		FOLD =  new Integer(System.getProperty("plusone.crossValidation.FOLD","10"));
		//	double trainPercent = new Double
		//    (System.getProperty("plusone.trainPercent", "0.95"));
		String experimentPath = System.getProperty("plusone.outPath", 
				"experiment");

		load_data(dataFile);
		//main.splitByTrainPercent(trainPercent, dataset.getDocuments());
		System.out.println("data file " + dataFile);
		//System.out.println("train percent " + trainPercent);
		//System.out.println("test word percent " + testWordPercent);
		System.out.println("Number of Documents: "+ dataset.getDocuments().size());
		System.out.println("Wordindexer size: " + wordIndexer.size());

		for (PaperAbstract paper:dataset.getDocuments())
			paper.setGroup(randGen.nextInt(FOLD));

		main.trainingIndices = new HashMap<PaperAbstract, Integer>();
		main.testIndices = new HashMap<PaperAbstract, Integer>();
		main.runExperiments(experimentPath);
		/* These values can be set on the command line.  For example, to set
		 * testWordPercents to {0.4,0.5}, pass the command-line argument
		 * -Dplusone.testWordPercents=0.4,0.5 to java (before the class name)
		 */
		/*
	  double[] testWordPercents = 
	  parseDoubleList(System.getProperty("plusone.testWordPercents", 
	  "0.1,0.3,0.5,0.7,0.9"));
	  int[] ks = 
	  parseIntList(System.getProperty("plusone.kValues", 
	  "1,5,10,15,20"));

	  for (int twp = 0; twp < testWordPercents.length; twp++) {
	  double testWordPercent = testWordPercents[twp];	    
	  File twpDir = null;
	  try {
	  twpDir = new File(new File(experimentPath), 
	  testWordPercent + "");
	  twpDir.mkdir();
	  } catch(Exception e) {
	  e.printStackTrace();
	  }

	  main.splitHeldoutWords(testWordPercent);


	  System.out.println("processing testwordpercent: " + 
	  testWordPercent);

	  main.runClusteringMethods(twpDir, ks);

	  }*/

		if (Boolean.getBoolean("plusone.dumpMeta")) {
			PlusoneFileWriter writer = 
					new PlusoneFileWriter(new File(new File(experimentPath),
							"metadata"));
			writer.write("var v = ");
			writer.write(Main.getMetadataLogger().getJson());
			writer.close();
		}
	}


}
