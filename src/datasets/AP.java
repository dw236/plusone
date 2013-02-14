package datasets;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class AP {

	/**
	 * Makes a json from the AP dataset at http://www.cs.princeton.edu/~blei/lda-c/
	 * 
	 * @param args unused
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		//final double tfIdfTolerance = 0.005;
		String filename = "data/ap.txt";
		System.out.println("Processing " + filename + "...");
		File input = new File(filename);
    	PrintWriter out = new PrintWriter( new BufferedWriter(
    			new FileWriter( "data/ap.json" ) ) );
    	ArrayList<String> stopWords = makeStopWords();
    	
		HashMap<String, Integer> idf = makeIdf(input);
		ArrayList<HashMap<String, Double>> tfs = makeTfs(input);
    	
		JSONObject json = new JSONObject();
		
		JSONArray docs = new JSONArray();
		int id = 0;
				
		HashSet<String> keptWords = new HashSet<String>();
		
		for (int i = 0; i < idf.keySet().size(); i++) {
			JSONObject user = new JSONObject();
			HashMap<String, Double> docTf = tfs.get(id);
			ArrayList<String> savedWordsInDoc = new ArrayList<String>();
			for (String s : docTf.keySet()) {
				//if (docTf.get(s)*Math.log(numDocs/idf.get(s)) > tfIdfTolerance) {
				if (idf.get(s) < idf.keySet().size()*0.8 && idf.get(s) > 4
						&& !stopWords.contains(s)) {
					savedWordsInDoc.add(s);
					keptWords.add(s);
				}
			}
			if (savedWordsInDoc.size() == 0) {
				continue;
			}
			user.put("id", id++);
			user.put("items", savedWordsInDoc);
			docs.put(user);
		}
		System.out.println("Kept " + keptWords.size() + " words out of "
							+ idf.keySet().size());
    	json.put( "users", docs );
    	out.println( json.toString() );
    	out.close();
    	System.out.println("Done!");
	}
	
	private static HashMap<String, Integer> makeIdf(File input) throws Throwable {
		Scanner lines = new Scanner(input);
		boolean realText = false;
		HashMap<String, Integer> allWords = new HashMap<String, Integer>();
		while (lines.hasNextLine()) {
			String line = lines.nextLine();
			ArrayList<String> seenWordsThisDoc = new ArrayList<String>();
			if (realText) {
				for (String word : line.split(" ")) {
					word = word.replaceAll("[^A-Za-z]", "").trim().toLowerCase();
					if (word.equals("")) {
						continue;
					}
					if (!allWords.keySet().contains(word)) {
						allWords.put( word , 1 );
						seenWordsThisDoc.add(word);
					} else if (!seenWordsThisDoc.contains(word)){
						allWords.put( word, allWords.get( word) + 1 );
						seenWordsThisDoc.add(word);
					}
				}
				realText = false;
			}
			if (line.equals("<TEXT>")) {
				realText = true;
			}
		}		
		return allWords;
	}
	
	private static ArrayList<HashMap<String, Double>> makeTfs(File input) throws Throwable {
		Scanner lines = new Scanner(input);
		boolean realText = false;
		ArrayList<HashMap<String, Double>> allTfs = new ArrayList<HashMap<String, Double>>();
		int docNumber = 0;
		while (lines.hasNextLine()) {
			String line = lines.nextLine();
			if (realText) {
				double totalWordsThisDoc = line.split(" ").length;
				for (String word : line.split(" ")) {
					word = word.replaceAll("[^A-Za-z]", "").trim().toLowerCase();
					if (word.equals("")) {
						continue;
					}
					if (!allTfs.get(docNumber).keySet().contains(word)) {
						allTfs.get(docNumber).put( word , 1/totalWordsThisDoc );
					} else {
						allTfs.get(docNumber).put( word, allTfs.get(docNumber).get( word)
													+ 1/totalWordsThisDoc );
					}
				}
				realText = false;
				docNumber++;
			}
			if (line.equals("<TEXT>")) {
				realText = true;
				allTfs.add(new HashMap<String, Double>());
			}
		}		
		return allTfs;
	}
	
	private static ArrayList<String> makeStopWords() throws Throwable {
		ArrayList<String> result = new ArrayList<String>();
		File input = new File("data/english.stop.txt");
		Scanner in = new Scanner(input);
		while (in.hasNext()) {
			result.add(in.next());
		}
		result.add(".");
		return result;
	}

}
