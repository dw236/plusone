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

public class BagOfWords {

	/**
	 * Makes a json from the bag of words datasets at
	 * http://archive.ics.uci.edu/ml/datasets/Bag+of+Words
	 * 
	 * @param args unused
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		String filename = "data/docword.nytimes.txt";
		System.out.println("Processing " + filename + "...");
		File input = new File(filename);
    	PrintWriter out = new PrintWriter( new BufferedWriter(
    			new FileWriter( filename + ".json" ) ) );
    	//ArrayList<String> stopWords = makeStopWords();
    	
		JSONObject json = new JSONObject();
		
		JSONArray docs = new JSONArray();

		Scanner lines = new Scanner(input);
		int numDocs = Integer.parseInt(lines.nextLine());
		lines.nextLine();
		lines.nextLine(); //Ignore these lines of input

		int currentDocId = 1;
		JSONObject doc = new JSONObject();
		ArrayList<String> document = new ArrayList<String>();
		
		while (lines.hasNextLine()) {
			String[] parsedLine = lines.nextLine().split(" ");
			int docId = Integer.parseInt(parsedLine[0]);
			int wordId = Integer.parseInt(parsedLine[1]);
			int count = Integer.parseInt(parsedLine[2]);
			if (docId != currentDocId) {
				doc.put("id", currentDocId);
				doc.put("items", document);
				docs.put(doc);
				document = new ArrayList<String>();
		    	doc = new JSONObject();
				currentDocId += 1;
				if (currentDocId % 10000 == 0) {
					System.out.println("Reached doc number " + currentDocId + " of " + numDocs);
					System.out.println((currentDocId*100)/numDocs + "% done");
				}
			}
			for (int i = 0; i < count; i++) {
				document.add(wordId + "");
			}
		}
		doc.put("id", currentDocId);
		doc.put("items", document);
		docs.put(doc);
		
    	json.put( "users", docs );
    	out.println( json.toString() );
    	out.close();
    	System.out.println("Done!");
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
