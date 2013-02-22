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

public class Classic4 {

	/**
	 * Makes a json from the datasets at
	 * http://www.dataminingresearch.com/index.php/2010/09/classic3-classic4-datasets/
	 * Put the classic folder obtained at that website in plusone/data/
	 * args[0] corresponds to the dataset being processed (cacm, cisi, cran, or med).
	 * if args[1] is true, stop words will be excluded, if args[1] is false, every word
	 * is included.
	 * 
	 * @param args the dataset being processed and a stop word flag
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		File input;
		Dataset ds = null;
		//Set the dataset with args[0]
		for (Dataset d : Dataset.values()) {
			if (args[0].equals(d.toString())) {
				ds = d;
			}
		}
		if (ds == null) {
			System.err.println("Usage: java Classic4 {cacm|cisi|cran|med} {true|false}");
			System.exit(1);
		}
		System.out.print("Making json from " + ds.toString() + "...");
    	PrintWriter out = new PrintWriter( new BufferedWriter( new FileWriter("data/" 
    		+ ds.toString() + ".json" ) ) );
    	ArrayList<String> stopWords = new ArrayList<String>();
    	if (Boolean.parseBoolean(args[1])) {
    		stopWords = makeStopWords();
    	}
    	int numFiles = 0;
    	switch (ds) {
    		case cacm:
    			numFiles = 3204;
    			break;
    		case cisi:
    			numFiles = 1460;
    			break;
    		case cran:
    			numFiles = 1400;
    			break;
    		case med:
    			numFiles = 1033;
    			break;
    	}
    	
		HashMap<Integer, ArrayList<String>> hm = new HashMap<Integer,ArrayList<String>>();
		for (int i = 1; i <= numFiles; i++) {
			boolean beginning = true;
			ArrayList<String> thisDoc = new ArrayList<String>();
			if (i < 10)
				input = new File("data/classic/" + ds.toString() + ".00000" + i);
			else if (i < 100)
				input = new File("data/classic/" + ds.toString() + ".0000" + i);
			else if (i < 1000)
				input = new File("data/classic/" + ds.toString() + ".000" + i);
			else
				input = new File("data/classic/" + ds.toString() + ".00" + i);
			Scanner in = null;
			try {
				in = new Scanner(input);
				while (in.hasNext()) {
					String word = in.next();
					if (ds == Dataset.cran) {
						while(beginning) {
							if (word.equals(".")) {
								beginning = false;
							}
							word = in.next();
						}
					}
					if (!stopWords.contains(word.toLowerCase())) {
				    	Stemmer stemmer = new Stemmer();
				    	stemmer.add(word.toCharArray(), word.length());
				    	stemmer.stem();
				    	thisDoc.add(stemmer.toString().toLowerCase());
					}
				}
				hm.put(i, thisDoc);
			} catch (FileNotFoundException e) {
				System.out.println("Could not find: " + i);
			}
		}

	    for( Integer user : hm.keySet() ) {
	    	JSONObject doc = new JSONObject();
	    	doc.put( "id", user );

	    	JSONArray terms = new JSONArray();
	    	ArrayList<String> thisUser = hm.get(user);

	    	for( String word : thisUser ) {
	    		terms.put(word);
	    	}

	    	doc.put( "items", terms );
	    	//doc.put( "scores", scores );
	    	out.println( doc );
	    }
    	out.close();
    	System.out.println("done!");
	}
	
	public enum Dataset {cacm, cisi, cran, med}
	
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
