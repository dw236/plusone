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

public class JACM {

	/**
	 * Makes a json from the JACM dataset at
	 * http://www.cs.princeton.edu/~blei/downloads/jacm.tgz
	 * 
	 * @param args unused
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		String filename = "data/jacm/jacm-info.dat";
		System.out.println("Processing " + filename + "...");
		File input = new File(filename);
    	PrintWriter out = new PrintWriter( new BufferedWriter(
    			new FileWriter( "data/jacm.json" ) ) );
    	//ArrayList<String> stopWords = makeStopWords();
    	
		JSONObject json = new JSONObject();
		
		JSONArray docs = new JSONArray();
		int id = 0;

		Scanner lines = new Scanner(input);
		boolean realText = false;
		while (lines.hasNextLine()) {
			String document = lines.nextLine();
			ArrayList<Integer> quoteLocs= new ArrayList<Integer>();
			for (int i = 0; i < document.length(); i++) {
				if (document.charAt(i) == '"') {
					quoteLocs.add(i);
				}
			}
			if (quoteLocs.size() > 4) {
				System.out.println("Too many quotes!");
			}
			String fixedDocument = document.substring(quoteLocs.get(2)+1, quoteLocs.get(3));
			System.out.println(fixedDocument);
			JSONObject user = new JSONObject();
			ArrayList<String> brokenWords = new ArrayList<String>();
			user.put("id", id++);
			for (String word : fixedDocument.split(" ")) {
				brokenWords.add(word);
			}
			user.put("items", brokenWords);
			docs.put(user);
		}		
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
