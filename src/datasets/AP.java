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
		String filename = "data/ap.txt";
		System.out.println("Processing " + filename + "...");
		File input = new File(filename);
    	PrintWriter out = new PrintWriter( new BufferedWriter(
    			new FileWriter( "data/ap.json" ) ) );
    	//ArrayList<String> stopWords = makeStopWords();
    	
		JSONObject json = new JSONObject();
		
		JSONArray docs = new JSONArray();
		int id = 0;

		Scanner lines = new Scanner(input);
		boolean realText = false;
		while (lines.hasNextLine()) {
			String line = lines.nextLine();
			if (realText) {
				JSONObject user = new JSONObject();
				ArrayList<String> brokenWords = new ArrayList<String>();
				user.put("id", id++);
				for (String word : line.split(" ")) {
					brokenWords.add(word);
				}
				user.put("items", brokenWords);
				docs.put(user);
				realText = false;
			}
			if (line.equals("<TEXT>")) {
				realText = true;
			}
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
