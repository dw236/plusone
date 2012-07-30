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

public class StackOverflow {

	/**
	 * Makes a json from the Stack Overflow dataset
	 * 
	 * @param args unused
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		String filename = "data/stackoverflow.1000.data";
		System.out.println("Processing " + filename + "...");
		File input = new File(filename);
    	PrintWriter out = new PrintWriter( new BufferedWriter(
    			new FileWriter( filename + ".json" ) ) );
    	//ArrayList<String> stopWords = makeStopWords();
    	
		JSONObject inJson = new JSONObject(new Scanner(input).nextLine());
		JSONObject outJson = new JSONObject();
		JSONArray questions = inJson.getJSONArray("questions");
		JSONArray docs = new JSONArray();
		int id = 0;
		
		for (int i = 0; i < questions.length(); i++) {
			JSONObject inUser = questions.getJSONObject(i);
			JSONObject outUser = new JSONObject();
			String inDoc = inUser.getString("body");
			ArrayList<String> questionText = new ArrayList<String>(
					Arrays.asList(inDoc.split(" ")));
			outUser.put("id", id++);
			outUser.put("items", questionText);
			docs.put(outUser);
			JSONArray answers;
			try {
				answers = inUser.getJSONArray("answers");
			} catch (Exception e) {
				answers = null;
			}
			if (answers != null) {
				for (int j = 0; j < answers.length(); j++) {
					outUser = new JSONObject();
					JSONObject answerer = answers.getJSONObject(j);
					String answererDoc = answerer.getString("body");
					ArrayList<String> answerText = new ArrayList<String>(
							Arrays.asList(answererDoc.split(" ")));
					outUser.put("id", id++);
					outUser.put("items", answerText);
					docs.put(outUser);
				}
			}
		}
		
    	outJson.put( "users", docs );
    	out.println( outJson.toString() );
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
