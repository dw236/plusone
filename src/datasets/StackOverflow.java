package datasets;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;
import java.io.*;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class StackOverflow {

	/**
	 * Makes a json from the Stack Overflow dataset
	 * 
	 * @param args if args[0] is true, tags are put in a separate component of the json
	 * (as opposed to not being put in at all)
	 * if args[1] is true, only questions get put in, no answers
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		String filename = "data/stackoverflow.1000.data";
		boolean onlyQuestions = Boolean.parseBoolean(args[1]);
		System.out.println("Processing " + filename + "...");
    	PrintWriter out;
    	if (onlyQuestions) {
    		out = new PrintWriter( new BufferedWriter(new FileWriter( filename + ".questions.json" ) ) );
    	} else {
    		out = new PrintWriter( new BufferedWriter(new FileWriter( filename + ".json" ) ) );
    	}
   
    	//ArrayList<String> stopWords = makeStopWords();
    	BufferedReader in = new BufferedReader(new FileReader(filename));
		JSONObject inJson = new JSONObject(in.readLine());
		JSONArray questions = inJson.getJSONArray("questions");
		int id = 0; boolean skip = false;
		
		for (int i = 0; i < questions.length(); i++) {
			JSONObject inUser = questions.getJSONObject(i);
			JSONObject outUser = new JSONObject();
			String inDoc = inUser.getString("body");
			ArrayList<String> questionText = new ArrayList<String>(
					Arrays.asList(inDoc.split(" ")));
			if (questionText.size() == 1 && questionText.get(0).equals("")) {
				skip = true;
			}
			ArrayList<String> tagText = new ArrayList<String>();
			JSONArray answers, tags;
			if (Boolean.parseBoolean(args[0]) == true) {
				try {
					tags = inUser.getJSONArray("tags");
				} catch (Exception e) {
					tags = null;
					System.out.println("Question " + i + " has no tags");
				}
				if (tags != null) {
					for (int j = 0; j < tags.length(); j++) {
						tagText.add(tags.getString(j));
					}
				}
			}
			if (!skip) {
				outUser.put("id", id++);
				outUser.put("items", questionText);
				outUser.put("tags", tagText);
				out.println(outUser);
			} else {
				skip = false;
			}
			try {
				answers = inUser.getJSONArray("answers");
			} catch (Exception e) {
				answers = null;
				System.out.println("Question " + i + " has no answers");
			}
			if (answers != null && !onlyQuestions) {
				for (int j = 0; j < answers.length(); j++) {
					outUser = new JSONObject();
					JSONObject answerer = answers.getJSONObject(j);
					String answererDoc = answerer.getString("body");
					ArrayList<String> answerText = new ArrayList<String>(
							Arrays.asList(answererDoc.split(" ")));
					if (answerText.size() == 1 && answerText.get(0).equals("")) {
						continue;
					}
					outUser.put("id", id++);
					outUser.put("items", answerText);
					out.println(outUser);
				}
			}
		}
		
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
