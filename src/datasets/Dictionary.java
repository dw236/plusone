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

public class Dictionary {

	/**
	 * Makes a json from the Stack Overflow dataset
	 * 
	 * @param args if args[0] is true, tags are put in a separate component of the json
	 * (as opposed to not being put in at all)
	 * if args[1] is true, only questions get put in, no answers
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		String filename = "data/cedict_ts.u8";
		System.out.println("Processing " + filename + "...");
		File input = new File(filename);
    	PrintWriter out = new PrintWriter( new BufferedWriter(new FileWriter( "data/dictionary.json" ) ) );
    	Scanner in = new Scanner(input, "UTF-8");
   
    	ArrayList<String> stopWords = makeStopWords();
    	
		JSONObject Json = new JSONObject();
		int id = 0; 
		final int cutoff = 400;
		JSONArray docs = new JSONArray();
		
		HashMap<String, Integer> charCounts = getCharCounts(input);
		
		while (in.hasNextLine()) {
			boolean skip = false;
			String fullLine = in.nextLine();
			if (fullLine.charAt(0) == '#') {
				continue;
			}
			String removedPinyin = "";
			boolean pinyin = false;
			for (int i = 0; i < fullLine.length(); i++) {
				if (fullLine.charAt(i) == '[') {
					pinyin = true;
				}
				if (!pinyin) {
					removedPinyin += fullLine.charAt(i);
				}
				if (fullLine.charAt(i) == ']') {
					pinyin = false;
				}
			}
			String[] splitLine = removedPinyin.split("/");
			String chinese = splitLine[0];
			String english = "";
			for (int i = 1; i < splitLine.length; i++) {
				english += splitLine[i] + " ";
			}
			String simplified = chinese.split(" ")[1];
			ArrayList<String> word = new ArrayList<String>();
			ArrayList<String> definition = new ArrayList<String>();
			for (int i = 0; i < simplified.length(); i++) {
				String character = simplified.charAt(i) + "";
				if (charCounts.get(character) < cutoff) {
					skip = true;
				}
				word.add(character);
			}
			
			for (String s : english.split(" ")) {
				String engWord = s.replaceAll("[^A-Za-z]", "").trim().toLowerCase();
				if(!stopWords.contains(engWord) && !engWord.equals("")) {
					definition.add(engWord);
				}
			}
			if (definition.size() == 0) {
				skip = true;
			}
			if (!skip) {
				JSONObject user = new JSONObject();
				user.put("id", id++);
				user.put("items", definition);
				user.put("tags", word);
				docs.put(user);
			}
			
		}
		
		
    	Json.put( "users", docs );
    	out.println( Json.toString() );
    	out.close();
    	System.out.println("Done!");
	}
	
	private static HashMap<String, Integer> getCharCounts(File input) throws Throwable {
		HashMap<String, Integer> charCounts = new HashMap<String, Integer>();
    	Scanner in = new Scanner(input, "UTF-8");
    	
		while (in.hasNextLine()) {
			String fullLine = in.nextLine();
			if (fullLine.charAt(0) == '#') {
				continue;
			}
			String chinese = fullLine.split("/")[0];
			String english = fullLine.split("/")[1];
			String simplified = chinese.split(" ")[1];
			for (int i = 0; i < simplified.length(); i++) {
				String thisChar = simplified.charAt(i) + "";
				if (charCounts.get(thisChar) == null) {
					charCounts.put(thisChar, 1);
				} else {
					charCounts.put(thisChar, charCounts.get(thisChar) + 1);
				}
			}
		}

		
		return charCounts;
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
