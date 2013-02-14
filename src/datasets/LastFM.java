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

public class LastFM {

	/**
	 * Makes a json from the last.fm dataset
	 * 
	 * @param args args[0] is the tag tolerance and ranges from 0-100, with 0 meaning
	 * all tags are inserted and 100 meaning only the most popular tags are inserted.
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		String dirName = "data/lastfmsubset";
		System.out.println("Processing files in " + dirName + "...");
		File inputDir = new File(dirName);
    	PrintWriter out = new PrintWriter( new BufferedWriter(new FileWriter( "data/lastfmsubset.json" ) ) );
    	
		JSONObject outJson = new JSONObject();
		JSONArray users = new JSONArray();

    	//ArrayList<String> stopWords = makeStopWords();
    	int id = 0;
    	
    	for (File firstSubDir : inputDir.listFiles()) {
    		if (firstSubDir.isDirectory()) {
	    		for (File secondSubDir : firstSubDir.listFiles()) {
	        		if (secondSubDir.isDirectory()) {
		    			for (File thirdSubDir : secondSubDir.listFiles()) {
		    				if (thirdSubDir.isDirectory()) {
			    				for (File inFile : thirdSubDir.listFiles()) {
			    					JSONObject inJson = new JSONObject(new Scanner(inFile).nextLine());
			    					JSONObject user = new JSONObject();
			    					JSONArray tags;
			    					
			    					ArrayList<String> text = new ArrayList<String>();
			    					text.add(inJson.getString("artist"));
			    					//text.add(inJson.getString("title"));
			    					try {
			    						tags = inJson.getJSONArray("tags");
			    					} catch (Exception e) {
			    						tags = null;
			    					}
			    					
			    					ArrayList<String> tagText = new ArrayList<String>();
			    					if (tags != null && tags.length() > 0) {
			    						for (int i = 0; i < tags.length(); i++) {
			    							JSONArray oneTag = tags.getJSONArray(i);
			    							if (oneTag.getInt(1) >= Integer.parseInt(args[0])) {
			    								tagText.add(oneTag.getString(0));
			    							}
			    						}
			    						user.put("tags", tagText);
			    					}
			    					user.put("id", id++);
			    					user.put("items", text);
			    					users.put(user);
			    				}
		    				}
		    			}
	        		}
	    		}
    		}
    	}
		
    	outJson.put( "users", users );
    	out.println( outJson.toString() );
    	out.close();
    	System.out.println("Done!");
	}
}
