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

public class TwentyNewsgroups {

	/**
	 * Makes a json from the 20 Newsgroups found at http://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.html
	 * 
	 * @param args unused
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		String dirName = "data/20_newsgroups";
		System.out.println("Processing files in " + dirName + "...");
		File inputDir = new File(dirName);
    	PrintWriter out = new PrintWriter( new BufferedWriter(new FileWriter( "data/20Newsgroups.json" ) ) );
    	
    	//ArrayList<String> stopWords = makeStopWords();
    	int id = 0;
    	for (File firstSubDir : inputDir.listFiles()) {
    		if (firstSubDir.isDirectory()) {
	    		for (File inFile : firstSubDir.listFiles()) {
	    			Scanner input = new Scanner(inFile);
					JSONObject user = new JSONObject();
					ArrayList<String> text = new ArrayList<String>();
					//JSONArray tags;
					
	    			while(input.hasNextLine()) {
	    				String[] parsedLine = input.nextLine().split("\"*\\s+\"*");
	    				for (String s : parsedLine) {
	    					if (!s.equals("")) {
	    						text.add(s);
	    					}
	    				}
	    			}
										
					user.put("id", id++);
					user.put("items", text);
					out.println(user.toString());
	    		}
			}
    	}
		
    	//outJson.put( "users", users );
    	//out.println( outJson.toString() );
    	out.close();
    	System.out.println("Done!");
	}
}
