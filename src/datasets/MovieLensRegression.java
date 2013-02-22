package datasets;

import java.io.*;
import java.util.*;

import org.json.*;

public class MovieLensRegression {
	public static void main( String[] args ) throws Throwable {
	    Scanner in = new Scanner( new File( "data/ratings.dat" ) );
    	PrintWriter out = new PrintWriter( new BufferedWriter( new FileWriter( "data/movielens-reg5.json" ) ) );
	    String s;
	    HashMap<Integer,HashMap<Integer,Double>> hm = new HashMap<Integer,HashMap<Integer,Double>>();
	    HashSet<Integer> movies = new HashSet<Integer>();

	    while( in.hasNextLine() ) {
	    	String[] arr = in.nextLine().split( "::" );
	    	int user = Integer.parseInt( arr[0] );
	    	int movie = Integer.parseInt( arr[1] );
	    	int rating = Integer.parseInt( arr[2] );
	    	int time = Integer.parseInt( arr[3] );

	    	if( hm.containsKey( user ) ) {
	    		hm.get( user ).put( movie, (double)rating );
	    	} else {
	    		hm.put( user, new HashMap<Integer,Double>() );
	    		hm.get( user ).put( movie, (double)rating );
	    	}

	    	movies.add( movie );
	    }

	    JSONObject json = new JSONObject();
    	
    	//JSONArray docs = new JSONArray();

	    for( Integer user : hm.keySet() ) {
	    	JSONObject doc = new JSONObject();
	    	doc.put( "id", user );

	    	JSONArray terms = new JSONArray();
	    	JSONArray scores = new JSONArray();
	    	HashMap<Integer,Double> hm2 = hm.get( user );

	    	for( Integer movie : hm2.keySet() ) {
	    		terms.put( String.valueOf( movie ) );
	    		scores.put(hm2.get(movie));
	    	}

	    	doc.put( "items", terms );
	    	//doc.put( "scores", scores );
	    	out.println( doc );
	    }

    	//json.put( "users", docs );
    	//out.println( json.toString() );
    	out.close();
    	//System.out.println( json.toString( 4 ) );
    }
}