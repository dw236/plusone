package plusone.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.json.*;


/**
 * Replace DatasetJSON with this class to use real data in new format (one doc on each line)
 */
public class DatasetJSONNew {

    /* Member fields. */
	public HashMap<Integer,Integer>[] users;
	
    List<PaperAbstract> documents = new ArrayList<PaperAbstract>();
    public List<PaperAbstract> getDocuments() { return documents; }

    Indexer<String> wordIndexer = new Indexer<String>();
    public Indexer<String> getWordIndexer() { return wordIndexer; }

    private Indexer<PaperAbstract> paperIndexer = new Indexer<PaperAbstract>();
    public Indexer<PaperAbstract> getPaperIndexer() { return paperIndexer; }
    
    //Map from the paper's index to the wordIndexer's indices for its tags 
    private HashMap<Integer, ArrayList<Integer>> tagMap =
    		new HashMap<Integer, ArrayList<Integer>>();
    public HashMap<Integer, ArrayList<Integer>> getTagMap() { return tagMap; }

    /** Reads in the JSON file and fills in documents, wordIndexer, and
     * paperIndexer appropriately. Will automatically change its behavior when
     * presented with a regression task.
     * 
     * @param filename path to the JSON file (should be passed from loadDatasetFromPath)
     */
    void loadInPlaceFromPath(String filename) {	
		try {
			BufferedReader in = new BufferedReader( new FileReader( filename ) );
//			JSONObject json = new JSONObject( in.readLine() ); //change
//			JSONArray users = json.getJSONArray( "users" ); //change
						
			if (isIndexed(filename)) {
				initializeIndexer(filename);
			}
			
			int index = 0;
			JSONObject user;
			JSONArray items = null, scores = null, tags = null;
			HashMap<Integer, Integer> tf = null;
//			for( int i = 0; i < users.length(); i++ ) { //change
//				user = users.getJSONObject( i ); //change
			user = new JSONObject( in.readLine() );
			while (user != null) {
				tf = new HashMap<Integer, Integer>();
				items = user.getJSONArray( "items" );
				try	{
					//If successful, this is a regression file
					scores = user.getJSONArray( "scores" );
				} catch (JSONException e) {
					scores = null;
				}
				try	{
					tags = user.getJSONArray( "tags" );
				} catch (JSONException e) {
					tags = null;
				}

				for( int j = 0; j < items.length(); j++ ) {
					String jthItem = items.getString( j );
					if (scores == null) {
						if ( !tf.keySet().contains(this.wordIndexer.fastIndexOf(jthItem))) {
							tf.put( this.wordIndexer.fastAddAndGetIndex(jthItem), 1 );
						} else {
							tf.put( this.wordIndexer.fastIndexOf(jthItem), 
									tf.get( this.wordIndexer.fastIndexOf(jthItem) ) + 1 );
						}
					} else {
						tf.put( this.wordIndexer.fastAddAndGetIndex(jthItem), scores.getInt( j ) );
					}
				}
				PaperAbstract p = new PaperAbstract(index++, null, null, tf);
				if (tags != null) {
					ArrayList<Integer> tagsList = new ArrayList<Integer>();
					//Each tag gets put into the tf as a separate entity, "tag " + tags.getString(j)
					//Ideally there will be no spaces in non-tag words, so no "fake tags" could be created
					for (int j = 0; j < tags.length(); j++) {
						String newTag = "tag " + tags.getString(j);
						if ( !tf.keySet().contains(this.wordIndexer.fastIndexOf(newTag))) {
							tf.put( this.wordIndexer.fastAddAndGetIndex(newTag), 1 );
						} else {
							tf.put( this.wordIndexer.fastIndexOf(newTag),
									tf.get( this.wordIndexer.fastIndexOf(newTag) ) + 1 );
						}
						tagsList.add(wordIndexer.fastIndexOf(newTag));
					}
					tagMap.put(index-1, tagsList);
				}
				documents.add(p);
				paperIndexer.add(p);
				try {
					user = new JSONObject( in.readLine() );
				} catch (NullPointerException e) {
					user = null;
				}
			}
		} catch(Exception e) {
		    e.printStackTrace();
		}
    }

    /**
     * This method is to be called in order to construct a datasetJSON
     * 
     * @param filename The path to the JSON file being loaded
     * @return a DatasetJSON with its document, wordIndexer, and paperIndexer fields instantiated with the information contained in the JSON
     */
    public static DatasetJSON loadDatasetFromPath(String filename) {
        DatasetJSON dataset = new DatasetJSON();
        dataset.loadInPlaceFromPath(filename);
        return dataset;
    }
    
    
    /**
     * Returns true if the input is already indexed
     * 
     * @param filename path to the JSON file (should be passed from loadDatasetFromPath)
     * @return
     */
    private boolean isIndexed(String filename) {
    	JSONObject user; JSONArray items;
    	try {
			BufferedReader in = new BufferedReader( new FileReader( filename ) );
//			JSONArray users = new JSONObject( in.readLine() ).getJSONArray( "users" ); //change
//			for (int i = 0; i < users.length(); i++) { //change
//				user = users.getJSONObject( i ); //change
			user = new JSONObject( in.readLine() );
			while (user != null) {
				items = user.getJSONArray( "items" );
				for (int j = 0; j < items.length(); j++) {
					try {
						int unused = new Integer(items.getString(j));
					} catch (NumberFormatException e) {
						return false;
					}
				}
				try {
					user = new JSONObject( in.readLine() );
				} catch (NullPointerException e) {
					user = null;
				}			}
			return true;
    	} catch (Exception e) {
    		e.printStackTrace();
    		return false;
    	}
    }
    
    /**
     * Returns true if the input is tagged. Tagged input will have a JSONArray
     * entitled "tags" for some users
     * 
     * @param filename path to the JSON file
     * @return
     */
    public boolean isTagged(String filename) {
    	JSONObject user; JSONArray items, tags;
    	try {
			BufferedReader in = new BufferedReader( new FileReader( filename ) );
//			JSONArray users = new JSONObject( in.readLine() ).getJSONArray( "users" ); //change	  	
//			for (int i = 0; i < users.length(); i++) { //change
//				user = users.getJSONObject( i ); //change
			user = new JSONObject( in.readLine() );
			while (user != null) {
				items = user.getJSONArray( "items" );
				try {
					tags = user.getJSONArray("tags");
					return true;
				} catch (Exception e) {
					//If we fall into this statement every time, return false
				}
				try {
					user = new JSONObject( in.readLine() );
				} catch (NullPointerException e) {
					user = null;
				}	
            }
    	} catch (Exception e) {
    		e.printStackTrace();
    		return false;
    	}
    	return false;
    }
    
    /**
     * If we have pre-indexed files, initialize the indexer with those indices
     * @param filename path to the JSON file (should be passed from loadDatasetFromPath)
     */
    private void initializeIndexer(String filename) {
    	int maxIndex = -1;
    	JSONObject user;
    	JSONArray users, items; //change
    	try {
			BufferedReader in = new BufferedReader( new FileReader( filename ) );
//			JSONObject json = new JSONObject( in.readLine() ); //change
//			users = json.getJSONArray( "users" ); //change
//			for (int i = 0; i < users.length(); i++) { //change
//				user = users.getJSONObject( i ); //change
			user = new JSONObject( in.readLine() );
			while (user != null) {
				 items = user.getJSONArray( "items" );
				for (int j = 0; j < items.length(); j++) {
					if (Integer.parseInt(items.getString(j)) > maxIndex) {
						maxIndex = Integer.parseInt(items.getString(j));
					}
				}
				try {
					user = new JSONObject( in.readLine() );
				} catch (NullPointerException e) {
					user = null;
				}			
            }
    	} catch (Exception e) {
    		e.printStackTrace();
    		System.out.println("Couldn't initialize indexer for pre-indexed files");
    	}
    	for (int i = 0; i < maxIndex; i++) {
    		this.wordIndexer.fastAddAndGetIndex(i  +"");
    	}
    }
}
