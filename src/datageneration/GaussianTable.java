package datageneration;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

public class GaussianTable {

	/**
	 * Reads in Gaussian files and makes a csv
	 *  
	 * @param args args[0] is the name of the directory with gaussian files
	 * ("gaussian" by default)
	 * args[1] is "true" if and only if the data was estimated with a different number
	 * of topics than it was generated with
	 * @throws JSONException 
	 */
	public static void main(String[] args) throws Throwable {
		String dir;
		if (args.length > 0) {
			dir = args[0];
		} else {
			dir = "gaussian";
		}
		
		boolean diffTopics = false;
		if (args.length > 1 && Boolean.parseBoolean(args[1])) {
			diffTopics = true;
		}
		
		PrintWriter out = new PrintWriter( new BufferedWriter(
    			new FileWriter( "gaussian.csv" ) ) );
		System.out.println("Processing files in " + dir + "...");
		File directory = new File(dir);
		
		out.print(",");
		ArrayList<GaussianFile> allFiles = new ArrayList<GaussianFile>();
		HashSet<Integer> kSet = new HashSet<Integer>();
		HashSet<Integer> diffSet = new HashSet<Integer>();
		HashSet<Double> aSet = new HashSet<Double>();
		HashSet<Double> bSet = new HashSet<Double>();

    	for (File gaussFile : directory.listFiles()) {
    		GaussianFile thisFile = null;
    		Scanner line = new Scanner(gaussFile);
    		String[] parsedLine = line.nextLine().split(" ");
    		double cosRatio = Double.parseDouble(parsedLine[0]);
    		double eucRatio = Double.parseDouble(parsedLine[1]);
    		double a = Double.parseDouble(parsedLine[2]);
    		double b = Double.parseDouble(parsedLine[3]);
    		int n = Integer.parseInt(parsedLine[4]);
    		int m = Integer.parseInt(parsedLine[5]);
    		int k = Integer.parseInt(parsedLine[6]);
    		int diff = 0;
    		if (diffTopics) {
    			diff = Integer.parseInt(parsedLine[7]);
    			thisFile = new GaussianFile(cosRatio, eucRatio, a, b, n, m, k, diff);
    		} else {
    			thisFile = new GaussianFile(cosRatio, eucRatio, a, b, n, m, k);
    		}
    		allFiles.add(thisFile);
    		kSet.add(k);
    		diffSet.add(diff);
    		aSet.add(a);
    		bSet.add(b);
    	}
    	//All sorted
    	ArrayList<Integer> kList = new ArrayList<Integer>(new TreeSet<Integer>(kSet));
    	ArrayList<Integer> diffList = new ArrayList<Integer>(new TreeSet<Integer>(diffSet));
    	ArrayList<Double> aList = new ArrayList<Double>(new TreeSet<Double>(aSet));
    	ArrayList<Double> bList = new ArrayList<Double>(new TreeSet<Double>(bSet));
    	System.out.println(diffList);

    	for (Integer k : kList) {
    		out.print(k+",");
    	}
    	out.println();
    	
    	for (Double a : aList) {
    		for (Double b : bList) {
    			out.print("(" + a + " " + b + "),");
    			for (Integer k : kList) {
    				if (diffTopics) {
	    				ArrayList<GaussianFile> correctABKs = new ArrayList<GaussianFile>();
	    				for (GaussianFile g : allFiles) {
	    					if (g.a == a && g.b == b && g.k == k) {
	    						correctABKs.add(g);
	    					}
	    				}
	    				for (GaussianFile g : correctABKs) {
	    					out.print(g.cosRatio + " " + g.eucRatio + " " + "("
	    							+ g.estK + ") | ");
	    				}
	    				out.print(",");
    				} else {
	    				GaussianFile correctABKTriple = null;
	    				for (GaussianFile g : allFiles) {
	    					if (g.a == a && g.b == b && g.k == k) {
	    						correctABKTriple = g;
	    					}
	    				}
	    				out.print(correctABKTriple.cosRatio + " "
	    					+ correctABKTriple.eucRatio + ",");
	    			}
    			}
    			out.println();
    		}
    	}
    	out.close();
    	System.out.println("Done!");
	}
	
	static class GaussianFile {
		public double cosRatio, eucRatio, a, b;
		public int n, m, k, estK;
		
		public GaussianFile() {
		}

		public GaussianFile(double cosRatio, double eucRatio,
				double a, double b, int n, int m, int k) {
			this.cosRatio = cosRatio;
			this.eucRatio = eucRatio;
			this.a = a; this.b = b; this.n = n; this.m = m; this.k = k;
		}
		
		public GaussianFile(double cosRatio, double eucRatio,
				double a, double b, int n, int m, int k, int diff) {
			this.cosRatio = cosRatio;
			this.eucRatio = eucRatio;
			this.a = a; this.b = b; this.n = n; this.m = m; this.k = k;
			this.estK = k + diff;
		}
	}
}
