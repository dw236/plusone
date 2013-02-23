package plusone.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Utils {

    public static void writeToFile(String filename, String content) {
	try {
	    FileWriter fstream = new FileWriter(filename);
	    BufferedWriter out = new BufferedWriter(fstream);
	    out.write(content);
	    out.close();
	} catch (Exception e) {
	    System.err.println("Error: " + e.getMessage());
	}
    }
    
    /**
     * Executes a command line input (eg ls -a) with the option to hide the
     * output (to System.out) that command usually would give. If the command
     * fails, will display all error output from the command line.
     * 
     * @param command		the command to be run, exactly as if typed
     * 						into the command line (a String)
     * @param streamOutput	a flag to determine whether or not to display
     * 						any output the command might give
     */
    public static boolean runCommand(String command, boolean streamOutput) {
    
	boolean success = true;
	System.out.println("Running command: " + command);
	try {
	    Process p = Runtime.getRuntime().exec(command);
		BufferedReader stdout = 
		    new BufferedReader(new InputStreamReader(p.getInputStream()));
	    
		BufferedReader stderr = 
		    new BufferedReader(new InputStreamReader(p.getErrorStream()));
		String s;
		// read any errors from the attempted command
		if (streamOutput)
			System.out.println("Here is the standard error of the command (if any):\n");
		while ((s = stderr.readLine()) != null) {
			success = false;
			if (streamOutput) {
				System.out.println(s);
			}
		}
		if (streamOutput)
			System.out.println("Here is the standard output of the command:\n");
		while ((s = stdout.readLine()) != null) {
			if (streamOutput) {
				System.out.println(s);
			}
		}
            

	    p.waitFor();
	}
	catch (Exception e) {
	    System.out.println("exception happened - here's what I know: ");
	    e.printStackTrace();
	    System.exit(-1);
	}
	return success;
    }
    
    public static void writeIndices(String filename, 
    		List<PredictionPaper> testDocs, 
    		Map<PaperAbstract, Integer> testIndices) {
    	PlusoneFileWriter fileWriter = 
			new PlusoneFileWriter(filename);
		for (PredictionPaper paper : testDocs) {
			fileWriter.write(testIndices.get(paper) + " ");
		}
		fileWriter.write("\n");
		fileWriter.close();
    }
    
    /**
	 * Reads in a file and interprets it as a matrix (each line is a row,
	 * each entry is a column)
	 * 
	 * @param filename
	 * 		Name of the file to be read
	 * @param exp
	 * 		Flag to exponentiate entries
	 * @return
	 * 		double[][] array containing the read matrix
	 */
	public static double[][] readMatrix(String filename, boolean exp) {
		List<String[]> gammas = new ArrayList<String[]>();
		double[][] results = null;
		
		try {
			FileInputStream fstream = new FileInputStream(filename);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;

			while ((strLine = br.readLine()) != null) {
				gammas.add(strLine.trim().split(" "));
			}
			br.close();
			
			results = new double[gammas.size()][];
			for (int i = 0; i < gammas.size(); i++) {
				results[i] = new double[gammas.get(i).length];
				for (int j = 0; j < gammas.get(i).length; j++) {
					results[i][j] = new Double(gammas.get(i)[j]);
					if (exp)
						results[i][j] = Math.exp(results[i][j]);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return results;
	}

	public static void createLdaInfo(String filename, int numTopics, 
			int numTerms) {
		PlusoneFileWriter fileWriter = new PlusoneFileWriter(filename);
	    fileWriter.write("num_topics " + numTopics + " \n");
	    fileWriter.write("num_terms " + numTerms + " \n");
	    fileWriter.write("alpha " + 
	                        readAlpha("src/datageneration/output/final.other") 
	                         + " \n");
	    fileWriter.close();
	}
	
	/**
    * Reads in the value of alpha from a *.other file, contained in the LDA output
    * 
    * @param filename the path to a *.other file
    * @return the numerical value of alpha
    */
    public static double readAlpha(String filename) {
        FileInputStream filecontents = null;
            try {
                filecontents = new FileInputStream(filename);
            } catch (FileNotFoundException e) {
                System.out.println("Couldn't read LDA alpha");
                e.printStackTrace();
        }
        Scanner lines = new Scanner(filecontents);
        String alphaLine = lines.nextLine();
        alphaLine = lines.nextLine();
        alphaLine = lines.nextLine();
        String[] splitLine = alphaLine.split(" ");
        return Double.parseDouble(splitLine[1]);
    }

    public static double readDoubleFromFile(String path) throws IOException {
        FileInputStream filecontents = new FileInputStream(path);
        Scanner lines = new Scanner(filecontents);
        String line = lines.nextLine();
        return Double.parseDouble(line);
    }
}
