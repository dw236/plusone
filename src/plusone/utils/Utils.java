package plusone.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;

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
}
