package plusone.clustering;

import java.io.File;
import plusone.utils.PlusoneFileWriter;
import plusone.utils.PredictionPaper;
import plusone.utils.MetadataLogger.TestMetadata;
import java.util.List;

public abstract class ClusteringTest implements ClusteringMethod {

    public String testName;
    public ClusteringTest(String testName) {
	this.testName = testName;
    }
    public String getName(){
	return testName;
    }


    public double[] predict(PredictionPaper testPaper) { 

	return null;
    }
    public double[][] predict(List<PredictionPaper> testPaper){
    	return null;
    }
    
    public String[] getHover(){
    	return new String[]{""};
    }

    protected PlusoneFileWriter makePredictionWriter(int k, 
						     File outputDirectory, 
						     String extra) {
	if (outputDirectory == null) {
            return new PlusoneFileWriter();
        } else {
	    return new PlusoneFileWriter(new File(outputDirectory, 
                                                  this.testName + "-" + 
						  k + "-" +
                                                  (extra == null ? "" : 
						   "-" + extra) +
                                                  ".predict"));
	}
    }

    public void addMetadata(TestMetadata meta) {
	meta.createSingleValueEntry("expName", testName);
    }
}