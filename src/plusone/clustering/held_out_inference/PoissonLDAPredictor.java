package plusone.clustering.held_out_inference;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.ejml.simple.SimpleMatrix;
import plusone.clustering.ClusteringTest;
import plusone.utils.PredictionPaper;

public class PoissonLDAPredictor extends ClusteringTest {
    public enum PredictionMethod {
        WORD_DIST,    // Use the estimated probability distribution over words.
        WORD_PRESENT  // Use the estimated probability that the word is present.
    };

    static final String PYTHON_COMMAND = "python";
    static final String POISSON_LDA_PATH =
        "src/plusone/clustering/held_out_inference/poisson_lda.py";

    static final int PROGRESS_INTERVAL = 50;

    final PredictionMethod predictionMethod;
    final double testWordPercent;
    // The rate of the Poisson distribution of document lengths.
    final double lambda;
    final int numIterations;
    // topicStrengths is (num topics) x 1
    final SimpleMatrix topicStrengths, wordTopicMatrix;

    public PoissonLDAPredictor(
            String nameBase,
            double testWordPercent, double lambda, int numIterations,
            SimpleMatrix topicStrengths, SimpleMatrix wordTopicMatrix,
            PredictionMethod predictionMethod) {
        super(nameBase + "_h" + Integer.toString(numIterations) + "i" +
              predictionMethod.ordinal());
        this.testWordPercent = testWordPercent;
        this.lambda = lambda;
        this.numIterations = numIterations;
        this.topicStrengths = topicStrengths;
        this.predictionMethod = predictionMethod;
        if (PredictionMethod.WORD_DIST != predictionMethod) {
            throw new UnsupportedOperationException(
                "Only the PREDICT_DIST method is implemented.");
        }
        this.wordTopicMatrix = wordTopicMatrix;

        if (1 != topicStrengths.numCols() ||
            topicStrengths.numRows() != wordTopicMatrix.numCols()) {
            throw new IllegalArgumentException(
                "topicStrengths must be (num topics) x 1 and " +
                "wordTopicMatrix must be (vocabulary size) x (num topics).");
        }
    }

    static void writeMatrix(SimpleMatrix matrix, Writer writer)
            throws IOException {
        for (int row = 0; row < matrix.numRows(); ++row) {
            for (int col = 0; col < matrix.numCols(); ++col) {
                if (col > 0) writer.write(" ");
                writer.write(Double.toString(matrix.get(row, col)));
            }
            writer.write("\n");
        }
    }

    static double[] parseRowMatrix(String s) {
        String[] parts = s.split(" ");
        double[] ret = new double[parts.length];
        for (int i = 0; i < parts.length; ++i)
            ret[i] = Double.parseDouble(parts[i]);
        return ret;
    }

    static class CopyStreamThread extends Thread {
        final InputStream input;
        final OutputStream output;

        public CopyStreamThread(InputStream input, OutputStream output) {
            this.input = input;
            this.output = output;
        }

        public void run() {
            try {
                while (true) {
                    int b = input.read();
                    if (-1 == b) break;  // EOF
                    output.write(b);
                }
            } catch (IOException e) {
                throw new RuntimeException("IOException copying stderr: " +
                                           e.getMessage());
            }
        }
    }

    @Override
    public double[][] predict(List<PredictionPaper> testPapers) {
        String[] inferenceCommand = {
            // Run the poisson_lda.py script.
            PYTHON_COMMAND, POISSON_LDA_PATH,
            /* Note: to enable profiling, try something like this instead of
               the previous line:
              PYTHON_COMMAND, "-m", "cProfile", "-o", "plda.prof",
              POISSON_LDA_PATH, */
            "--test_word_prob", Double.toString(testWordPercent),
            "--lambda", Double.toString(lambda),
            "--num_iterations", Integer.toString(numIterations)
        };
        final int vocabSize = wordTopicMatrix.numRows();

        double[][] predictions = new double[testPapers.size()][];

        try {
            Process p = Runtime.getRuntime().exec(inferenceCommand);

            Writer stdin =
                new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));
            BufferedReader stdout =
                new BufferedReader(new InputStreamReader(p.getInputStream()));
            // Copy p's stderr to our stderr.
            (new CopyStreamThread(p.getErrorStream(), System.err)).start();
            writeMatrix(topicStrengths.transpose(), stdin);
            writeMatrix(wordTopicMatrix.transpose(), stdin);
            stdin.flush();
            for (int i = 0; i < testPapers.size(); ++i) {
                if (0 == i % PROGRESS_INTERVAL)
                    System.out.printf("PoissonLDAPredictor: paper %d of %d\n", i,
                                      testPapers.size());
                PredictionPaper testPaper = testPapers.get(i);
                writeMatrix(testPaper.getTrainingTfAsSimpleMatrixRow(vocabSize),
                            stdin);
                stdin.flush();
                predictions[i] = parseRowMatrix(stdout.readLine());
            }

            stdin.close();
            int exitStatus = p.waitFor();
            if (0 != exitStatus) {
                throw new RuntimeException(
                    "The poisson_lda script terminated with non-zero exit status "
                    + exitStatus);
            }
        } catch (IOException e) {
            throw new RuntimeException(
                "IOException while running the poisson_lda script: " +
                e.getMessage());
        } catch (InterruptedException e) {
            throw new RuntimeException(
                "InterruptedException while waiting for the poisson_lda script " +
                "to finish.  This is strange, because we have already " +
                "successfully processed all the output we needed from the " +
                "script.  Message: " + e.getMessage());
        }

        return predictions;
    }
}
