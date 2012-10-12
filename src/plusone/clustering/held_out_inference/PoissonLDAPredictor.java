package plusone.clustering;

import org.ejml.simple.SimpleMatrix;

class PoissonLDAPredictor extends ClusteringTest {
    public enum PredictionMethod {
        WORD_DIST,    // Use the estimated probability distribution over words.
        WORD_PRESENT  // Use the estimated probability that the word is present.
    };

    final PredictionMethod predictionMethod;
    // The rate of the Poisson distribution of document lengths.
    final double lambda;
    // topicStrengths is (num topics) x 1
    final SimpleMatrix topicStrengths, wordTopicMatrix;

    public HeldOutInferencePredictor(
            String nameBase,
            double lambda,
            SimpleMatrix topicStrengths, SimpleMatrix wordTopicMatrix,
            PredictionMethod predictionMethod) {
        super(nameBase + "_hoi_" + predictionMethod.toString());
        this.lambda = lambda
        this.topicStrengths = topicStrengths;
        this.predictionMethod = predictionMethod;
        this.wordTopicMatrix = wordTopicMatrix;

        if (1 != topicStrengths.numCols() ||
            topicStrengths.numRows() != wordTopicMatrix.numCols()) {
            throw new IllegalArgumentException(
                "topicStrengths must be (num topics) x 1 and "
                "wordTopicMatrix must be (vocabulary size) x (num topics).");
        }
    }

    @Override
    public double[][] predict(List<PredictionPaper> testPaper) {
       return TODO();
    }
}
