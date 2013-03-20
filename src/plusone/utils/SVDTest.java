package plusone.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

import static org.junit.Assert.*;

public class SVDTest {
    final static double eps = 1e-2;

    protected static class Corpus {
        List<PaperAbstract> papers = new ArrayList<PaperAbstract>();

        public Corpus() {
        }

        public void addPaper(int ... frequencies) {
            PaperAbstract paper = new PaperAbstract(papers.size(), null, null,
                                                    freqMap(frequencies));
            paper.generateTf(0, null, false);
            papers.add(paper);
        }

        public PaperAbstract getPaperAbstract(int i) {
            return papers.get(i);
        }

        public List<TrainingPaper> asTrainingPapers() {
            return new ArrayList<TrainingPaper>(papers);
        }
    }

    static Map<Integer, Integer> freqMap(int[] freqArray) {
        Map<Integer, Integer> freqMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < freqArray.length; ++i) {
            if (freqArray[i] > 0) freqMap.put(i, freqArray[i]);
        }
        return freqMap;
    }

    /**
     * Ask for the first two singular vectors of a simple diagonal matrix.
     */
    @Test public void svdFindsSingularValuesOfDiagonalMatrix() {
        Corpus corpus = new Corpus();
        corpus.addPaper(5, 0, 0, 0, 0);
        corpus.addPaper(0, 0, 0, 0, 0);
        corpus.addPaper(0, 0, 2, 0, 0);
        corpus.addPaper(0, 0, 0, 7, 0);

        SVD svd = new SVD(2, corpus.asTrainingPapers(), 5);

        double[] expectedSingularValues = {7, 5};
        assertArrayEquals(expectedSingularValues, svd.getSingularValues(), eps);
    }

    /**
     * See whether SVD's project() method decomposes papers as we expect.
     */
    @Test public void svdProjectsCorrectly() {
        Corpus corpus = new Corpus();
        corpus.addPaper(1, 0, 0, 0, 0, 0);
        corpus.addPaper(0, 2, 0, 0, 0, 0);
        corpus.addPaper(0, 0, 3, 4, 0, 0);
        corpus.addPaper(0, 0, 0, 0, 5, 0);
        corpus.addPaper(0, 0, 0, 0, 6, 0);

        SVD svd = new SVD(4, corpus.asTrainingPapers(), 6);

        /* We expect the following topics:
         *   [0, 0, 0  , 0  , 1, 0]  sigma = sqrt(61)
         *   [0, 0, 0.6, 0.8, 0, 0]  sigma = 5
         *   [0, 1, 0  , 0  , 0, 0]  sigma = 2
         *   [1, 0, 0  , 0  , 0, 0]  sigma = 1
         */

        double[] expectedSingularValues = {Math.sqrt(61), 5, 2, 1};
        assertArrayEquals(expectedSingularValues, svd.getSingularValues(), eps);

        double[][] expectedNormalizedTopics =
            {{0, 0, 0, 0, 1, 0},
             {0, 0, 0.6, 0.8, 0, 0},
             {0, 1, 0, 0, 0, 0},
             {1, 0, 0, 0, 0, 0}};
        double[][] actualNormalizedTopics = svd.getNormalizedTopics();
        for (int topic = 0; topic < expectedNormalizedTopics.length; ++topic)
            assertArrayEquals(expectedNormalizedTopics[topic],
                              actualNormalizedTopics[topic], eps);

        double[][] expectedProjections =
            {{0, 0, 0, 1},
             {0, 0, 2, 0},
             {0, 5, 0, 0},
             {5, 0, 0, 0},
             {6, 0, 0, 0}};
        for (int doc = 0; doc < expectedProjections.length; ++doc)
            assertArrayEquals(expectedProjections[doc],
                    svd.projection(corpus.getPaperAbstract(doc)), eps);
    }

    /**
     * Test whether SVD.predict() is fair to topics of different popularity.
     */
    @Test public void svdPredictsTopicsOfDifferentMagnitudesFairly() {
        Corpus corpus = new Corpus();
        corpus.addPaper(1000, 1000, 0, 0);
        corpus.addPaper(0, 0, 1, 1);

        SVD svd = new SVD(2, corpus.asTrainingPapers(), 4);

        int[] testFrequencies = {1, 0, 1, 0};
        PaperAbstract testPaper =
            new PaperAbstract(2, null, null, freqMap(testFrequencies));
        testPaper.generateTf(0, null, false);

        double[] predictions = svd.predict(testPaper);
        /* We shouldn't predict words that have already been seen. */
        assertEquals(0, predictions[0], eps);
        assertEquals(0, predictions[2], eps);
        /* The remaining words should be predicted with equal weight. */
        assertTrue(String.format("The norm-1000 topic got weight %f, but the norm-1 topic got weight %f",
                       predictions[1], predictions[3]),
                   Math.abs(predictions[1] - predictions[3]) < eps);
        assertTrue(String.format("magnitude: %f", predictions[1]), predictions[1] > eps);
    }

    @Test public void svdProjectsToOverlappingTopics() {
        Corpus corpus = new Corpus();
        corpus.addPaper(1, 1, 1, 0);
        corpus.addPaper(0, 1, 1, 2);

        SVD svd = new SVD(2, corpus.asTrainingPapers(), 4);

        int[] testFrequencies = {0, 1, 0, 0};
        PaperAbstract testPaper =
            new PaperAbstract(2, null, null, freqMap(testFrequencies));
        testPaper.generateTf(0, null, false);

        double[] predictions = svd.predict(testPaper);
        /* We shouldn't predict words that have already been seen. */
        assertEquals(0, predictions[1], eps);
        /* The remaining words should be predicted with nonzero weights of a
         * certain ratio. */
        assertTrue(Math.abs(predictions[2]) > eps);
        assertEquals(predictions[2] * 0.8, predictions[0], eps);
        assertEquals(predictions[2] * 0.4, predictions[3], eps);
    }
};
