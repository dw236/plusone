package plusone.utils;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

import static org.junit.Assert.*;

public class SVDTest {
    final static double eps = 1e-2;

    protected static class Corpus {
        List<PaperAbstract> papers = new ArrayList<PaperAbstract>();

        public Corpus() {
        }

        public void addPaper(int ... frequencies) {
            PaperAbstract paper = new PaperAbstract(papers.size(), null, null, frequencies);
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

    /**
     * Ask for the first two singular vectors of a simple diagonal matrix.
     */
    @Test public void singularValuesOfDiagonalMatrixTest() {
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
    @Test public void projectionTest() {
        Corpus corpus = new Corpus();
        corpus.addPaper(1, 0, 0, 0, 0, 0);
        corpus.addPaper(0, 2, 0, 0, 0, 0);
        corpus.addPaper(0, 0, 3, 4, 0, 0);
        corpus.addPaper(0, 0, 0, 0, 5, 0);
        corpus.addPaper(0, 0, 0, 0, 6, 0);

        SVD svd = new SVD(4, corpus.asTrainingPapers(), 6);

        /* We expect the following topics:
         *   [0, 0, 0, 0, sqrt(61), 0]  sigma = sqrt(61)
         *   [0, 0, 3, 4, 0, 0]  sigma = 5
         *   [0, 2, 0, 0, 0, 0]  sigma = 2
         *   [1, 0, 0, 0, 0, 0]  sigma = 1
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
             {0, 0, 1, 0},
             {0, 1, 0, 0},
             {5 / Math.sqrt(61), 0, 0, 0},
             {6 / Math.sqrt(61), 0, 0, 0}};
        for (int doc = 0; doc < expectedProjections.length; ++doc)
            assertArrayEquals(expectedProjections[doc],
                    svd.projection(corpus.getPaperAbstract(doc)), eps);
    }
};
