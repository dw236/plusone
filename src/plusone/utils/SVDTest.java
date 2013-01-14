package plusone.utils;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

import static org.junit.Assert.*;

public class SVDTest {
    protected static class Corpus {
        List<TrainingPaper> papers = new ArrayList<TrainingPaper>();

        public Corpus() {
        }

        public void addPaper(int ... frequencies) {
            PaperAbstract paper = new PaperAbstract(papers.size(), null, null, frequencies);
            paper.generateTf(0, null, false);
            papers.add(paper);
        }

        public List<TrainingPaper> asTrainingPapers() {
            return papers;
        }
    }

    /*
    protected static void assertArrayEquals(double[] expecteds, double[] actuals,
                                            double delta) {
        assertEquals(expecteds.length, actuals.length);
        for (int i = 0; i < expecteds.length; ++i)
            assertEquals(expecteds[i], actuals[i]);
    }
    */

    /**
     * Ask for the first two singular vectors of the diagonal matrix
     *
     *   5 0 0 0 0
     *   0 0 0 0 0
     *   0 0 2 0 0
     *   0 0 0 7 0
     *
     *   (five columns for five words, and four rows for four documents).
     */
    @Test public void singularValuesOfDiagonalMatrix() {
        Corpus corpus = new Corpus();
        corpus.addPaper(5, 0, 0, 0, 0);
        corpus.addPaper(0, 0, 0, 0, 0);
        corpus.addPaper(0, 0, 2, 0, 0);
        corpus.addPaper(0, 0, 0, 7, 0);

        SVD svd = new SVD(2, corpus.asTrainingPapers(), 5);

        double[] expectedSingularValues = {7, 5};
        assertArrayEquals(expectedSingularValues, svd.getSingularValues(), 1e-4);
    }
};
