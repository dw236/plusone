package plusone.utils;

import java.util.Set;
import org.ejml.simple.SimpleMatrix;

/**
 * Interface extended by training and prediction papers.
 */
public interface PaperIF {
    public Integer getIndex();
    public Integer getTrainingTf(Integer word);
    public SimpleMatrix getTrainingTfAsSimpleMatrixRow(int vocabSize);
    public Set<Integer> getTrainingWords();
    public Integer[] getInReferences();
    public Integer[] getOutReferences();
}
