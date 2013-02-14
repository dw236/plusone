package plusone.clustering;

import java.util.List;
import plusone.utils.SVD;
import plusone.utils.Terms;
import plusone.utils.TrainingPaper;

public class LSIOld0 extends LSI {
    public LSIOld0(int DIMENSION, List<TrainingPaper> trainingSet, Terms terms) {
        super(DIMENSION, trainingSet, terms, "LSIOld0");
        svd.setOldBehavior0(true);
    }
}
