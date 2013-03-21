package plusone.utils;

import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import plusone.utils.PaperAbstract;

import static org.junit.Assert.*;
import static plusone.utils.PaperAbstract.freqMap;

public class PaperAbstractTest {
	static PaperAbstract simplePaperAbstract(Integer ... abstractWords) {
		Integer[] inRefs = {};
		Integer[] outRefs = {};
		return new PaperAbstract(0, inRefs, outRefs, abstractWords);
	}

    /**
     * Tests the constructor that takes a <code>Map&lt;Integer,Integer&gt; tf</code>.
     */
    @Test public void mapConstructerInitializesFields() {
		int index = 5;
		Integer[] inRefs = {0, 4, 3};
		Integer[] outRefs = {3, 5, 1, 9};
		PaperAbstract pa = new PaperAbstract(index, inRefs, outRefs, freqMap(0, 1));
		assertEquals(index, (int)pa.getIndex());
		assertArrayEquals(inRefs, pa.getInReferences());
		assertArrayEquals(outRefs, pa.getOutReferences());
    }

    /**
     * Tests the constructor that takes a <code>int[] abstractWords</code>.
     */
    @Test public void arrayConstructerInitializesFields() {
		int index = 5;
		Integer[] inRefs = {0, 4, 3};
		Integer[] outRefs = {3, 5, 1, 9};
		Integer[] abstractWords = {1};
		PaperAbstract pa =
			new PaperAbstract(index, inRefs, outRefs, abstractWords);
		assertEquals(index, (int)pa.getIndex());
		assertArrayEquals(inRefs, pa.getInReferences());
		assertArrayEquals(outRefs, pa.getOutReferences());
    }

	@Test public void getGroupReturnsParamOfSetGroup() {
		int group = 8;
		PaperAbstract pa = simplePaperAbstract();
		pa.setGroup(group);
		assertEquals(group, pa.getGroup());
	}

	// TODO: get{Training,Testing}{Tf,Words} for test and non-test docs
	// TODO: getTrainingTfAsSimpleMatrixRow
	// TODO: getNorm, isTest, similarity, combinedTf, freqMap
}
