package plusone.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.BeforeClass;
import org.junit.Test;
import plusone.Main;
import plusone.utils.PaperAbstract;

import static org.junit.Assert.*;
import static plusone.utils.PaperAbstract.freqMap;

public class PaperAbstractTest {
	static PaperAbstract simplePaperAbstract(int ... wordFreqs) {
		Integer[] inRefs = {};
		Integer[] outRefs = {};
		return new PaperAbstract(0, inRefs, outRefs, freqMap(wordFreqs));
	}

	@BeforeClass
	public static void setUpClass() {
		Main.initRandGen();
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

	@Test public void generateTfForTrainingHasCorrectCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		pa.generateTf(1, null, false);
		assertFalse(pa.isTest());

		Integer[] expectedTrainingWords = {0, 2, 3};
		assertEquals(new HashSet<Integer>(Arrays.asList(expectedTrainingWords)),
					 pa.getTrainingWords());
		assertEquals(5, (int)pa.getTrainingTf(0));
		assertEquals(0, (int)pa.getTrainingTf(1));
		assertEquals(3, (int)pa.getTrainingTf(2));
		assertEquals(1, (int)pa.getTrainingTf(3));
	}

	/**
	 * A testing document, with testWordpercent = 0.
	 */
	@Test public void generateTfForAllHeldOutHasCorrectCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		pa.generateTf(1, null, true);
		assertTrue(pa.isTest());

		assertEquals(0, pa.getTrainingWords().size());
		for (int i = 0; i < 4; ++i)
			assertEquals(0, (int)pa.getTrainingTf(i));

		Integer[] expectedTestingWords = {0, 2, 3};
		assertEquals(new HashSet<Integer>(Arrays.asList(expectedTestingWords)),
					 pa.getTestingWords());
		assertEquals(5, (int)pa.getTestingTf(0));
		assertEquals(0, (int)pa.getTestingTf(1));
		assertEquals(3, (int)pa.getTestingTf(2));
		assertEquals(1, (int)pa.getTestingTf(3));
	}

	/**
	 * A testing document, but with testWordpercent = 0.
	 */
	@Test public void generateTfForTestingNoneHeldOutHasCorrectCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		pa.generateTf(0, null, true);
		assertTrue(pa.isTest());

		Integer[] expectedTrainingWords = {0, 2, 3};
		assertEquals(new HashSet<Integer>(Arrays.asList(expectedTrainingWords)),
					 pa.getTrainingWords());
		assertEquals(5, (int)pa.getTrainingTf(0));
		assertEquals(0, (int)pa.getTrainingTf(1));
		assertEquals(3, (int)pa.getTrainingTf(2));
		assertEquals(1, (int)pa.getTrainingTf(3));

		assertEquals(0, pa.getTestingWords().size());
		for (int i = 0; i < 4; ++i)
			assertEquals(0, (int)pa.getTestingTf(i));
	}

	////////
	@Test public void generateTagTfWithNoneHeldHasOutCorrectCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Integer[] myTags = {0, 1, 2, 3};
		pa.generateTagTf(new ArrayList<Integer>(Arrays.asList(myTags)),
						 0, null);
		assertTrue(pa.isTest());

		Integer[] expectedTrainingWords = {0, 2, 3};
		assertEquals(new HashSet<Integer>(Arrays.asList(expectedTrainingWords)),
					 pa.getTrainingWords());
		assertEquals(5, (int)pa.getTrainingTf(0));
		assertEquals(0, (int)pa.getTrainingTf(1));
		assertEquals(3, (int)pa.getTrainingTf(2));
		assertEquals(1, (int)pa.getTrainingTf(3));

		assertEquals(0, pa.getTestingWords().size());
		for (int i = 0; i < 4; ++i)
			assertEquals(0, (int)pa.getTestingTf(i));
	}

	@Test public void generateTagTfWithAllHeldOutHasCorrectCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Integer[] myTags = {0, 1, 3};
		pa.generateTagTf(new ArrayList<Integer>(Arrays.asList(myTags)),
						 1, null);
		assertTrue(pa.isTest());

		Integer[] expectedTrainingWords = {2};
		assertEquals(new HashSet<Integer>(Arrays.asList(expectedTrainingWords)),
					 pa.getTrainingWords());
		assertEquals(0, (int)pa.getTrainingTf(0));
		assertEquals(0, (int)pa.getTrainingTf(1));
		assertEquals(3, (int)pa.getTrainingTf(2));
		assertEquals(0, (int)pa.getTrainingTf(3));

		Integer[] expectedTestingWords = {0, 3};
		assertEquals(new HashSet<Integer>(Arrays.asList(expectedTestingWords)),
					 pa.getTestingWords());
		assertEquals(5, (int)pa.getTestingTf(0));
		assertEquals(0, (int)pa.getTestingTf(1));
		assertEquals(0, (int)pa.getTestingTf(2));
		assertEquals(1, (int)pa.getTestingTf(3));
	}

	// TODO: interaction with Term
	// TODO: getTrainingTfAsSimpleMatrixRow
	// TODO: getNorm, isTest, similarity, combinedTf, freqMap
	// TODO: equals()
}
