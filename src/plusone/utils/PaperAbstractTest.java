package plusone.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.ejml.simple.SimpleMatrix;
import org.junit.BeforeClass;
import org.junit.Test;
import plusone.Main;
import plusone.utils.PaperAbstract;

import static org.junit.Assert.*;
import static plusone.utils.PaperAbstract.freqMap;

public class PaperAbstractTest {
	final static double eps = 1e-9;

	static PaperAbstract simplePaperAbstract(int ... wordFreqs) {
		Integer[] inRefs = {};
		Integer[] outRefs = {};
		return new PaperAbstract(0, inRefs, outRefs, freqMap(wordFreqs));
	}

	static Terms.Term[] newTermArray(int numTerms) {
		Terms.Term[] terms = new Terms.Term[numTerms];
		for (int i = 0; i < numTerms; ++i) {
			terms[i] = new Terms.Term(i);
		}
		return terms;
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

	/**
	 * Test that <code>generateTf</code> updates its <code>terms</code>
	 * parameter when the document is a training document.
	 */
	@Test public void generateTfUpdatesNonHeldOutTerms() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Terms.Term[] terms = newTermArray(4);
		pa.generateTf(0, terms, false);

		List<PaperAbstract> justPa = new ArrayList<PaperAbstract>();
		justPa.add(pa);
		List<PaperAbstract> empty = new ArrayList<PaperAbstract>();

		assertEquals(5, terms[0].totalCount);
		assertEquals(justPa, terms[0].getDocTrain());
		assertEquals(empty, terms[0].getDocTest());

		assertEquals(0, terms[1].totalCount);
		assertEquals(empty, terms[1].getDocTrain());
		assertEquals(empty, terms[1].getDocTest());

		assertEquals(3, terms[2].totalCount);
		assertEquals(justPa, terms[2].getDocTrain());
		assertEquals(empty, terms[2].getDocTest());

		assertEquals(1, terms[3].totalCount);
		assertEquals(justPa, terms[3].getDocTrain());
		assertEquals(empty, terms[3].getDocTest());
	}

	/**
	 * Test that <code>generateTf</code> updates its <code>terms</code>
	 * parameter when the document is a testing document and all worsd are held
	 * out.  It should update the list of testing documents of each term but
	 * never change the word count.
	 */
	@Test public void generateTfUpdatesHeldOutTermDocsButNotCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Terms.Term[] terms = newTermArray(4);
		pa.generateTf(1, terms, true);

		List<PaperAbstract> justPa = new ArrayList<PaperAbstract>();
		justPa.add(pa);
		List<PaperAbstract> empty = new ArrayList<PaperAbstract>();

		assertEquals(0, terms[0].totalCount);
		assertEquals(empty, terms[0].getDocTrain());
		assertEquals(justPa, terms[0].getDocTest());

		assertEquals(0, terms[1].totalCount);
		assertEquals(empty, terms[1].getDocTrain());
		assertEquals(empty, terms[1].getDocTest());

		assertEquals(0, terms[2].totalCount);
		assertEquals(empty, terms[2].getDocTrain());
		assertEquals(justPa, terms[2].getDocTest());

		assertEquals(0, terms[3].totalCount);
		assertEquals(empty, terms[3].getDocTrain());
		assertEquals(justPa, terms[3].getDocTest());
	}

	/**
	 * Test that <code>generateTf</code> does not update the term counts for a
	 * testing document, even for words that are not held out.
	 */
	@Test public void generateTfDoesNotUpdateTestCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Terms.Term[] terms = newTermArray(4);
		pa.generateTf(0, terms, true);

		for (int i = 0; i < 4; ++i)
			assertEquals(0, terms[i].totalCount);
	}

	/**
	 * Test that <code>generateTestingTagTf</code> does not update the term
	 * counts.
	 */
	@Test public void generateTestingTagTfDoesNotUpdateTestCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Terms.Term[] terms = newTermArray(4);
		Integer[] myTags = {0, 1, 2, 3};
		pa.generateTestingTagTf(new ArrayList<Integer>(Arrays.asList(myTags)),
								0, terms);

		for (int i = 0; i < 4; ++i)
			assertEquals(0, terms[i].totalCount);
	}

	@Test public void generateTestingTagTfWithNoneHeldHasOutCorrectCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Integer[] myTags = {0, 1, 2, 3};
		pa.generateTestingTagTf(new ArrayList<Integer>(Arrays.asList(myTags)),
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

	@Test public void generateTestingTagTfWithAllHeldOutHasCorrectCounts() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Integer[] myTags = {0, 1, 3};
		pa.generateTestingTagTf(new ArrayList<Integer>(Arrays.asList(myTags)),
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

	@Test public void getTrainingTfAsSimpleMatrixRowCorrectCorrectNonHeldOut() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		pa.generateTf(0, null, true);
		SimpleMatrix result = pa.getTrainingTfAsSimpleMatrixRow(4);
		assertEquals(1, result.numRows());
		assertEquals(4, result.numCols());
		assertEquals(5, result.get(0, 0), eps);
		assertEquals(0, result.get(0, 1), eps);
		assertEquals(3, result.get(0, 2), eps);
		assertEquals(1, result.get(0, 3), eps);
	}

	@Test public void getTrainingTfAsSimpleMatrixRowCorrectCorrectTesting() {
		PaperAbstract pa = simplePaperAbstract(5, 0, 3, 1);
		Integer[] myTags = {0, 1, 3};
		pa.generateTestingTagTf(new ArrayList<Integer>(Arrays.asList(myTags)),
								1, null);
		SimpleMatrix result = pa.getTrainingTfAsSimpleMatrixRow(4);
		assertEquals(1, result.numRows());
		assertEquals(4, result.numCols());
		assertEquals(0, result.get(0, 0), eps);
		assertEquals(0, result.get(0, 1), eps);
		assertEquals(3, result.get(0, 2), eps);
		assertEquals(0, result.get(0, 3), eps);
	}

	@Test public void normIsL2NormOfNonHeldOut() {
		PaperAbstract pa = simplePaperAbstract(42, 0, 3, 4);
		Integer[] myTags = {0};
		pa.generateTestingTagTf(new ArrayList<Integer>(Arrays.asList(myTags)),
								1, null);
		// norm is sqrt(3^2+4^2)
		assertEquals(5.0, pa.getNorm(), eps);
	}

	/**
	 * We test this separately since <code>generateTf</code> and
	 * <code>generateTestingTagTf</code> separately implement the norm
	 * calulation.
	 */
	@Test public void normIsL2NormForTraining() {
		PaperAbstract pa = simplePaperAbstract(0, 5, 12);
		pa.generateTf(0, null, false);
		// norm is sqrt(5^2+12^2)
		assertEquals(13.0, pa.getNorm(), eps);
	}

	@Test public void similarityIsDotProductOfNonHeldOut() {
		PaperAbstract pa0 = simplePaperAbstract(42, 0, 3, 4);
		PaperAbstract pa1 = simplePaperAbstract(8, 5, 12, 0);
		Integer[] myTags = {0};
		List<Integer> myTagsList =
			new ArrayList<Integer>(Arrays.asList(myTags));
		pa0.generateTestingTagTf(myTagsList, 1, null);
		pa1.generateTestingTagTf(myTagsList, 1, null);
		// Ignoring word 0, pa0 has norm 5 and pa1 has norm 13.
		assertEquals((3.0/5.0*12.0/13.0), pa0.similarity(pa1), eps);
	}

	/**
	 * Tests the <code>getCombinedTf</code> method with a list of two
	 * documents: one training and one testing with some words held out.
	 */
	@Test public void combinedTfIgnoresHeldOut() {
		PaperAbstract pa0 = simplePaperAbstract(2, 1, 0, 3);
		PaperAbstract pa1 = simplePaperAbstract(4, 0, 8, 12);
		Integer[] myTags = {0, 2};
		List<Integer> myTagsList =
			new ArrayList<Integer>(Arrays.asList(myTags));
		// pa0 is a training document.
		pa0.generateTf(0, null, false);
		// pa1 is a testing document.
		pa1.generateTestingTagTf(myTagsList, 1, null);

		List<TrainingPaper> papers = new ArrayList<TrainingPaper>();
		papers.add(pa0);  papers.add(pa1);
		Map<Integer, Integer> tf = PaperAbstract.getCombinedTf(papers);

		assertEquals(3, tf.size());
		assertEquals(1, (int)tf.get(0));
		assertEquals(1, (int)tf.get(1));
		assertEquals(2, (int)tf.get(3));
	}

	// TODO: freqMap
	// TODO: equals()
}
