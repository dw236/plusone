package plusone.utils;

import java.util.*;

/**
 * This class stores the results of a sequence of experiments.
 *
 * Typically, all the results listed in a single Results instance will be for
 * different runs of the same experiment.
 */
public class Results{
    final Set<String> allFieldNames;

    List<Map<String, Double>> resultValues;
    private String expName;

    public Results(Collection<String> allFieldNames, String expName){
	this.allFieldNames = new HashSet<String>(allFieldNames);
	this.expName=expName;
	resultValues = new ArrayList<Map<String, Double>>();
    }

    /**
     * Adds the results of another experiment run to this record of results.
     */
    public void addResult(Map<String, Double> values) {
	if (!allFieldNames.equals(values.keySet())) {
	    throw new IllegalArgumentException(
		"Results.addResult: wrong set of keys " +
		"(missing: " + setDifference(allFieldNames, values.keySet()) +
		"; extra: " + setDifference(values.keySet(), allFieldNames) +
		")");
	}
	// We make a copy of values, in case the caller re-uses the object.
	resultValues.add(new HashMap<String, Double>(values));
    }

    private double fieldMean(String fieldName) {
	double ret=0;
	for (Map<String, Double> m : resultValues)
	    ret += m.get(fieldName);
	ret = ret / resultValues.size();
	return ret;
    }
    private double fieldVariance(String fieldName) {
	double avg = fieldMean(fieldName);
	double ret=0;
	for (Map<String, Double> m : resultValues) {
	    double v = m.get(fieldName);
	    ret +=(v-avg)*(v-avg);
	}
	ret = ret / resultValues.size();
	return ret;
    }

    public Map<String, Double> getResultsMean() {
	Map<String, Double> ret = new HashMap<String, Double>();
	for (String fieldName : allFieldNames)
	    ret.put(fieldName, fieldMean(fieldName));
	return ret;
    }

    public Map<String, Double> getResultsVariance() {
	Map<String, Double> ret = new HashMap<String, Double>();
	for (String fieldName : allFieldNames)
	    ret.put(fieldName, fieldVariance(fieldName));
	return ret;
    }

    public Set setDifference(Set a, Set b) {
	Set diff = new HashSet(a);
	diff.removeAll(b);
	return diff;
    }
}
