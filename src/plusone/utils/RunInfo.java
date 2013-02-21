package plusone.utils;

import java.util.HashMap;

public class RunInfo extends HashMap<String, Object> {
	public Double getDoubleOrInfinity(String key) {
		if (containsKey(key)) {
			return (Double)get(key);
		} else {
			return Double.POSITIVE_INFINITY;
		}
	}
}
