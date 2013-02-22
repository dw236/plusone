package cc.mallet.topics;

import cc.mallet.types.*;
import java.util.Arrays;

public class HierarchicalLDAWithPrediction extends HierarchicalLDA {
	// Based on HierarchicalLDA.empiricalLikelihood().
	public double[] predictNextWord(int numSamples, FeatureSequence doc) {
		NCRPNode[] path = new NCRPNode[numLevels];
		NCRPNode node;
		double weight;
		path[0] = rootNode;

		int sample, level, type, token, seqLen;

		Dirichlet dirichlet = new Dirichlet(numLevels, alpha);
		double[] levelWeights;
		double[] multinomial = new double[numTypes];

		double[][] likelihoods = new double[ numSamples ][ numTypes ];
		double maxLikelihood = Double.NEGATIVE_INFINITY;

		for (sample = 0; sample < numSamples; sample++) {
			Arrays.fill(multinomial, 0.0);

			for (level = 1; level < numLevels; level++) {
				path[level] = path[level-1].selectExisting();
			}
	    
			levelWeights = dirichlet.nextDistribution();
	    
			for (type = 0; type < numTypes; type++) {
				for (level = 0; level < numLevels; level++) {
					node = path[level];
					multinomial[type] +=
						levelWeights[level] * 
						(eta + node.typeCounts[type]) /
						(etaSum + node.totalTokens);
				}

			}

			for (type = 0; type < numTypes; type++) {
				multinomial[type] = Math.log(multinomial[type]);
			}

			seqLen = doc.getLength();
			
			double sampleLikelihood = 0.0;
			for (token = 0; token < seqLen; token++) {
				type = doc.getIndexAtPosition(token);
				sampleLikelihood += multinomial[type];
			}
			for (type = 0; type < numTypes; type++) {
				double typeLikelihood = sampleLikelihood + multinomial[type];
				likelihoods[sample][type] = typeLikelihood;
				if (typeLikelihood > maxLikelihood)
					maxLikelihood = typeLikelihood;
			}
		}
	
		double[] nextWordWeights = new double[ numTypes ];
		for (sample = 0; sample < numSamples; sample++) {
			for (type = 0; type < numTypes; type++) {
				nextWordWeights[type] +=
					Math.exp(likelihoods[sample][type] - maxLikelihood);
			}
		}

		double totalWeight = 0.0;
		for (type = 0; type < numTypes; type++) {
			totalWeight += nextWordWeights[type];
		}

		double[] predictions = new double[ numTypes ];
		for (type = 0; type < numTypes; type++) {
			predictions[type] = nextWordWeights[type] / totalWeight;
		}

		return predictions;
	}
}
