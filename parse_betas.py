import argparse
import numpy as np
from src.datageneration.util import *

def main(filename):
    projector_beta = 'projector/data/final.beta'
    lda_beta = 'lda/trained/final.beta'
    real_beta = 'src/datageneration/output/results.pickle'
    
    projector_labels = match_beta(projector_beta, real_beta, 
                                  metric='L1', plot=False)[-2]
    projector_norms = [pair[1] for pair in projector_labels]
    
    lda_labels = match_beta(lda_beta, real_beta, metric='L1', plot=False)[-2]
    lda_norms = [pair[1] for pair in lda_labels]
    
    output = np.array([projector_norms, lda_norms])
    output.tofile(filename, ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="computes L1 norms of \
    matched topics for lda and projector (matched to the real topics)")
    parser.add_argument('f', metavar='ouput', action="store", 
                        help="name of file to be written")
    args = parser.parse_args()
    main(args.f)
