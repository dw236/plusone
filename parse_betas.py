import argparse
import os
import numpy as np
from src.datageneration.util import *

def main(filename, save):
    """computes cosine similarities for a given experiment
    
    Computes the cosine similarities between matched topics for projector,
    LDA, and Mallet to the true topics the model uses to generate the data. 
    Writes the similarities to file.
    Also (optionally) saves the learned betas for future use.
    
    Args:
        filename: name of the file to be written
    
    Returns:
        none
    """
    projector_beta = 'projector/data/final.beta'
    lda_beta = 'lda/trained/final.beta'
    real_beta = 'src/datageneration/output/results.pickle'
    mallet_beta = 'Mallet/beta'
    save_dir = 'data'
    
    print "matching betas..."
    print "projector"
    projector_labels = match_beta(projector_beta, real_beta, metric='cosine', 
                                  plot=False, save=save, 
                                  save_name='projector', save_dir=save_dir)[-2]
    projector_norms = [pair[1] for pair in projector_labels]
    print "lda"
    lda_labels = match_beta(lda_beta, real_beta, metric='cosine', 
                            plot=False, save=save, save_name='lda',
                            save_dir=save_dir)[-2]
    lda_norms = [pair[1] for pair in lda_labels]
    print "mallet"
    mallet_labels = match_beta(mallet_beta, real_beta, metric='cosine',
                               plot=False, save=save, save_name='mallet', 
                               save_dir=save_dir)[-2]
    mallet_norms = [pair[1] for pair in mallet_labels]
    print "done"
    
    print "writing to file...",
    with open(filename, 'w') as f:
        for norm in projector_norms: 
            f.write(str(norm) + ' ')
        f.write('\n')
        
        for norm in lda_norms:
            f.write(str(norm) + ' ')
        f.write('\n')
        
        for norm in mallet_norms:
            f.write(str(norm) + ' ')
        f.write('\n')
    print "done"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="computes cosine similarities \
    of learned topics for lda, projector, and mallet matched to the real topics\
    (default values in parentheses)")
    parser.add_argument('f', metavar='ouput', action="store", 
                        help="name of file to be written")
    parser.add_argument('-s', action='store_true', default=False,
                        help="flag to save beta files (true)")
    args = parser.parse_args()
    
    if args.s:
        print "read contents will be saved to file"
    
    main(args.f, args.s)
