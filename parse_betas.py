import argparse
import os
import numpy as np
from src.datageneration.util import *

def main(filename):
    projector_beta = 'projector/data/final.beta'
    lda_beta = 'lda/trained/final.beta'
    real_beta = 'src/datageneration/output/results.pickle'
    
    projector_labels = match_beta(projector_beta, real_beta, 
                                  metric='cosine', plot=False)[-2]
    projector_norms = [pair[1] for pair in projector_labels]
    
    lda_labels = match_beta(lda_beta, real_beta, metric='cosine', plot=False)[-2]
    lda_norms = [pair[1] for pair in lda_labels]
    
    with open(filename, 'w') as f:
        for norm in projector_norms: 
            f.write(str(norm) + ' ')
        f.write('\n')
        for norm in lda_norms:
            f.write(str(norm) + ' ')
        f.write('\n')

def parse_all(dir):
    filenames = os.listdir(dir)
    files_found = 0
    projector_norms = []
    lda_norms = []
    results = {}
    for filename in filenames:
        if "norms" in filename:
            files_found += 1
            params = parse_params(filename)
            for param in params:
                if results.has_key(param):
                    results[param].append(params[param])
                else:
                    results[param] = [params[param]]
            norm = np.fromfile(dir + '/' + filename, sep=' ')
            norm.reshape(2, len(norm)/2)
            projector_norms.append(np.mean(norm[0]))
            lda_norms.append(np.mean(norm[1]))
    print "processed", files_found, "files"
    results['projector'] = projector_norms
    results['lda'] = lda_norms
    return results

def parse_params(filename):
    values = filename.split('.')
    values[-4] += '.' + values[-3]
    values[-2] += '.' + values[-1]
    values = values[1:-3] + [values[-2]]
    params = {}
    for param in values:
        key = param[0]
        value = float(param[1:])
        params[key] = value
    params['sig_topics'] = avg_sig_words(params['a'], int(params['k']))
    params['sig_words'] = avg_sig_words(params['b'], int(params['m']))
    params['x'] = params['sig_topics'] / params['k'] * \
                  (max(params['sig_words'] * params['k'] / params['m'], 1))**2
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="computes L1 norms of \
    matched topics for lda and projector (matched to the real topics)")
    parser.add_argument('f', metavar='ouput', action="store", 
                        help="name of file to be written")
    args = parser.parse_args()
    main(args.f)
