import json
import argparse
import src.datageneration.util as util

TEST = 'tests'
PARAMS = 'parameters'
DATA = 'data'
HIDDEN = ['LSI-5', 'LSI-10', 'knn-5', 'knn-10']

def parse(filename, show=False, external=True):
    """@TODO
    Assumes json file is of the form: 
    dict with three entries: 'tests', which is a one-element list of dicts,
    'parameters', which is a dict of parameters used to generate the data,
    and 'data', a dict of relevant statistics regarding the model used to
    generate the data.
    In the case of real data, 'parameters' and 'data' will be empty dicts.
    
    For 'tests':
    The inner dict has experiment names as the keys and dicts with various
    performance metrics as values
    
    Args:
        external:
            flag for whether this function is being called directly from 
            the command line (ie it is being called by main() below); 
            set this to False if you want to call it from anywhere else
    
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    names = []
    scores = []
    for test in data[TEST]:
        for key in sorted(test.keys(), cmp=(lambda x,y: cmp(test[x], test[y]))):
            if key not in HIDDEN:
                names.append(str(key))
                scores.append(test[key]['Predicted_Mean'])
    
    if external:
        util.figure(figsize=(12,7))
        util.plot_dist(scores, labels=names)
        if data[DATA]:
            for datum in data[DATA]:
                if datum == 'sig_words' or datum == 'sig_topics':
                    util.plot(0, label=str(datum) + " " 
                              + str(data[DATA][datum]))
            util.legend(loc='best')
           
        if show:
            util.show()
        else:
            util.savefig(filename + '.pdf')
    
    return names, scores, data[TEST][0], data[PARAMS], data[DATA]

def main():
    parser = argparse.ArgumentParser(description="reads a json file and plots\
    its constituent data")
    parser.add_argument('f', metavar='filename', action="store", 
                        help="json file to be read")
    parser.add_argument('-q', action="store_true", default=False, 
                        help="flag to suppress writing to file")
    
    args = parser.parse_args()
    print "reading file:", args.f
    if args.q:
        print "not writing to file and instead displaying plot with console"
        
    return parse(args.f, show=args.q)

if __name__ == '__main__':
    data = main()
