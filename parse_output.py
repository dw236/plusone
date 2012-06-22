import json
import argparse
import src.datageneration.util as util

TEST = 'tests'
PARAMS = 'parameters'
DATA = 'data'

def parse(filename, table=False, show=False, external=True):
    """
    Assumes json file is of the form: 
    dict with two entries: 'tests', which is a one-element list of dicts, and
    'parameters', which is a dict of parameters used to generate the data
    (only for synthetic datasets)
    For 'tests':
    The inner dict has experiment names as the keys and dicts with various
    performance metrics as values
    
    Args:
        external:
            flag for whether this function is being called from the command 
            line (set this to False if you want to call it from another python
            script)
    
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    names = []
    scores = []
    for test in data[TEST]:
        for key in sorted(test.keys(), cmp=(lambda x,y: cmp(test[x], test[y]))):
            names.append(str(key))
            scores.append(test[key]['Predicted_Mean'])
    
    if external:
        util.plot_dist(scores, labels=names)
        if table:
            """so hacky I don't even want to talk about it"""
            data_info = make_table()
            sig_words, sum_squares, num_docs, avg_words = data_info
            util.plot(0, label='sig_words: ' + str(round(sig_words, 2)))
            util.plot(0, label='sum_squares: ' + str(round(sum_squares, 2)))
            util.plot(0, label='num_docs: ' + str(int(num_docs)))
            util.plot(0, label='avg_words: ' + str(round(avg_words, 2)))
            util.legend(loc='best')
           
        if show:
            util.show()
        else:
            util.savefig(filename + '.pdf')
    
    return names, scores, data[TEST][0], data[PARAMS], data[DATA]

def make_table():
    with open('src/datageneration/output/documents_other-out', 'r') as f:
        data = f.readlines()
    data = [float(line.strip(' \n')) for line in data]
    return data

def main():
    parser = argparse.ArgumentParser(description="reads a json file and plots\
    its constituent data")
    parser.add_argument('f', metavar='filename', action="store", 
                        help="json file to be read")
    parser.add_argument('-s', action="store_true", default=False, 
                        help="flag for if synthetic data was used")
    parser.add_argument('-q', action="store_true", default=False, 
                        help="flag to suppress writing to file")
    
    args = parser.parse_args()
    print "reading file:", args.f
    if args.q:
        print "not writing to file and instead displaying plot with console"
        
    return parse(args.f, args.s, args.q)

if __name__ == '__main__':
    data = main()