import os
import argparse
import src.datageneration.util as util
import parse_output
import numpy as np

#globals
UNIVERSALS = ['k', 'n', 'l', 'm']
HIDDEN = UNIVERSALS + ['a', 'b'] \
                    + ['median', 'sum_squares_words', 'sum_squares_topics'] \
                    #+ ['LSI-5', 'LSI-10', 'knn-5', 'knn-10']
CHEAT = ['ldaC', 'ldaT']
PARAMS = UNIVERSALS + ['a', 'b']

def generate_html(dir, overwrite=False):
    filenames = os.listdir(dir)
    results = {}
    results[tuple(PARAMS)] = {}
    files_found = 0
    for filename in filenames:
        if "experiment" in filename and filename[-5:] == ".json":
            files_found += 1
            result = parse_output.parse(dir + '/' + filename, external=False)
            results = add_result(results, result)
        else:
            continue
    print "processed", files_found, "files"
    
#    overwrite = 'w' if overwrite else 'a'
#    with open('data/test.html', overwrite) as f:
#        f.write('<script src="sorttable.js"></script>\n')
#        for result in results:
#            write_table(f, result)
#            f.write('<br></br>')
#    assert(f.closed)
    return results

def add_result(results, new_result):
    params = tuple([new_result[3][option] for option in PARAMS])
    if params in results[tuple(PARAMS)]:
        for algorithm in sorted(new_result[2]):
            results[tuple(PARAMS)][params][algorithm] \
             = new_result[2][algorithm]['Predicted_Mean']
    else:
        results[tuple(PARAMS)][params] = {}
        for algorithm in sorted(new_result[2]):
            results[tuple(PARAMS)][params][algorithm] \
             = new_result[2][algorithm]['Predicted_Mean']
    
    return results

def main():
    parser = argparse.ArgumentParser(description="reads all json files in a \
    directory and writes an html table displaying results; assumes synthetic \
    data (default values are in parentheses)")
    parser.add_argument('f', metavar='directory', action="store", 
                        help="directory containing json files to be read")
    parser.add_argument('-o', action="store_true", default=False,
                        help="flag to overwrite existing table (False)")
    
    args = parser.parse_args()
    print "reading experiment files from directory:", args.f
    if args.o:
        print "overwriting existing table with new results"
    
    return generate_html(args.f, args.o)

if __name__ == '__main__':
    results = main()