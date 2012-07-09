import os
import argparse
import src.datageneration.util as util
import parse_output
import numpy as np

#globals
UNIVERSALS = ['k', 'n', 'l', 'm']
HIDDEN = UNIVERSALS + ['a', 'b'] \
                    + ['median', 'sum_squares_words', 'sum_squares_topics']

def generate_html(dir):
    filenames = os.listdir(dir)
    results = []
    files_found = 0
    for filename in filenames:
        if "experiment" in filename and filename[-5:] == ".json":
            files_found += 1
            #TODO: split different results into different lists
            result = parse_output.parse(dir + '/' + filename, external=False)
            results.append(result)
        else:
            continue
    print "processed", files_found, "files"
    with open('data/test.html', 'w') as f:
        f.write('<script src="sorttable.js"></script>\n')
        write_table(f, results)
    return results

def write_table(file_object, results):
    parameters = "" 
    for option in UNIVERSALS:
        if results[0][3].has_key(option):
            parameters += option + "=" + str(results[0][3][option]) + ", "
    parameters = parameters[:-2]
    with file_object as f:
        f.write('<table border="1">\n')
        f.write('\t<th colspan="'+ str(len(results[0][3].keys()) + 
                                      len(results[0][4].keys()) - 
                                      len(HIDDEN)) 
               +'">Parameters ('+ parameters + ')</th>\n')
        f.write('</table>\n')
        f.write('<table border="1" class="sortable">\n')
        #=======================================================================
        # write the parameter names
        #=======================================================================
        f.write('\t<tr>\n')
        for param in results[0][3]:
            if param not in HIDDEN:
                f.write('\t\t<th>' + param + '</th>\n')
        #relevant statistics for data
        for datum in results[0][4]:
            if datum not in HIDDEN:
                f.write('\t\t<th>' + datum + '</th>\n')
        #=======================================================================
        # write the algorithm names
        #=======================================================================
        for algorithm in sorted(results[0][2]):
            f.write('\t\t<th>' + algorithm + '</th>\n')
        f.write('\t</tr>\n')
        #=======================================================================
        # write the numerical results
        #=======================================================================
        for result in results:
            f.write('\t<tr>\n')
            for param in result[3]:
                if param not in HIDDEN:
                    f.write('\t\t<td>' + str(result[3][param]) + '</td>\n')
            for datum in result[4]:
                if datum not in HIDDEN:
                    f.write('\t\t<td>' + str(result[4][datum]) + '</td>\n')
            scores = [result[2][algorithm]['Predicted_Mean'] \
                      for algorithm in sorted(result[2])]
            best_score = max(scores)
            worst_score = min(scores)
            median_score = np.median(scores)
            for algorithm in sorted(result[2]):
                score = result[2][algorithm]['Predicted_Mean']
                #highlight the best and worst score
                best = ''
                worst = ''
                median = ''
                if round(score, 2) == round(best_score, 2):
                    best = ' bgcolor=#00FF00'
                if round(score, 2) == round(worst_score, 2):
                    worst = ' bgcolor=#FF0000'
                if abs(score - median_score) < 0.01:
                    median = ' bgcolor=#FFFF00'
                f.write('\t\t<td ' + best + worst + median + '>' +
                        str(round(score, 2)) + '</td>\n')
            f.write('\t</tr>\n')
        f.write('</table>')

def main():
    parser = argparse.ArgumentParser(description="reads all json files in a \
    directory and writes an html table displaying results")
    parser.add_argument('f', metavar='directory', action="store", 
                        help="directory containing json files to be read")
    
    args = parser.parse_args()
    print "reading experiment files from directory:", args.f
    
    return generate_html(args.f)

if __name__ == '__main__':
    results = main()
