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
            results = add_result(results, result)
        else:
            continue
    print "processed", files_found, "files"
    with open('data/test.html', 'w') as f:
        f.write('<script src="sorttable.js"></script>\n')
        for result in results:
            write_table(f, result)
            f.write('<br></br>')
    assert(f.closed)
    return results

def add_result(results, new_result):
    for result in results:
        all_true = True
        for option in UNIVERSALS:
            result_has_option = result[0][3].has_key(option)
            new_result_has_option = new_result[3].has_key(option)
            result_option = result[0][3][option]
            new_result_option = new_result[3][option]
            if result_has_option and new_result_has_option:
                if  result_option != new_result_option:
                    all_true = False
                    break
            elif result_has_option != new_result_option:
                all_true = False
                break
        if all_true:
            result.append(new_result)
            return results
    results.append([new_result])
    return results

def write_table(f, results):
    """
    note: DOES NOT CLOSE f
    """
    parameters = "" 
    for option in UNIVERSALS:
        if results[0][3].has_key(option):
            parameters += option + "=" + str(results[0][3][option]) + ", "
    parameters = parameters[:-2]

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
            color = Color()
            if round(score, 2) == round(best_score, 2):
                color.add('g', 0xff)
            if round(score, 2) == round(worst_score, 2):
                color.add('r', 0xff)
            if round(score, 2) == round(median_score, 2):
                color.add('r', 0xff)
                color.add('g', 0xff)
            f.write('\t\t<td ' + str(color) + '>' +
                    str(round(score, 2)) + '</td>\n')
        f.write('\t</tr>\n')
    f.write('</table>')

class Color(object):
    """class to handle colors for table cells
    """
    def __init__(self, r=0, g=0, b=0):
        self.r = r
        self.g = g
        self.b = b
    
    def white(self):
        self.r = 0
        self.g = 0
        self.b = 0
    
    def add(self, color, amount):
        def smooth(color, amount):
            if color + amount > 255:
                self.white()
                return 255
            else:
                return color + amount
        if color in ['r', 'red']:
            self.r = smooth(self.r, amount)
        elif color in ['g', 'green']:
            self.g = smooth(self.g, amount)
        elif color in ['b', 'blue']:
            self.b = smooth(self.b, amount)
        else:
            print "unrecognized color:", color
            
    def __str__(self):
        """html attribute 'bgcolor' with appropriate color
        """
        def string(color):
            if color == 0:
                return '00'
            else:
                return hex(color)[2:]
        color = ''
        if self.r or self.g or self.b:
            color = 'bgcolor=#'
            color += string(self.r) + string(self.g) +string(self.b)
        return color 

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
