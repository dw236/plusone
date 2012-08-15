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

class Algorithms(object):
    algorithms = UNIVERSALS

def generate_html(dir, overwrite=False):
    filenames = os.listdir(dir)
    results = []
    files_found = 0
    for filename in filenames:
        if "experiment" in filename and filename[-5:] == ".json":
            files_found += 1
            result = parse_output.parse(dir + '/' + filename, external=False)
            results = add_result(results, result)
        else:
            continue
    print "processed", files_found, "files"
    
    overwrite = 'w' if overwrite else 'a'
    with open('data/test.html', overwrite) as f:
        f.write('<script src="sorttable.js"></script>\n')
        f.write('<head> <style type="text/css"> '
                + css() + ' </style></head>\n')
        for result in results:
            write_table(f, result)
            f.write('<br></br>\n')
    assert(f.closed)
    return results

def add_result(results, new_result):
    Algorithms.algorithms = set(list(Algorithms.algorithms) 
                                + new_result[2].keys())
    for result in results:
        params_true = check_all(result, new_result)
        algs_true = check_all(result, new_result, 2)
        all_true = params_true and algs_true
        if all_true:
            result.append(new_result)
            return results
    results.append([new_result])
    return results

def check_all(result, new_result, index=None):
    if index == None:
        index = 3
        all_true = True
        for option in UNIVERSALS:
                result_has_option = result[0][index].has_key(option)
                new_result_has_option = new_result[index].has_key(option)
                if result_has_option and new_result_has_option:
                    result_option = result[0][index][option]
                    new_result_option = new_result[index][option]
                    if  result_option != new_result_option:
                        all_true = False
                        break
                elif result_has_option != new_result_has_option:
                    all_true = False
                    break
    else:
        all_true = (set(result[0][index].keys()) 
                    == set(new_result[index].keys()))
    return all_true

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
        if algorithm not in HIDDEN:
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
        if any([any([cheat in algorithm for cheat in CHEAT]) \
                for algorithm in result[2]]):
            best_score_no_cheats = max(scores[:-2]) #ASSUMES ldaC15 and ldaT15 ARE LAST
        else:
            best_score_no_cheats = max(scores)
        best_cheating = max(scores[-2:]) #best score of cheating algorithms
        worst_score = min(scores)
        median_score = np.median(scores)
        for algorithm in sorted(result[2]):
            if algorithm not in HIDDEN:
                score = result[2][algorithm]['Predicted_Mean']
                if result[2][algorithm].has_key('Hover'):    
                    hoverList = result[2][algorithm]['Hover']
                else:
                    hoverList = []
                #highlight the best and worst score
                color = Color()
                to_bold = False
                if round(score, 2) == round(best_score_no_cheats, 2) \
                and not any([cheat in algorithm for cheat in CHEAT]):
                    color.add('g', 0xff)
                    to_bold = True 
                elif round(score, 2) == round(best_cheating, 2) \
                and any([cheat in algorithm for cheat in CHEAT]):
                    color.add('b', 0xFF)
                    color.add('g', 0x06)
                    to_bold = True
                elif round(score, 2) == round(worst_score, 2):
                    color.add('r', 0xFF)
                    color.add('g', 0x04)
                    to_bold = True
                elif round(score, 2) == round(median_score, 2):
                    color.add('r', 0xff)
                    color.add('g', 0xff)
                if hoverList is None or hoverList == []:
                    f.write('\t\t<td ' + str(color) + '>' +
                        bold(str(round(score, 2)), to_bold) + '</td>\n')
                else:
                    f.write('\t\t<td ' + str(color) + '>' +
                        hover(bold(str(round(score, 2)), to_bold), hoverList) +
                        '</td>\n')
        f.write('\t</tr>\n')
    f.write('</table>\n')

def bold(string, flag=True):
    if flag:
        return '<b>' + string + '</b>'
    else:
        return string

def hover(displayText, hoverList, flag=True):
    if flag:
        ret = ''
        for s in hoverList:
           ret += str(s) + '<br>'
        return ('<span class="dropt">'
                    + displayText
                + '<span style="width:500px;">' + ret + '</span> </span>')
    else:
        return displayText

def css():
    """
    first line:
        what shows up in the actual text box (it appears slightly below the
        actual text)
    second line:
        how the original text gets modified on mouseover
    third line:
        still trying to figure it out
    """
    return 'span.dropt {border-bottom: none; background: transparent;} \
    span.dropt:hover {text-decoration: none; background: transparent; z-index: 6; } \
    span.dropt span {position: absolute; left: -9999px; \
    margin: 20px 0 0 0px; padding: 3px 3px 3px 3px; \
    border-style:solid; border-color:black; border-width:1px; z-index: 6;} \
    span.dropt:hover span {left: 2%; background: #ffffff;} \
    span.dropt span {position: absolute; left: -9999px; \
    margin: 4px 0 0 0px; padding: 3px 3px 3px 3px; \
    border-style:solid; border-color:black; border-width:1px;} \
    span.dropt:hover span {margin: 20px 0 0 170px; background: #ffffff; z-index:6;}'

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
    directory and writes an html table displaying results (default values are \
    in parentheses)")
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
