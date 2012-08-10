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
CHEATS = ['ldaC', 'ldaT']
PARAMS = ['a', 'b']
STATISTICS = ['sig_topics', 'sig_words']

def generate_html(dir, overwrite=False, star=False, short=False, quiet=False):
    filenames = os.listdir(dir)
    results = {}
    files_found = 0
    for filename in filenames:
        if "experiment" in filename and filename[-5:] == ".json":
            files_found += 1
            result = parse_output.parse(dir + '/' + filename, external=False)
            results = add_result(results, result)
        else:
            continue
    print "processed", files_found, "files"
    
    if not quiet:        
        overwrite = 'w' if overwrite else 'a'
        with open('data/results.html', overwrite) as f:
            f.write('<script src="sorttable.js"></script>\n')
            for result in results:
                write_table(f, results[result], result, star, short)
                f.write('<br></br>\n')
        assert(f.closed)
    return results

def add_result(results, new_result):
    universals = tuple([new_result[3][option] for option in UNIVERSALS])
    if universals not in results:
        results[universals] = {}
        #keep track of algorithm names and types
        results[universals]['algorithms'] = {}
        #=======================================================================
        # assumes each algorithm is run with double-digit topics as a parameter
        # names: includes topic number (eg lda15)
        # types: does not include topic number (eg lda)
        #=======================================================================
        results[universals]['algorithms']['names'] = set()
        results[universals]['algorithms']['types'] = set()
        results[universals]['algorithms']['totals'] = {}
    algorithms = results[universals]['algorithms']
    totals = results[universals]['algorithms']['totals']
    #alpha and beta parameters (sig_topics and sig_words, respectively)
    params = tuple([new_result[3][param] for param in PARAMS])
    if params not in results[universals]:
        results[universals][params] = {}
    
    for algorithm in sorted(new_result[2]):
        score = new_result[2][algorithm]['Predicted_Mean']
        results[universals][params][algorithm] = score
        algorithm_type = algorithm.strip('1234567890')
        assert(algorithm_type != None)
        if not is_cheat(algorithm):
            algorithms['names'].add(algorithm)
        else:
            algorithms['names'].add(algorithm_type)
            results[universals][params][algorithm_type] = score
        algorithms['types'].add(algorithm_type)
        
        #find the best algorithm by type:
        if results[universals][params].has_key(algorithm_type + '*'):
            current_score = results[universals][params][algorithm_type + '*']
            results[universals][params] \
            [algorithm_type + "*"] = max(current_score, score)
        else:
            results[universals][params][algorithm_type + "*"] = score
        #add score to totals (for determining best algorithm by topic param)
        algorithm_name = algorithm
        if is_cheat(algorithm):
            algorithm_name = algorithm_type
        if totals.has_key(algorithm_name):
            totals[algorithm_name] += score
        else:
            totals[algorithm_name] = score
        
    for statistic in sorted(new_result[4]):
        if statistic not in HIDDEN:
            results[universals][params][statistic] \
            = new_result[4][statistic]

    return results

def write_table(f, results, params, star=False, short=False):
    """
    note: DOES NOT CLOSE f
    """
    parameters = "" 
    for i in range(len(UNIVERSALS)):
        parameters += UNIVERSALS[i] + "=" + str(params[i]) + ", "
    parameters = parameters[:-2]

    f.write('<table border="1">\n')
    f.write('\t<th> Parameters ('+ parameters + ')</th>\n')
    f.write('</table>\n')
    f.write('<table border="1" class="sortable">\n')
    #===========================================================================
    # write the relevant statistics
    #===========================================================================
    for statistic in STATISTICS:
        f.write('\t\t<th>' + statistic + '</th>\n')
    #===========================================================================
    # write the algorithm names
    #===========================================================================
    algorithm_titles = get_algorithm_names(results['algorithms'], star, short)
    for algorithm in algorithm_titles:
        f.write('\t\t<th>' + algorithm + '</th>\n')
    #===========================================================================
    # write the numerical results
    #===========================================================================
    for result in results:
        if result != 'algorithms':
            scores = get_scores(results[result], results['algorithms'])
            f.write('\t<tr ' + mouse() + '>\n')
            for statistic in STATISTICS:
                f.write('\t\t<td>'
                         + str(results[result][statistic]) + '</td>\n')
            for algorithm in algorithm_titles:
                if results[result].has_key(algorithm):
                    color = Color()
                    to_bold = star and not short
                    if algorithm in scores['names'] or '*' in algorithm:
                        to_bold = True
                    score = round(results[result][algorithm], 2)
                    if is_cheat(algorithm):
                        if score == scores['best_cheating']:
                            color.add('b', 0xFF)
                            color.add('g', 0x06)
                    else:
                        if score == scores['worst']:
                            color.add('r', 0xFF)
                            color.add('g', 0x04)
                        elif score == scores['best']:
                            color.add('g', 0xff)
                        elif abs(score - scores['median']) \
                             <= 0.1 * scores['median']:
                            color.add('r', 0xff)
                            color.add('g', 0xff)
                        else:
                            pass
                    mouseover_text = []
                    alt = False
                    for name in scores['names']:
                        if not is_cheat(name) and star \
                        and ('*' in algorithm and algorithm[:-1] in name):
                            mouseover_text.append(name)
                            alt = True
                    if short and not to_bold:
                        algorithm_type = algorithm.strip('1234567890')
                        for name in scores['names']:
                            if not is_cheat(name) and algorithm_type in name:
                                mouseover_text.append(name)
                        alt = True
                    f.write('\t\t<td ' + alt_text(mouseover_text, alt) 
                            + str(color) + '>' +  bold(str(score), to_bold) 
                            + '</td>\n')
                else:
                    f.write('\t\t<td></td>\n')
            f.write('\t</tr>\n')
    f.write('</table>\n')

def get_algorithm_names(algorithms, star, short):
    if star:
        star_names = [algorithm + '*' for algorithm in algorithms['types']]
    if short: 
        short_names = get_scores(algorithms['totals'], algorithms)['names']
    if star and short:
        algorithm_names = set(star_names + short_names)
        new_names = []
        for algorithm in algorithm_names:
            if '*' in algorithm and algorithm[:-1] in algorithm_names:
                pass
            else:
                new_names.append(algorithm)
        algorithm_names = new_names
    elif star or short:
        if star:
            algorithm_names = star_names
        elif short:
            algorithm_names = short_names
        else:
            pass
    else:
        algorithm_names = sorted(algorithms['names'])
    
    return sorted(algorithm_names)

def is_cheat(algorithm):
    return any([cheat in algorithm for cheat in CHEATS])

def get_scores(result, algorithms):
    best_score_names = []
    for algorithm_type in algorithms['types']:
        best_score = -np.inf
        to_add = []
        for algorithm in algorithms['names']:
            if result.has_key(algorithm):
                if algorithm_type in algorithm:
                    #special case for lda (because 'lda' is in 'ldaC')
                    if algorithm_type == 'lda' and is_cheat(algorithm): 
                        continue
                    score = round(result[algorithm], 2)
                    if score > best_score:
                        to_add = [algorithm]
                        best_score = score
                    elif score == best_score:
                        to_add.append(algorithm)
        best_score_names += to_add
    best_cheating_score = -np.inf
    best_score = -np.inf #does not include cheating algorithms
    list_of_scores = [] #does not include cheating algorithms
    for algorithm in algorithms['names']:
        if result.has_key(algorithm):
            score = round(result[algorithm], 2)
            if is_cheat(algorithm):
                best_cheating_score = max(best_cheating_score, score)
            else:
                best_score = max(best_score, score)
                list_of_scores.append(score)
    median_score = np.median(list_of_scores)
    worst_score = min(list_of_scores)
    scores = {}
    scores['names'] = sorted(best_score_names)
    scores['best'] = best_score
    scores['best_cheating'] = best_cheating_score
    scores['worst'] = worst_score
    scores['median'] = median_score
    return scores

def bold(string, flag=True):
    if flag:
        return '<b>' + string + '</b>'
    else:
        return string

def mouse(mouse_off='"this.style.fontSize=\'medium\';', 
          mouse_on='"this.style.fontSize=\'x-large\';'):
    mouse_on += 'this.style.textDecoration=\'underline\'"'
    mouse_off += 'this.style.textDecoration=\'none\'"'
    
    return 'onMouseOut=' + mouse_off + ' onMouseOver=' + mouse_on

def alt_text(text=[], flag=True):
    if flag:
        to_add = ""
        for name in text:
            to_add += name + ', '
        to_add = to_add[:-2]
        return "title='algorithm(s): " + to_add + "'"
    else:
        return ''

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
    directory and writes an html table displaying results; assumes synthetic \
    data (default values are in parentheses)")
    parser.add_argument('f', metavar='directory', action="store", 
                        help="directory containing json files to be read")
    parser.add_argument('-o', action="store_true", default=False,
                        help="flag to overwrite existing table (False)")
    parser.add_argument('-q', action="store_true", default=False,
                        help="flag to suppress writing to file (False)")
    parser.add_argument('-*', action="store_true", default=False,
                        help="flag to display best of each algorithm \
                        by row (False)")
    parser.add_argument('-s', action="store_true", default=False,
                        help="flag to display best of each algorithm \
                        by column (False)")
    
    args = parser.parse_args()
    star = args.__getattribute__('*')
    print "reading experiment files from directory:", args.f
    if args.o:
        print "overwriting existing table with new results"
    if args.q:
        print "returning dictionary of results rather than writing html"
    if star:
        print "displaying best of each algorithm by row" 
        print "    (each entry is the best result for that run)"
    if args.s:
        print "displaying best of each algorithm by column"
        print "    (each column is the algorithm that did the best overall)"
    
    return generate_html(args.f, args.o, star, args.s, args.q)

if __name__ == '__main__':
    results = main()
    universals = results.keys()
     