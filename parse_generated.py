import os
import argparse
import parse_output
import numpy as np
from parse_batch import css, hover
import pickle #HACK--REMOVE ASAP

#globals
UNIVERSALS = ['k', 'n', 'l', 'm']
HIDDEN = UNIVERSALS + ['a', 'b'] \
                    + ['median', 'sum_squares_words', 'sum_squares_topics']
CHEATS = ['ldaC', 'ldaT']
PARAMS = ['a', 'b']
STATISTICS = ['sig_topics', 'sig_words']

def generate_html(dir, overwrite=False, star=False, short=False, quiet=False):
    filenames = os.listdir(dir)
    # See the docstring for add_result for the structure of the results dict.
    results = {}
    flat_results = []
    files_found = 0
    for filename in filenames:
        if "experiment" in filename and filename[-5:] == ".json":
            files_found += 1
            result = parse_output.parse(dir + '/' + filename, external=False)
            flat_result = flatten_result(result)
            flat_results.append(flat_result)
            results = add_result(results, result)
        else:
            continue
    print "processed", files_found, "files"
    
    if not quiet:        
        overwrite = 'w' if overwrite else 'a'
        with open('data/results.html', overwrite) as f:
            f.write('<script src="sorttable.js"></script>\n')
            f.write('<head><style type="text/css"> ' + css() + 
                    '</style></head>\n')
            for result in results:
                write_table(f, results[result], result, star, short)
                f.write('<br></br>\n')
        assert(f.closed)
    return results, flat_results

def flatten_result(new_result):
    result = dict((option, new_result[3][option]) for option in UNIVERSALS)
    for param in PARAMS:
         result[param] = new_result[3][param]
    for algorithm in sorted(new_result[2]):
        result[algorithm] = new_result[2][algorithm]['Predicted_Mean']
    for statistic in sorted(new_result[4]):
        if statistic not in HIDDEN:
            result[statistic] = new_result[4][statistic]
    
    return result

def add_result(results, new_result):
    """
    add_result is used to build a Python dictionary object with information
    about a collection of experimental results.  The keys of the dictionary are
    tuples of option values in the same order as the options appear in the
    UNIVERSALS variable.  The value corresponding to a particular key is itself
    a dictionary, and contains information about all experiments that were run
    on data generated with those options.

    Given an option tuple o, results[o] has an entry for each setting of
    parameters (alpha and beta, corresponding to sig_topics and sig_words).
    Each entry is a dictionary with an entry for each algorithm ("knn-15") and
    algorithm type ("knn"), and some special entries like "knn*".  The value
    corresponding to each of these is a dictionary with two keys: "score" maps
    to a list of prediction score means, and "hover" is related to hovertext
    that should appear in an html table (but might be ignored right now).

    results[o] also has a special entry with key "algorithms", which is itself a
    dictionary with four keys: names, types and totals.  names is a list of the
    algorithms that were run (e.g. "knn-15"), and types includes just the basic
    algorithm names (e.g. "knn").  totals is a dictionary mapping n algorithm
    name to the sum of the prediction scores for that algorithm.

    To use add_result, start with an empty dictionary called "results".  For each new
    result new_result, call add_result(results, new_result).

    Arguments:
    results -- Information about a collection of results, as described above.
               This parameter is modified by add_results.
    new_result -- The new result to add.

    Returns: results
    """
    universals = tuple([new_result[3][option] for option in UNIVERSALS])
    if universals not in results:
        results[universals] = {}
        #=======================================================================
        # Keep track of algorithm names and types
        # names: includes topic number (eg lda15)
        # types: does not include topic number (eg lda)
        #=======================================================================
        results[universals]['algorithms'] = {'names':set(),
                                             'types':set(),
                                             'totals':{}}
    algorithms = results[universals]['algorithms']
    totals = results[universals]['algorithms']['totals']
    
    #alpha and beta parameters (sig_topics and sig_words, respectively)
    params = tuple([new_result[3][param] for param in PARAMS])
    if params not in results[universals]:
        results[universals][params] = {}
    entry = results[universals][params]
    
    for algorithm in sorted(new_result[2]):
        #extract score and hover text
        score = new_result[2][algorithm]['Predicted_Mean']
        if new_result[2][algorithm].has_key('Hover'):    
            hoverList = new_result[2][algorithm]['Hover']
        else:
            hoverList = []
        
        check(entry, algorithm, score) #initialize dictionary if necessary
        entry[algorithm]['score'].append(score)
        entry[algorithm]['hover'] = hoverList
        
        algorithm_type = algorithm.strip('1234567890-')
        if not is_cheat(algorithm):
            algorithms['names'].add(algorithm)
        else:
            algorithms['names'].add(algorithm_type)
            check(entry, algorithm_type, score)
            entry[algorithm_type]['score'].append(score)
            entry[algorithm_type]['hover'] = hoverList
        algorithms['types'].add(algorithm_type)
        
        #find the best algorithm by type:
        if entry.has_key(algorithm_type + '*'):
            current_score = entry[algorithm_type + '*']['score']
            entry[algorithm_type + "*"]['score'] = max(current_score, score)
        else:
            entry[algorithm_type + "*"] = {'score':score,
                                           'hover':[]}
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
            entry[statistic] = new_result[4][statistic]
    
    #get subsets of algorithms (for comparisons by algorithm)
    algorithms['subsets'] = get_algorithm_subsets(algorithms)

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
    algorithm_subsets = results['algorithms']['subsets']
    for result in results:
        if result != 'algorithms':
            scores = get_scores(results[result], results['algorithms'])
            f.write('\t<tr ' + mouse() + '>\n')
            for statistic in STATISTICS:
                top_three = ''#hack_1(statistic, params, result, results) #HACK
                if statistic == 'sig_topics':
                    top_three = [result[0]]
                if statistic == 'sig_words':
                    top_three = [result[1]]
                f.write('\t\t<td>'
                         + hover(str(results[result][statistic]), 
                                 top_three) + '</td>\n')
            for algorithm in algorithm_titles:
                if results[result].has_key(algorithm):
                    color = Color()
                    to_bold = star and not short
                    if algorithm in scores['names'] or '*' in algorithm:
                        to_bold = True
                    score = round(np.mean(results[result][algorithm]['score']),
                                  2)
                    if '~' in algorithm:
                        pass
                    elif is_cheat(algorithm):
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
                        and ('*' in algorithm and \
                             name in algorithm_subsets[algorithm[:-1]]):
                            mouseover_text.append(name)
                            alt = True
                    if short and not to_bold:
                        algorithm_type = algorithm.strip('1234567890-')
                        for name in scores['names']:
                            if not is_cheat(name) and \
                            name in algorithm_subsets[algorithm_type]:
                                mouseover_text.append(name)
                        alt = True
                    if not short and not star:
                        mouseover_text = results[result][algorithm]['hover']
                        alt = not (mouseover_text == [] 
                                   or mouseover_text == [""])
                        if alt:
                            mouseover_text = ''#hack_2(mouseover_text) #HACK
                    mouseover_text = results[result][algorithm]['score']
                    alt = True
                    f.write('\t\t<td ' + str(color) + '>' 
                            + hover(bold(str(score), to_bold), 
                                    mouseover_text, alt) 
                            + '</td>\n')
                else:
                    f.write('\t\t<td></td>\n')
            f.write('\t</tr>\n')
    f.write('</table>\n')

def hack_1(statistic, params, result, results):
    """THIS IS A HACK--REMOVE ASAP"""
    filename = "k" + str(params[0]) + "." \
    + "n" + str(params[1]) + "." \
    + "l" + str(params[2]) + "." \
    + "m" + str(params[3]) + "." \
    + "a" + str(float(result[0])) + "." \
    + "b" + str(result[1])
    with open("src/datageneration/output/" + filename +"/results.pickle", 'r') as d:
        docs, doc_topics, words, topics, args = pickle.load(d)
    if statistic == 'sig_topics':
        to_traverse = topics
        lines = [str(line).split(" ") for line in results[result]['ldaC']['hover']]
        indices = []
        for line in lines:
            indices.append(int([float(num) for num in line][0]))
        indices = np.array(indices)
    if statistic == 'sig_words':
        to_traverse = words
        indices = np.array(range(args.k))
    top_three = list(np.array([sorted(range(len(topic)), 
                                      cmp=lambda x,y: cmp(topic[x],
                                                          topic[y]),
                                      reverse=True)[:3]
                               for topic in to_traverse])[indices])
    return top_three

def hack_2(mouseover_text):
    """THIS IS A HACK--REMOVE ASAP"""
    mouseover_text = [str(line).split(" ")[1:] for line in 
                      mouseover_text]
    predictions = []
    for line in mouseover_text:
        predictions.append([float(num) for num in line])
    top_three = [sorted(range(len(prediction)), 
                        cmp=lambda x,y: cmp(prediction[x],
                                           prediction[y]),
                        reverse=True)[:3]
                 for prediction in predictions]
    mouseover_text = top_three
    return mouseover_text

def check(entry, algorithm, current_score):
    """ Checks if an entry has results for an algorithm. If it doesn't, it 
        creates an empty dictionary
    NOTE: modifies entry in-place
    """
    if not entry.has_key(algorithm):
        entry[algorithm] = {'score':[]}
    #else:
        #if entry[algorithm]['score'] != current_score:
            #print "adding new score for:", algorithm

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
        algorithm_names = algorithms['names']
    
    return sorted(algorithm_names)

def get_algorithm_subsets(algorithms):
    subsets = {}
    for algorithm_type in algorithms['types']:
        if not subsets.has_key(algorithm_type):
            subsets[algorithm_type] = set()
        for name in algorithms['names']:
            try:
                match = (algorithm_type == name[:len(algorithm_type)])
            except:
                match = False
            if match and (not is_cheat(name) or is_cheat(algorithm_type)):
                subsets[algorithm_type].add(name)
    return subsets

def is_cheat(algorithm):
    return any([cheat in algorithm for cheat in CHEATS])

def get_scores(result, algorithms):
    subsets = algorithms['subsets']
    best_score_names = []
    for algorithm_type in algorithms['types']:
        if '~' in algorithm_type:
            continue
        best_score = -np.inf
        to_add = []
        for algorithm in algorithms['names']:
            if result.has_key(algorithm):
                if algorithm in subsets[algorithm_type]:
                    score = round(np.mean(result[algorithm]['score']), 2)
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
        if '~' in algorithm:
            continue
        if result.has_key(algorithm):
            score = round(np.mean(result[algorithm]['score']), 2)
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
    mouse_on += '"'#'this.style.textDecoration=\'underline\'"'
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
    results, flat_results = main()
    universals = results.keys()
 
