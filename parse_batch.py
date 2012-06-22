import os
import argparse
import src.datageneration.util as util
import parse_output

def generate_html(dir):
    filenames = os.listdir(dir)
    results = []
    files_found = 0
    for filename in filenames:
        if "experiment" in filename and filename[-5:] == ".json":
            files_found += 1
            results.append(parse_output.parse(dir + '/' + filename, 
                                              external=False))
        else:
            continue
    print "processed", files_found, "files"
    with open('data/test.html', 'w') as f:
        universals = ['k', 'n', 'l', 'm']
        hidden = ['a', 'b'] + universals
        parameters = ""
        for option in universals:
            parameters += option + "=" + str(results[0][3][option]) + ", "
        parameters = parameters[:-2]
        f.write('<table border="1">\n')
        f.write('\t<th colspan="'+ str(len(results[0][3].keys()) + 
                                       len(results[0][4].keys())) 
                +'">Parameters ('+ parameters + ')</th>\n')
        f.write('\t<th colspan="'+ str(len(results[0][2].keys())) 
                +'">Experiments</th>\n')
        #=======================================================================
        # write the parameter names
        #=======================================================================
        f.write('\t<tr>\n')
        for param in results[0][3]:
            if param not in hidden:
                f.write('\t\t<td>' + param + '</td>\n')
        for datum in results[0][4]:
            f.write('\t\t<td>' + datum + '</td>\n')
        #=======================================================================
        # write the algorithm names
        #=======================================================================
        for algorithm in results[0][2]:
            f.write('\t\t<td>' + algorithm + '</td>\n')
        f.write('\t</tr>\n')
        #=======================================================================
        # write the numerical results
        #=======================================================================
        for result in results:
            f.write('\t<tr>\n')
            for param in result[3]:
                if param not in hidden:
                    f.write('\t\t<td>' + str(result[3][param]) + '</td>\n')
            for datum in result[4]:
                f.write('\t\t<td>' + str(result[4][datum]) + '</td>\n')
            for algorithm in result[2]:
                score = result[2][algorithm]['Predicted_Mean']
                f.write('\t\t<td>' + str(round(score, 2)) + '</td>\n')
            f.write('\t</tr>\n')
        f.write('</table>')
    return results

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