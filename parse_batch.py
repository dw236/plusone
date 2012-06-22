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
        f.write('<table border="1">\n')
        f.write('\t<th colspan="100">Experiments</th>\n')
        names = True
        for result in results:
            f.write('\t<tr>\n')
            scores = []
            for algorithm in result[2]:
                f.write('\t\t<td>' + algorithm + '</td>\n')
                scores.append(result[2][algorithm]['Predicted_Mean'])
            f.write('\t</tr>\n')
            f.write('\t<tr>\n')
            for score in scores:
                f.write('\t\t<td>' + str(round(score, 2)) + '</td>\n')
            f.write('\t</tr>\n')
            f.write('\t<tr>\n')
            params = []
            for param in result[3]:
                f.write('\t\t<td>' + param + '</td>\n')
                params.append(result[2][param])
            f.write('\t</tr>\n')
            f.write('\t<tr>\n')
            for param in params:
                f.write(f.write('\t\t<td>' + str(param) + '</td>\n'))
            f.write('\t</tr>\n')
#            for name in result[0]:
#                f.write('\t\t<td>' + name + '</td>\n')
#            f.write('\t</tr>\n')
#            f.write('\t<tr>\n')
#            for score in result[1]:
#                f.write('\t\t<td>' + str(score) + '</td>\n')
#            f.write('\t</tr>\n')
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