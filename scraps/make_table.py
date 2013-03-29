#!/usr/bin/python

from os import walk
from os.path import join
from re import sub
from sys import argv

def printResults(table, fw, testingPercent, b):
    keys = table.keys()
    keys.pop(keys.index('k'))
    cn = filter(lambda x: 'cn-' in x, keys)
    knns = filter(lambda x: 'knn-' in x, keys)
    knncs = filter(lambda x: 'knnc-' in x, keys)
    knncbfs = filter(lambda x: 'knncbf-' in x, keys)
    knnrw = filter(lambda x: 'KNNRandomWalkPredictor' in x, keys)
    dtrw = filter(lambda x: 'DTRandomWalkPredictor' in x, keys)
    lsi = filter(lambda x: 'LSI-' in x, keys)
    

    cn = sorted(cn, key=lambda x: int(x.split('-')[1]))
    knns = sorted(knns, key=lambda x: int(x.split('-')[1]))
    knncs = sorted(knncs, key=lambda x: int(x.split('-')[1]))
    knncbfs = sorted(knncbfs, key=lambda x: int(x.split('-')[1]))
    knnrw = sorted(knnrw, key=lambda x: int(x.split('-')[1]))
    #dtrw = sorted(dtrw, key=lambda x: int(x.split('-')[1]))
    lsi = sorted(lsi, key=lambda x: int(x.split('-')[1]))


    """
    for i in range(len(table['k'])):
        f = open('graphdata/knn-k=%s-p=%s-b=%s.graphdata' % (table['k'][i], testingPercent, b), 'w')
        for knn in knns:
            f.write(knn.split('-')[1] + " " + table[knn][i] + "\n")

        f.close()            
    """
    
    rest = filter(lambda x: '-' not in x, keys)
    knns.extend(knncs)
    knns.extend(knncbfs)
    knns.extend(rest)
    knns.extend(lsi)
    knns.extend(cn)

    knns.insert(0, 'k')

    for i in knns:
        tmp = table[i]
        tmp.insert(0, i)
        print ' '.join([s.center(fw) for s in tmp])

def readFile(filename):
    f = open(filename, 'r')
    lines = f.read().strip().split('\n')
    return [x.split(':')[1] for x in lines]

if __name__ == '__main__':
    for a in walk(argv[1]).next()[1]:
        testingPercent = a
        print 'Testing percent:', testingPercent

        table = {'k':[]}
        for b in walk(join(argv[1], testingPercent)).next()[1]:
            k = b
            table['k'].append(b)
            for d in walk(join(argv[1], a, b)).next()[2]:
                if '.out' in d:
                    prefix = d.split('.')[0]
                    #true_table.append([prefix])
                    #false_table.append([prefix])
                    if prefix not in table:
                        table[prefix] = []

                        
                    predict = readFile(join(argv[1], a, b, d))[0]
                    table[prefix].append('%.3f' % (float(predict)))

                """
                lda_predict = readFile(join(argv[1], a, b, c, 'lda.out'))[0]
                #print "lda:", lda_predict
                table[1].append('%.3f' % (float(lda_predict)))

                knn_predict = readFile(join(argv[1], a, b, c, 'knn.out'))[0]
                #print "knn:", knn_predict
                table[2].append('%.3f' % (float(knn_predict)))

                baseline_predict = readFile(join(argv[1], a, b, c, 'baseline.out'))[0]
                #print "baseline:", baseline_predict
                table[3].append('%.3f' % (float(baseline_predict)))
                """

        fw = 12

        printResults(table, fw, testingPercent, True)
