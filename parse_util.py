import itertools
import numpy as np

from parse_generated import generate_html
from matplotlib.pyplot import *
from src.datageneration.util import ind_cmp

def group_flat_results(flat_results, group_keys=['a', 'b']):
    results = {}
    for result in flat_results:
        try:
            params = result[group_keys]
        except:
            params = tuple(result[key] for key in group_keys)
        print params
        if results.has_key(params):
            results[params].append(result)
        else:
            results[params] = [result]
    groups = []
    for param in results:
        groups.append(results[param])
    return groups

def get_stats(grouped_flat_results):
    stats = []
    for group in grouped_flat_results:
        params = group[0].keys()
        stat = {param:np.mean([entry[param] for entry in group])
                for param in params}
        stats.append(stat)
    all_stats = {param:[stat[param] for stat in stats]
                 for param in params}
    return all_stats

def get_all_stats(all_results):
    """grouped-flat-results of multiple tables
    """
    algs = ['lda', 'ldaT', 'ldaC', 'projector', 'Baseline', 'LSI']
    all_stats = []
    for group in all_results:
        stats = {}
        for entry in group:
            params = entry.keys()
            for param in params:
                suffix = '-' + str(entry['k'])
                if param in [alg + suffix for alg in algs]:
                    key = param[:-len(suffix)]
                else:
                    key = param
                if stats.has_key(key):
                    stats[key].append(entry[param])
                else:
                    stats[key] = [entry[param]]
        all_stats.append(stats)
    return all_stats

def plot_stats(stats, x, algs=None, flat=False):
    if algs == None:
        algs = ['lda', 'ldaT', 'ldaC', 'projector', 'Baseline', 'LSI']
    colors = 'brgmcyk'
    clf()
    k = int(stats['k'][0])
    for i in range(len(algs)):
        color = colors[i]
        alg = algs[i]
        if flat and alg != 'Baseline':
            alg += '-' + str(k)
        x_s = stats[x]
        scores = stats[alg]
        indices = sorted(range(len(x_s)), cmp=ind_cmp(x_s))
        plot(sorted(x_s), [scores[i] for i in indices], color + '.', label=alg)
        legend(loc="best")
        
def count_words(docs, voc_size=1000):
    pairwise_totals = np.zeros((voc_size, voc_size))
    print "calculating pairwise totals..."
    index = 0
    for doc in docs:
        if index % 100 == 0:
            print "reached document", index
        for i,j in itertools.combinations(doc, 2):
            pairwise_totals[i, j] += 1
            pairwise_totals[j, i] += 1
        index += 1
    print "done"
    totals = np.zeros(voc_size)
    for doc in docs:
        for word in doc:
            totals[word] += 1
    return totals, pairwise_totals

def which_topics(words, voc_size=1000):
    which_topics = {}
    sig_words = get_sig_words(words)
    top_words = [sorted(range(len(topic)), cmp=ind_cmp(topic), reverse=True) 
                 for topic in words]
    for word in range(voc_size):
        which_topics[word] = []
        for i in range(len(top_words)):
            if top_words[i].index(word) < sig_words[i]:
                which_topics[word].append(i)
    return which_topics

def distance(metric='ratio'):
    if metric == 'ratio':
        return lambda x,y: x / y
    elif metric == 'diff':
        return lambda x,y: np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return lambda x,y: 0
    else:
        print "supported types: ratio, diff"
        raise Exception("unrecognized input: " + str(metric))

def get_scores(results, metric=distance('ratio')):
    universals = results.keys()
    k, n, l, m = universals[0]
    scores = results[universals[0]]
    tc_scores, pc_scores, tp_scores, x_s = [], [], [], []
    points = []
    for r in scores:
        if r != "algorithms":
            ldaT = np.mean(scores[r]['ldaT-' + str(k)]['score'])
            ldaC = np.mean(scores[r]['ldaC-' + str(k)]['score'])
            projector = np.mean(scores[r]['projector-' + str(k)]['score'])
            sig_topics = float(scores[r]['sig_topics'])
            sig_words = float(scores[r]['sig_words'])
            tc_scores.append(metric(ldaT, ldaC))
            pc_scores.append(metric(projector, ldaC))
            tp_scores.append(metric(projector, ldaT))
            x = sig_topics / k * (max(sig_words * k / m, 1))**2
            x_s.append(x)
            points.append([k]+ list(r))    
    return x_s, tc_scores, pc_scores, tp_scores, points

def get_all_scores(k_s, metric='diff'):
    metric = distance(metric)
    x_s, tc_scores, pc_scores, tp_scores = [], [], [], []
    points = []
    for k in k_s:
        filename = "data/k" + str(k) + ".n1000.l75.m1000"
        results = generate_html(filename, quiet=True)[0]
        x, tc, pc, tp, p = get_scores(results, metric)
        x_s += x
        tc_scores += tc
        pc_scores += pc
        tp_scores += tp
        points += p
        
    x_s = np.array(x_s)
    tc_scores = np.array(tc_scores)
    pc_scores = np.array(pc_scores)
    tp_scores = np.array(tp_scores)
    points = np.array(points)    
    return x_s, tc_scores, pc_scores, tp_scores, points

def plot_scores(x_s, y_s, max_x=np.inf, color='g', clear=True):
    plot_xs = sorted(x_s[x_s < max_x])
    indices = sorted(range(len(plot_xs)), cmp=ind_cmp(x_s))
    if clear:
        clf()
    #plot(plot_xs, [tc_scores[i] for i in indices], '.')
    #plot(plot_xs, [pc_scores[i] for i in indices], 'r.-')
    plot(plot_xs, [y_s[i] for i in indices], color + '.')
