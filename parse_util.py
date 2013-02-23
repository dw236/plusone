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

def plot_all_deep_stats(stats, k=20, clear=True, save=False):
    for i in range(len(stats)):
        figure(i)
        plot_deep_stats(stats, k, i, clear)
        xlabel('sig words')
        ylabel('precision')
        sig_topics = np.mean(stats[i]['sig_topics'])
        title('average sig topics: ' + str(sig_topics))
        if save:
            savefig('k' + str(k) + 'a' + str(sig_topics) + '.pdf')

def plot_deep_stats(stats, k=20, index=0, clear=True):
    """plot the output of get_deep_stats()
    """
    if clear:
        clf()
    #algs = ['lda', 'ldaT', 'ldaC', 'projector', 'Baseline', 'LSI']
    algs = ['lda', 'ldaC', 'projector', 'Baseline', 'LSI', 'kmeans', 
            'malletlda']
    markers = ['.', 'x', 'd', 'o', '-', '*', '^']
    colors = 'bgrymkc'
    for alg in algs:
        if alg != 'Baseline':
           alg_name = alg + '-' + str(k)
        else:
            alg_name = alg
        sig_words = sorted(stats[index]['sig_words'])
        indices = sorted(range(len(sig_words)), 
                         cmp=ind_cmp(stats[index]['sig_words']))
        scores = stats[index][alg_name]
        plot(sig_words, [scores[i] for i in indices], 
             colors[algs.index(alg)] + markers[algs.index(alg)] + '-', 
             label=alg)
    legend(loc='best')

def group_deep_result(result, key_ind=0):
    results = {}
    for key in result:
        if key == 'algorithms':
            continue
        param = key[key_ind]
        if results.has_key(param):
            results[param].append(result[key])
        else:
            results[param] = [result[key]]
    groups = []
    for key in results:
        groups.append(results[key])
    return groups, results

def get_deep_stats(deep_groups):
    stats = []
    for group in deep_groups:
        params = group[0].keys()
        stat = {}
        for param in params:
            nums = []
            for entry in group:
                try:
                    num = np.mean(entry[param]['score'])
                except:
                    num = entry[param]
                nums.append(num)
            stat[param] = nums
        stats.append(stat)
    return stats

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

def plot_scores(x_s, y_s, max_x=np.inf, color='g.', label=None, clear=True):
    plot_xs = sorted(x_s[x_s < max_x])
    indices = sorted(range(len(plot_xs)), cmp=ind_cmp(x_s))
    if clear:
        clf()
    #plot(plot_xs, [tc_scores[i] for i in indices], '.')
    #plot(plot_xs, [pc_scores[i] for i in indices], 'r.-')
    return plot(plot_xs, [y_s[i] for i in indices], color, label=label)[0]

def plot_norms(norm_results, k=20, clear=True):
    if clear:
        clf()
    colors = {0.01:'b', 0.1:'g', 0.4:'r', 4.0:'k'}
    markers = {0.01:'+', 0.1:'x', 0.4:'o', 4.0:'.'}
    indices = mlab.find(np.array(norm_results['k']) == k)
    start, end = 0, 4
    plots = []
    for i in range(4):
        sig_words = np.array(norm_results['sig_words'])[indices][start:end]
        norms = np.array(norm_results['projector'])[indices][start:end]
        alpha = norm_results['a'][start]
        sig_topics = np.mean(norm_results['sig_topics'][start:end])
        plots.append(plot_scores(sig_words, norms, 
                                 color=colors[alpha] + '-' + markers[alpha],
                                 label=str(sig_topics),
                                 clear=False))
        start += 4
        end += 4
    legend(title='average sig topics', loc='best')
    return plots
    
def extract(flat_results, n):
    groups = group_flat_results(flat_results, ['n'])
    for group in groups:
        if group[0]['n'] == n:
            break
    subgroups = group_flat_results(group, ['l'])
    def combine_scores(subgroup):
        return {param:np.mean([entry[param] for entry in subgroup])
                     for param in subgroup[0].keys()}
    new_entries = []
    for subgroup in subgroups:
        new_entries.append(combine_scores(subgroup)) #list of dictionaries
    return {param:[entry[param] for entry in new_entries]
            for param in new_entries[0].keys()}

def plot_extract(flat_results, n_s=[1000, 1500, 2000, 2500], y='projector-20',
                 clear=True):
    if clear:
        clf()
    colors = {1000:'b', 1500:'g', 2000:'r', 2500:'k'}
    markers = {1000:'+', 1500:'x', 2000:'o', 2500:'.'}
    plots = []
    for n in n_s:
        results = extract(flat_results, n)
        indices = sorted(range(len(results['l'])), cmp=ind_cmp(results['l']))
        if y == 'ratio':
            y_s = np.array([results['projector-20'][i] for i in indices]) / \
                  np.array([results['ldaT-20'][i] for i in indices])
        else:
            y_s = [results[y][i] for i in indices]
        plots.append(plot(sorted(results['l']), y_s,
                          colors[n] + '-' + markers[n], 
                          label=str(n) + ' documents')[0])
    legend(loc='best')
    return plots
