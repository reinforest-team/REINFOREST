#!/bin/python3

import sys
import os
from itertools import combinations

non_zero_sim_count = 0
total = 0

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line
            
def get_grammar_from_file(f):
    filename = nb_lines.next()
    grammar = nb_lines.next()
    return grammar



def usage():
    print('sim_calculator.py [outputs_dir] [sim_file]')

def get_file_similarity(f1, f2):
    with open(f1) as inp1, open(f2) as inp2:
        nb_lines_1 = nonblank_lines(inp1)
        nb_lines_2 = nonblank_lines(inp2)

        fn1 = next(nb_lines_1)
        fn2 = next(nb_lines_2)

        g1 = next(nb_lines_1)
        g2 = next(nb_lines_2)

        if g1 != g2: return 0.0

        total = 0
        sim_count = 0
        for o1, o2 in zip(nb_lines_1, nb_lines_2):
            if o1.strip() == o2.strip(): sim_count += 1
            total += 1

        score = 0
        if total != 0:
            score = sim_count/total
        global non_zero_sim_count
        non_zero_sim_count += 1
        return score
            
def write_results(results, sim_file):
    with open(sim_file, 'a+') as outf:
        for r in results:
            outf.write(' '.join(r) + "\n")
    print('DUMPED {} results'.format(len(results)))

def make_sim_file(outputs_dir, sim_file):
    global total
    for root, dirs, files in os.walk(outputs_dir):
        results = []
        combos = None
        if len(files) > 0:
            combos = list(combinations(files,2))
        if not combos: continue
        for n1, n2 in combos:
            p1 = os.sep.join([root, n1])
            p2 = os.sep.join([root, n2])
            sim_score = get_file_similarity(p1, p2)
            res = (n1, n2, str(sim_score))
            total += 1
            if total % 10000 == 0: 
                print('{} combinations processed, {} nonzero'.format(total, non_zero_sim_count))
            results.append(res)
        write_results(results, sim_file)

if __name__ == '__main__':
    make_sim_file(sys.argv[1], sys.argv[2])
