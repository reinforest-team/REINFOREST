#!/usr/bin/python
import sys
import os
import random

def generate_int():
    return random.randint(0,100000)

def generate_long():
    return random.randint(100000, 10000000000)

def generate_float():
    return float(random.uniform(0,100000))

def generate_double():
    return generate_float()

def generate_string():
    l = random.randint(0,30)
    s = ''.join([ chr(random.randint(33,126)) for x in range(l) ])
    return s

def get_generator_function(t):
    if t == 'int': return generate_int
    elif t == 'long': return generate_long
    elif t == 'float': return generate_float
    elif t == 'double': return generate_double
    elif t == 'string': return generate_string
    else: 
        print("WARNING: no generator for type [{}]".format(t))
        return generate_int
    
def generate_input_from_grammar(grammar):
    res = []
    gen_funcs = map(lambda x: get_generator_function(x), grammar.split())
    return [ g() for g in gen_funcs ]

def generate_inputs_from_grammar(grammar, count):
    inps = []
    c = 0
    while c < count:
       inps.append(generate_input_from_grammar(grammar))
       c += 1
    return inps

def write_inputs(inputs, output_dir, g):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.sep.join([output_dir,'_'.join(g.split())]), 'w+') as outfile:
        for inp in inputs:
            strinps = [ str(x) for x in inp ]
            outfile.write(' '.join(strinps) + '\n')

def get_grammars_from_file(grammar_file):
    res = set()
    with open(grammar_file) as gf:
        for line in gf:
            tmp = line.split(',')
            gs = tmp[1].split('||')
            for g in gs:
                res.add(g)
    return res

def usage():
    print('input_generator.py [code_directory] [count] [output_directory]')

def get_all_grammars(problem_dir):
    res = set()
    for root, dirs, files in os.walk(problem_dir):
        for f in files:
            if 'gram.txt' == f:
                grams = get_grammars_from_file(os.path.join(root, f))
                res = res.union(grams)
    return res

def generate_inputs(code_dir, count, output_dir):
    grammars = get_all_grammars(code_dir)
    for g in grammars:
        inps = generate_inputs_from_grammar(g, count)
        try:
            write_inputs(inps, output_dir, g)
        except Exception as e:
            print('could not write to file [{}] for grammar [{}]'.format(str(output_dir+'/'+'_'.join(g.split())), g))

if __name__ == '__main__':
    if len(sys.argv) < 4: usage()
    generate_inputs(sys.argv[1], int(sys.argv[2]), sys.argv[3])
