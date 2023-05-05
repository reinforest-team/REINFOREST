#!/bin/python3

import sys
import os
import subprocess
from multiprocessing import Pool, Value


successful = 0
unsuccessful = 0
multi_suc = Value('i', 0)
multi_unsuc = Value('i', 0)

all_grams = []

def run_proc(filepath, inp, cmd):
    try:
        output = subprocess.check_output([cmd, filepath], 
            input=bytes(inp, 'utf-8'), stderr=subprocess.STDOUT, timeout=1)
        return output.decode('utf-8')
    except subprocess.CalledProcessError as ex:
        output = ex.output
        #print("Could not run {} with input {} because \n {}".format(filepath, inp, output))
        return ''
    except subprocess.TimeoutExpired as ex:
        output = ex.output
        #print("{} timed out with input {}".format(filepath, inp))
        return ''

def get_inputs_path(g, inputs_dir):
    input_filename = '_'.join(g.split())
    input_filepath = os.path.join(inputs_dir, input_filename)
    if os.path.exists(input_filepath):
        return input_filepath
    else:
        print("ERROR: Could not find inputs file for grammar: {}".format(g))
        sys.exit(1)

def run_file(filepath, possible_grammars, inputs_dir):
    runner = None
    outputs = []
    if '.py' in filepath:
        cmd = 'python' 
    else: 
        cmd = 'java'
    grammar = None
    for g in possible_grammars:
        if grammar: break
        inputs_file = get_inputs_path(g, inputs_dir)
        with open(inputs_file) as inf:
            #print("running {} with grammar {}".format(filepath, g))
            for line in inf:
                output = run_proc(filepath, line, cmd)
                if output == '': break
                else: 
                    outputs.append(output)
                    grammar = g
    return grammar, outputs

                

def parse_gram_file(filepath):
    res = {}
    with open(filepath) as f:
        for line in f:
            splt = line.split(',')
            res[splt[0]] = splt[1].split('||')
    return res

def save_outputs(filepath, grammar, outputs, output_dir):
    global successful
    splt = filepath.split('/')
    cohort = list(filter(lambda x: 'C' in x, splt))[0]
    problem = list(filter(lambda x: 'P' in x, splt))[0]
    h = splt[-1]
    output_filename = '_'.join([cohort, problem, h, '.output'])
    output_dir = os.path.join(output_dir, '_'.join(grammar.split()))
    output_filepath = os.path.join(output_dir, output_filename)
    if not os.path.exists(output_dir):
       os.makedirs(output_dir) 
    with open(output_filepath, 'w+') as outf:
        #print('saving outputs to: {}'.format(output_filename))
        outf.write(filepath + '\n\n')
        outf.write(grammar + '\n\n')
        try:
            outf.write('\n'.join(outputs))
            successful += 1
        except Exception as ex:
            print("could not write outputs for file {}".format(output_filepath))
    
def print_status():
    suc = multi_suc.value
    unsuc = multi_unsuc.value
    try: 
        print("Success: {} Failed: {} Total: {} ({}% success rate)".format(suc, 
            unsuc, 
            (suc+unsuc), 
            round(suc/(suc+unsuc)*100, 2)))
    except Exception as e:
        print('status unavailable')
        print(e)
    
def inc_suc():
    with multi_suc.get_lock():
        multi_suc.value += 1

def inc_unsuc():
    with multi_unsuc.get_lock():
        multi_unsuc.value += 1

def process_and_save_file(root, f, grammars, inputs_dir, output_dir):
    global unsuccessful
    if '.py' in f or '.java' in f:
        filepath = os.path.join(root,f)
        if f not in grammars: 
            print("WARNING no grammar found for {}".format(filepath))
            inc_unsuc()
            return
        gs = grammars[f]  
        try:
            g, outputs = run_file(filepath, gs, inputs_dir)
            if not g or len(outputs) == 0: 
                #print("WARNING: Could not find grammar for {}".format(filepath))
                inc_unsuc()
            else:
                inc_suc()
                save_outputs(filepath, g, outputs, output_dir)
                
        except Exception as e:
            print("Unknown exception running and saving file: {} with grammars {}".format(str(filepath), str(gs)))
        print_status()

def process_and_save(code_dir, inputs_dir, output_dir):
    global unsuccessful
    global all_grams 
    tmp = os.listdir(inputs_dir)
    all_grams = [ x.replace('_', ' ') for x in tmp ]
    for root, dirs, files in os.walk(code_dir):
        if 'gram.txt' in files:
            grammars = parse_gram_file(os.path.join(root, 'gram.txt'))

            for f in files:
                try: 
                    print("Success: {} Failed: {} Total: {} ({}% success rate)".format(successful, 
                        unsuccessful, 
                        (successful+unsuccessful), 
                        round(successful/(successful+unsuccessful)*100, 2)))
                except Exception as e:
                    print('status unavailable')
                if '.py' in f or '.java' in f:
                    filepath = os.path.join(root,f)
                    if f not in grammars: 
                        print("WARNING no grammar found for {}".format(filepath))
                        unsuccessful += 1
                        continue
                    gs = grammars[f]  
                    try:
                        g, outputs = run_file(filepath, gs, inputs_dir)
                        if not g or len(outputs) == 0: 
                            unsuccessful += 1
                            continue
                        save_outputs(filepath, g, outputs, output_dir)
                    except Exception as e:
                        print("Unknown exception running and saving file: {} with grammars {}".format(str(filepath), str(gs)))
                        print(e)
    print("---")

def process_and_save_multi(code_dir, inputs_dir, output_dir):
    for root, dirs, files in os.walk(code_dir):
        if 'gram.txt' in files:
            grammars = parse_gram_file(os.path.join(root, 'gram.txt'))
            args = [ [root, f, grammars, inputs_dir, output_dir] for f in files]
                
            with Pool(20) as p:
                p.starmap(process_and_save_file, args)
            print("--- finished {}".format(root))

def usage():
    print('python runner.py [code_dir] [inputs_dir] [output_dir]')

if __name__ == '__main__':
    process_and_save_multi(sys.argv[1], sys.argv[2], sys.argv[3])
