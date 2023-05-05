#!/bin/python3
import sys
import os

def get_grammar_from_lines(filepath, lines):
    grammar = []
    for line in lines:
        if 'nextInt' in line:
            grammar.append('int')
        elif 'nextDouble' in line:
            grammar.append('double')
        elif 'nextFloat' in line:
            grammar.append('float')
        elif 'nextLong' in line:
            grammar.append('long')
        elif 'next()' in line and 'Long' in line:
            grammar.append('long')
        elif 'next()' in line and 'Int' in line:
            grammar.append('int')
        elif 'String' in line:
            grammar.append('string')
        else:
            print(str(filepath) + "-"+ str(line))
            return None
    return grammar

def get_grammar_from_file(filepath):
    grammar_definition = []
    if '.py' in filepath:
        return None
    if '.java' in filepath:
        with open(filepath) as infile:
            for line in infile:
                if '.next' in line:
                    grammar_definition.append(line)
    gram = get_grammar_from_lines(filepath, grammar_definition)
    #print(' '.join(gram))
    if gram: return ' '.join(gram)
    else: return None
    #print(gram)
    #return ' '.join(gram)

def extract_grammar_for_directory(dirpath):
    dir_list = os.listdir(dirpath)
    grams = {}
    total_grams = set()
    for f in dir_list:
        g = get_grammar_from_file(dirpath+"/"+f)
        if g: 
            grams[f] = g
            if g not in total_grams: total_grams.add(g)
        
    res = {}
    for f in dir_list:
        if f in grams:
          res[f] = grams[f] 
        else:
          res[f] = '||'.join(total_grams)
    return res

def write_grammars_to_file(grams, output_file):
    with open(output_file, 'w+') as outf:
        for doc in grams:
            outf.write(doc + ", " + grams[doc]+'\n')

def usage():
    print('grammar_extractor.py [code_dir] [output_file]')

if __name__ == '__main__':
    if len(sys.argv) < 2: usage()
    else: 
        grams = extract_grammar_for_directory(sys.argv[1])
        write_grammars_to_file(grams, sys.argv[2])

