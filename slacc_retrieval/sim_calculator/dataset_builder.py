#!/usr/bin/python3
import os
import sys
import random
import json

from pathlib import Path

NONZERO_SAMPLES = 0

def getSuff(language):
    return '.py' if language == 'python' else '.java'

def getPathFromFilename(srcDir, fname):
    if srcDir in fname: return fname
    splt = fname.split('_')
    return os.sep.join([srcDir] + splt[:3])
    
def getScores(inputFilepath):
    res = {} 
    with open(inputFilepath) as inpF:
        for line in inpF:
            f1, f2, score = line.split()
            res[f1+'|COMP|'+f2] = score
    return res

def get_c_p_from_name(name):
    splt = name.split('_')
    return splt[0], splt[1]

def getRandomSamples(language, outputsDir, c, p, isPos, numSamples):
    # isPos ==> should return a positive sample if true
    #           and a negative sample if false
    allSamples = []
    suff = getSuff(language)
    for root, dirs, files in os.walk(outputsDir):
        for f in files:
            if suff not in f: continue
            fc, fp = get_c_p_from_name(f)
            #print(f'fc: {fc}, fp: {fp}')
            posSample = (fc == c and fp == p)
            if isPos == posSample: 
                allSamples.append(f)
    if len(allSamples) == 0: return set()
    randomIndices = [ random.randint(0,len(allSamples)-1) for x in range(numSamples) ] 
    res = [ allSamples[x] for x in randomIndices ]
    st = set(res)
    return st


def getSamples(sourceLanguage, comparisonLanguage, outputsDir, scores):
    # for each file of source language type in outputs dir
    # pick 10 positive samples
    # pick 10 negative samples
    #output a list of [rootSampleFilepath, (positiveSamplePath, score), (negativeSamplePath, score)
    sourceSuf = getSuff(sourceLanguage)
    compSuf = getSuff(comparisonLanguage)
    numSamples = 10 
    sampleObjs = []

    count = 0
    for root, dirs, files in os.walk(outputsDir):
        for f in files:
            if sourceSuf not in f: continue
            c, p = get_c_p_from_name(f)
            posSamplePaths = getRandomSamples(comparisonLanguage, outputsDir, c, p, True, numSamples)
            negSamplePaths = getRandomSamples(comparisonLanguage, outputsDir, c, p, False, numSamples)
            sampleObjs.append({"base_name": f,  "positives": posSamplePaths, "negatives": negSamplePaths } )
            count += 1
            if count %500 == 0: print(f"processed: {count} files")
    return sampleObjs

def getTextFromFile(fpath):
    Path(fpath).read_text()

def getScore(f1, f2, scores):
    if os.sep in f1 or os.sep in f2:
        return -1
    if f1+"|COMP|"+f2 in scores:
        score = scores[f1+"|COMP|"+f2]
        if score != 0:
            global NONZERO_SAMPLES
            NONZERO_SAMPLES += 1
        return score
    else:
        return 0

def addCodeToSamples(samples, srcDir, scores):
    #samples are of the form {fname: {positives: [fnames,], negatives: [fnames]}}
    res = []
    count = 0
    for s in samples:
        basePath = getPathFromFilename(srcDir, s['base_name'])
        baseText = Path(basePath).read_text()

        positivePaths = [ getPathFromFilename(srcDir, x) for x in s['positives'] ]
        positiveScores = [ getScore(s['base_name'], p, scores) for p in s['positives'] ]

        negativePaths = [ getPathFromFilename(srcDir, x) for x in s['negatives'] ]
        negativeScores = [ getScore(s['base_name'], p, scores) for p in s['negatives'] ]

        positiveTexts = [ Path(x).read_text() for x in positivePaths ] 
        negativeTexts = [ Path(x).read_text() for x in negativePaths ]
        finalObj = {
            'base_name': s['base_name'],
            'code': baseText,
            'positives': [ {"code": c, "score": float(sc), "path": p} for c,sc,p in zip(positiveTexts, positiveScores, positivePaths) ],
            'negatives': [ {"code": c, "score": float(sc), "path": p} for c,sc,p in zip(negativeTexts, negativeScores, negativePaths) ]
        }
        res.append(finalObj)
        if count % 500 == 0:
            print(f"Added code to {count} samples")
            count += 1
    return res    

def getRandomFillerSamples(srcDir, numSamples, basePath, isPos, suffix):
    c, p = get_c_p_from_name(basePath)
    dirs = os.listdir(srcDir)
    samples = [] 
    while len(samples) < numSamples:
        if not isPos:
            ind = random.randint(0, len(dirs)-1)
            while dirs[ind] == c:
                ind = random.randint(0, len(dirs)-1)
        else:
            ind = dirs.index(c)

        pDirs = os.listdir(os.sep.join([srcDir, dirs[ind]]))
        if not isPos:
            pind = random.randint(0, len(pDirs)-1)
        else:
            pind = pDirs.index(p)
        
        fnames = os.listdir(os.sep.join([srcDir,dirs[ind], pDirs[pind]]))
        fInd = random.randint(0, len(fnames)-1)
        samples.append(os.sep.join([srcDir, dirs[ind], pDirs[pind], fnames[fInd]]))
    return samples

def addFillerSamples(objs, srcDir, language, numSamples=10):
    '''if num samples in the obj positives or negatives list is less than num samples augment with random samples from srcdir'''
    suffix = getSuff(language)
    for i in range(len(objs)):
        numPosSamples = len(objs[i]['positives'])
        numNegSamples = len(objs[i]['negatives'])
        suffix = '.py'
        if numPosSamples < numSamples:
            newPosSamples = getRandomFillerSamples(srcDir, numSamples - len(objs[i]['positives']), objs[i]['base_name'], True, suffix )
            map(lambda x: objs[i]['positives'].add(x), newPosSamples)
        if numNegSamples < numSamples:
            newNegSamples = getRandomFillerSamples(srcDir, numSamples - len(objs[i]['negatives']), objs[i]['base_name'], False, suffix )
            map(lambda x: objs[i]['negatives'].add(x), newPosSamples)

def writeObjectsToFile(objects, fpath):
    with open(fpath,'w+') as out:
        json.dump(objects, out, indent=4, sort_keys=True)

def usage():
    print("./dataset_builder.py <base_language> <scores_filepath> <output_filepath> <src_dir> <outputs_dir>")

if __name__ == '__main__':
    if len(sys.argv) < 6:
        usage()
        sys.exit(1)

    baseLanguage = sys.argv[1]
    scoresFilepath = sys.argv[2]
    outputFilepath = sys.argv[3]
    srcDir = sys.argv[4]
    outputsDir = sys.argv[5]
    loaded_scores = getScores(scoresFilepath)
    print("loaded_scores")
    comparisonLanguage = "java" if baseLanguage == "python" else "python"
    print("getting samples")
    samples = getSamples(baseLanguage, comparisonLanguage, outputsDir, scoresFilepath) 
    print("filling out samples")
    addFillerSamples(samples, srcDir, comparisonLanguage)
    print("adding code to samples")
    codedSamples = addCodeToSamples(samples, srcDir, loaded_scores)
    print("Found {} nonzero samples".format(NONZERO_SAMPLES))
    print("----")
    #print("writing to file [{}]".format(outputFilepath))
    #writeObjectsToFile(codedSamples,outputFilepath)

