#!/usr/bin/python3
'''
    1) make a list of list of problems with at least 1 runnable solution in the source language
    2) Divide that into a set of test, val, train problems
    3) For each runnable sample in the set get n positive samples and n negative samples 
    4) Positive samples are created by checking for runnable samples in the same problem directory first and then 
        if not runnable taking another sample in the directory
        negative samples are taken from runnable programs in different directories
    5) once samples are chosen then create objects according to the format 

        { base_sample_name: "CX_PY_aaaabbbbsssbbb.(java|py)",
          base_sample_code: "import * .... "
          positives: [
            {
                comparison_sample_name: "CX_PY_aaabbbb.(java|py)",
                comparison_sample_code: "import * ..... ",
                score: [0-1] if runnable else -1
            }
         ]
         negatives: [...]
        }

     A Functional Description for fun!
     
     STATE VARIABLES:
     input params: [outputsDir, scoresFile, codeDir, sourceLanguage, outputPath]
     allRunnableSamples = getAllSamples(outputsDir)
     scores = loadScores(scores_file)
     runnableProblems = getProblems(runnableSamples)
     trainProblems, valProbs, testProbs = divide801010(runnableProblems)
     #########################
     
     HELPER FUNCTIONS:
     divide801010(runnableProblems:str[]) => [[tP], [vP], [tstP]]
     write(jsonObj, filepath) => void 
     getAllSamples(outputsDir)
     getProblems(samples)
     makeBaseSampleObjects(probs)
     getCodeFromName(sampleName)

     
     ##########

     write(makeFinalObj(trainProbs, valProbs, testProbs, sourceLang, outputsDir, scores, codeDir), outputPath)

     makeFinalObj(trainProbs, valProbs, testProbs, sourceLang, outputsDir, scores, codeDir) => {
        trainProbs: trainProbs:string[],
        valProbs: valProbs:string[],
        testProbs: testProbs:string[],
        trainDataset: makeBaseSampleObjects(trainProbs):BaseSampleObject[]
        valDataset: makeBaseSampleObjects(valProbs):BaseSampleObject[]
        testDataset: makeTestSampleObjects(valProbs):BaseSampleObject[]
     }

             makeBaseSampleObjects(probs) = map(makeBaseSampleObject, getProbsRunnableSamples(probs))
             makeBaseSampleObject(sampleName) => {
                base_sample_name: sampleName,
                base_sample_code: getCodeFromName(sampleName)
                positives: map(makeComparisonObject, getNPositiveNameScores(sampleName, N))
                negatives: map(makeComparisonObject, getNNegativeNameScores(sampleName, N))
                
                getNPositiveNameScores(sampleName, N) = ...(scores, runnableSamples, codeDir, outputsDir) => (compSampleName, score)
                getNNegativeNameScores(sampleName, N) = ...(scores, runnableSamples, codeDir, outputsDir) => (compSampleName, score)
             }



             makeComparisonObject(compSampleName, score) => {
                comparison_sample_name: compSampleName,
                code: getCodeFromNSame(compSampleName),
                score: score
             }

'''
#### 
import os
import sys
import math
import random
import json
from pathlib import Path

#### STATE VARIABLES ###
OUTPUTS_DIR = SCORES_FILEPATH = CODE_DIR = SOURCE_LANGUAGE = COMP_LANGUAGE = OUTPUT_PATH = ""
ALL_RUNNABLE_SAMPLES = SCORES = RUNNABLE_PROBLEMS = []
TRAIN_PROBLEMS = VAL_PROBLEMS = TEST_PROBLEMS = []
TOTAL_COMPARISONS = TOTAL_SCORES = 0

RUNNABLE_SAMPLE_INPUT_TYPES = {}

#################
# helper functions
def getRunnableProblems(samples):
    pythonProblems = set() 
    javaProblems = set()
    for x in samples:
        splt = x.split('_')
        s = pythonProblems if '.py' in x else javaProblems
        s.add('_'.join(splt[:2]))
    return pythonProblems.intersection(javaProblems)

def getSamples(dirPath):
    samples = []
    for root, dirs, files in os.walk(dirPath):
        sampleList = [ f for f in files ]

        sampleList = [ '_'.join(f.split('_')[:-1]) if '.output' in f else f for f in sampleList ]
        samples += sampleList
    return samples  

def getInputTypes(dirPath):
    res = {}
    for root, dirs, _ in os.walk(dirPath):
        for d in dirs:
            files = os.listdir(os.sep.join([root, d]))
            sampleList = [ '_'.join(f.split('_')[:-1]) if '.output' in f else f for f in files ]
            for f in sampleList:
                res[f] = d
    return res

def getTrainValTestProblems(runnableProblems):
    total = len(runnableProblems)
    numTrain = math.floor(total*.8) 
    numVal = numTest = math.ceil(total*.1)
    
    trainProbs = valProbs = testProbs = []
    cpy = runnableProblems

    trainProbs = set(random.sample(list(cpy), k=numTrain))
    cpy = cpy.difference(trainProbs)

    valProbs = set(random.sample(list(cpy), k=numVal))
    cpy = cpy.difference(valProbs)

    testProbs = cpy
    
    return list(trainProbs), list(valProbs), list(testProbs)

def write(filepath, results):
    json.dump(results, open(filepath, 'w+'), indent=4, sort_keys=True)

def getCodeFromSampleName(sampleName):
    pathName = os.sep.join([CODE_DIR, *sampleName.split('_')])
    return Path(pathName).read_text() 
    
###################
def getRunnableSamples(problem):
    tmp = list(filter(lambda x: problem in x, ALL_RUNNABLE_SAMPLES))
    return tmp

def getCompSamples(sampleNames):
    filString = ".py" if COMP_LANGUAGE == "python" else ".java" 
    return list(filter(lambda x: filString in x, sampleNames))

def getBaseSamples(sampleNames):
    filString = ".py" if SOURCE_LANGUAGE == "python" else ".java" 
    return list(filter(lambda x: filString in x, sampleNames))

def makeComparisonObject(compSampleScore):
    sampleName, score = compSampleScore
    return {
        "comparison_sample_name": sampleName,
        "score": score,
        "code": getCodeFromSampleName(sampleName)
    }

def getNonRunnablePositiveSamples(existingSamples, baseSampleName, N):
    problemBase = '_'.join(baseSampleName.split('_')[:2])
    problemPath = os.sep.join(problemBase.split('_'))
    allSamples = os.listdir(os.path.sep.join([CODE_DIR, problemPath]))
    allSamples = [ '_'.join([problemBase, x]) for x in allSamples ]
    compSamples = getCompSamples(allSamples)
    if len(compSamples) < N: 
        print("Warning: Could not find enough non runnable positives")
        return compSamples
    newSamples = []
    while len(newSamples) < N:
        nextSample = random.sample(compSamples, 1)[0]
        if nextSample not in existingSamples:
            newSamples.append(nextSample)

    return newSamples
    
def getScore(sample1, sample2):
    global TOTAL_SCORES, TOTAL_COMPARISONS
    TOTAL_COMPARISONS += 1
    isS1Runnable = sample1 in ALL_RUNNABLE_SAMPLES
    isS2Runnable = sample2 in ALL_RUNNABLE_SAMPLES
    if not isS1Runnable or not isS2Runnable: return -1
    
    key = f'{sample1}|COMP|{sample2}' 
    if key in SCORES:
        TOTAL_SCORES += 1
        return SCORES[key]
    if isS1Runnable and isS2Runnable: return 0.0

def getCompTypeSamples(baseSampleName, runnableCompSamples):
    t = RUNNABLE_SAMPLE_INPUT_TYPES[baseSampleName]
    res = filter(lambda x: RUNNABLE_SAMPLE_INPUT_TYPES[x] == t, runnableCompSamples)
    return list(res)

def getNPositiveNameScores(baseSampleName, N=10):
    runnableSamples = getRunnableSamples('_'.join(baseSampleName.split('_')[:2]))
    runnableCompSamples = getCompSamples(runnableSamples)
    runnableCompTypeSamples = getCompTypeSamples(baseSampleName, runnableCompSamples)
    sampleNames = []
    if len(runnableCompTypeSamples) >= N:
       sampleNames = random.sample(runnableCompTypeSamples, k=N) # add same input type first
    else: 
        sampleNames = runnableCompTypeSamples
        unsampledRunnableComp = list(set(runnableCompSamples) - set(sampleNames)) 
        if len(unsampledRunnableComp) >= N - len(sampleNames):
            sampleNames += random.sample(unsampledRunnableComp, k=(N - len(sampleNames))) # then add runnable with different input type
        else:
            sampleNames += unsampledRunnableComp
            nonRunnablePositives = getNonRunnablePositiveSamples(sampleNames, baseSampleName, N-len(sampleNames))# then add nonrunnable positives
            sampleNames += nonRunnablePositives
    scores = [ getScore(baseSampleName, s) for s in sampleNames ]
    return zip(sampleNames, scores)


def getNNegativeNameScores(baseSampleName, N=10):
    problem = baseSampleName.split('_')[:2]
    sampleNames = []
    
    runnableCompSamples = getCompSamples(ALL_RUNNABLE_SAMPLES)
    negSamples = random.sample(runnableCompSamples, k=N)

    # prob a better way to do this but w.e.
    flag = True
    while flag:
        flag = False
        for ind, x in enumerate(negSamples):
            pBase = '_'.join(x.split('_')[:2])
            if pBase == problem:
                flag = True
                negSamples[ind] = random.sample(ALL_RUNNABLE_SAMPLES, k=N)
    scores = [ getScore(baseSampleName, s) for s in negSamples ]
    return zip(negSamples, scores)


def makeBaseSampleObject(sampleName, runnableSamples):
    trimmedSampleName = '_'.join(sampleName.split('_')[:-1])

    return {
                "base_sample_name": trimmedSampleName,
                "base_sample_code": getCodeFromSampleName(sampleName),
                "positives": list(map(makeComparisonObject, getNPositiveNameScores(sampleName))),
                "negatives": list(map(makeComparisonObject, getNNegativeNameScores(sampleName)))
    }

def makeBaseSampleObjects(problem):
    runnableSamples = getRunnableSamples(problem)
    baseSamples = getBaseSamples(runnableSamples)
    return [ makeBaseSampleObject(bs, runnableSamples) for bs in baseSamples ]

def makeAllBaseSampleObjects(problems):
    return [ tmp for p in problems for tmp in makeBaseSampleObjects(p) ]

def makeFinalObj():
    trainBaseObjects, valBaseObjects, testBaseObjects = [ makeAllBaseSampleObjects(x) for x in [TRAIN_PROBLEMS, VAL_PROBLEMS, TEST_PROBLEMS] ]
    return {
        "source_language": SOURCE_LANGUAGE,
        "train_problems": TRAIN_PROBLEMS,
        "val_problems": VAL_PROBLEMS,
        "test_problems": TEST_PROBLEMS,
        
        "train_data": trainBaseObjects,
        "val_data": valBaseObjects,
        "test_data": testBaseObjects
    }

def loadScores(scoresPath):
    res = {} 
    with open(scoresPath) as inpF:
        for line in inpF:
            f1, f2, score = line.split()
            f1 = '_'.join(f1.split('_')[:-1])
            f2 = '_'.join(f2.split('_')[:-1])
            res[f1+'|COMP|'+f2] = float(score)
    return res

def setGlobals(oDir, sPath, cDir, sLang, oPath):
    global OUTPUTS_DIR, SCORES_FILEPATH, CODE_DIR, SOURCE_LANGUAGE, COMP_LANGUAGE, OUTPUT_PATH
    global ALL_RUNNABLE_SAMPLES, SCORES, RUNNABLE_PROBLEMS
    global TRAIN_PROBLEMS, VAL_PROBLEMS, TEST_PROBLEMS
    global RUNNABLE_SAMPLE_INPUT_TYPES

    OUTPUTS_DIR = oDir
    SCORES_FILEPATH = sPath
    CODE_DIR = cDir
    SOURCE_LANGUAGE = sLang
    OUTPUT_PATH = oPath 
    COMP_LANGUAGE = "java" if SOURCE_LANGUAGE == 'python' else 'python'

    print("processing scores")
    SCORES = loadScores(SCORES_FILEPATH)
    print("collecting runnable samples")
    ALL_RUNNABLE_SAMPLES = getSamples(OUTPUTS_DIR)

    print("collecting runnable sample inputTypes")
    RUNNABLE_SAMPLE_INPUT_TYPES = getInputTypes(OUTPUTS_DIR)
    
    print("collecting runnable problems")
    RUNNABLE_PROBLEMS = getRunnableProblems(ALL_RUNNABLE_SAMPLES)

    TRAIN_PROBLEMS, VAL_PROBLEMS, TEST_PROBLEMS = getTrainValTestProblems(RUNNABLE_PROBLEMS)


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: ./dataset_builder_2.py [outputsDir] [scoresFile] [codeDir] [sourceLanguage] [outputPath]')
        sys.exit(1)

    outputsDir = sys.argv[1]
    scoresFile = sys.argv[2]
    codeDir = sys.argv[3]
    sourceLanguage = sys.argv[4]
    outputPath = sys.argv[5]

    setGlobals(outputsDir, scoresFile, codeDir, sourceLanguage, outputPath)
        
    print(f'scores length: {len(SCORES)} numRunnableSamples: {len(ALL_RUNNABLE_SAMPLES)} numRunnableProblems: {len(RUNNABLE_PROBLEMS)}')

    print(f'len trainProps: {len(TRAIN_PROBLEMS)} numValProbs: {len(VAL_PROBLEMS)} numTestProbs: {len(TEST_PROBLEMS)}')


    results = makeFinalObj()

    print(f'outputting results')
    write(outputPath, results)

    print(f'total scores: {TOTAL_SCORES} total comparisons: {TOTAL_COMPARISONS}')




