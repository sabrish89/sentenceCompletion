'''
1. Import dataset .txt
2. Append datasets .txt + .txt ....
'''

import re
from langdetect import detect
from roman import fromRoman
from re import sub
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from math import ceil
from time import perf_counter
from deepdiff import DeepDiff


def importData(filename, pattern = "\*\*\*\s"):
    '''
    Starts read at first pattern + 1,
    Ends read at next pattern - 1
    1. Import dataset (.txt) formats
    TODO: allow other formats
    2. Lowercase the text
    3. Return a list of sentences
    Ex:
    print(importData(".\\corpus\\edgar allen poe\\theFallOfTheHouseOfUsher.txt"))
    '''
    data = []
    read = False
    try:
        f = open(filename, "r")
        for line in f:
            if re.match(pattern, line):
                read = not read
                if not read:
                    break
                continue
            if read and not line.isspace():
                data.append(''.join([word.lower() if word not in ['I'] else word for word in line]))
        return data
    except Exception as e:
        return None

def processData(sentList, time = False):
    '''
    1. Clean list
    2. Combine list
    3. Return a string of data
    TODO: Translate roman numerals to english numbers ✔
    TODO: Language detection - only english ✔
    TODO: Currently adding period & semi-colon to punct removal; research ways to incorporate discontinuity
    '''

    def procPunct(sentence):
        '''
        Remove punctuations and special rules
        '''
        return sub("(--|-)"," ",''.join([sub("(\n|\t|\r|\v|\f|,|\"|'|\*|\(|\)|\?|\.|;|!|:)", "", word) for word in sentence]))

    def process(sentence):
        '''
        Macro holds all processing steps
        '''
        try:
            if detectEnglish(sentence):
                sentence = procPunct(sentence)
                return sentence
            else:
                return None
        except Exception as e:
            return None

    def detectEnglish(sentence):
        '''
        Uses langdetect: detect
        '''
        return detect(sentence) == "en"

    def romanToLatin_num(str):
        '''
        Takes an input str in Roman numerals and returns Latin number
        '''
        return fromRoman(str)

    def flattenAndCompress(sentenceList):
        '''
        Remove spaces and None
        '''
        return [sentence.strip().replace("  "," ") for sentence in sentenceList if sentence]

    timeS = perf_counter()
    for idx, sentence in enumerate(sentList): #idx first
        sentList[idx] = process(sentence)

    sentList = flattenAndCompress(sentList)
    if time:
        return sentList, ceil(perf_counter() - timeS)
    else:
        return sentList

def parallelize(sentList, perSize = 200, time = False):
    '''
    Multi-processing port
    '''

    timeS = perf_counter()
    with ThreadPoolExecutor() as executor:
        result = executor.map(processData, [sentList[i*perSize:(i+1)*perSize] for i in range(ceil(sentList.__len__()/perSize))])
        fOutput = [str for res in result for str in res]

    if time:
        return fOutput, perf_counter() - timeS
    else:
        return fOutput

if __name__ == '__main__':
    sentList = importData("./corpus/edgar allen poe/theFallOfTheHouseOfUsher.txt")
    sentStringListP, timeP = parallelize(sentList, time=True)
    sentStringListL, timeL = processData(sentList, time=True)
    #checkDiff = DeepDiff(sentList,[SubL for List in [sentList[i * 300:(i + 1) * 300] for i in range(3)] for SubL in List])
    checkDiff = DeepDiff(sentStringListP,sentStringListL)
    if not checkDiff:
        print(f"multiprocess took {timeL - timeP} seconds less")