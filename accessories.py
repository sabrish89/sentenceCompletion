'''
1. Import dataset .txt
2. Append datasets .txt + .txt ....
'''

import re
from langdetect import detect
from roman import fromRoman
from re import sub
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, ALL_COMPLETED
from math import ceil
from time import perf_counter
from numpy import array
from tqdm import tqdm
import numpy as np
from string import punctuation, digits
from random import random

def importData(filename, pattern = "(\*\*\*)+.*", threshold = 5):
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
            if re.search(pattern, line):
                read = not read
                if not read:
                    break
                continue
            if read and not line.isspace() and line.__len__() > threshold:
                data.append(''.join([word.lower() if word not in ['I'] else word for word in line]))
        return data
    except Exception as e:
        return None

def processData(sentList, time = False):
    '''
    Used for word sequences
    1. Clean list
    2. Combine list
    3. Return a string of data
    TODO: Translate roman numerals to english numbers ✔
    TODO: Language detection - only english ✖ - probabilistic so moved to post
    TODO: Currently adding period & semi-colon to punct removal; research ways to incorporate discontinuity
    '''

    def procPunct(sentence):
        '''
        Remove punctuations and special rules
        NOTE: Replace below with more efficient string class operations
        #return sub("(--|-)",", ",''.join([sub("(\n|\t|\r|\v|\f|\*|\(|\)|,|[:punct:])", "", word) for word in sentence]))
        '''
        return sentence.translate(str.maketrans('', '', punctuation + digits))

    def process(sentence):
        '''
        Macro holds all processing steps
        '''

        try:
            if detectEnglish(sentence): #Language Detection <here> is probabilistic # Move to post processing
                sentence = procPunct(prelimPass(sentence))
                return sentence
            else:
                return None
        except Exception as e:
            return None

    def prelimPass(sentence):
        '''
        Some issues like although--this => althoughthis
        Need to sub hypen with space
        '''
        return sub("(--|-)",", ",''.join([sub("(\n|\t|\r|\v|\f|\*|\(|\)|,)", "", word) for word in sentence]))

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

    def shaveArticles(sentence):
        '''
        The model learns only articles 'the', 'a'
        NOTE: Instead shave datapoints with Y in articles
        '''
        return ' '.join([word for word in sentence.split() if word not in ['a', 'an', 'the']])

    def flattenAndCompress(sentenceList):
        '''
        Remove spaces and None
        '''
        return [sentence.strip().replace("  "," ") for sentence in sentenceList if sentence]

    def createWordList(sentList):
        '''
        Make a sentence into a wordlist
        '''
        wordList = []
        for sentence in sentList:
            wordList.extend(sentence.split())
        return wordList

    timeS = perf_counter()
    for idx, sentence in enumerate(sentList): #idx first
        sentList[idx] = process(sentence)

    sentList = flattenAndCompress(sentList)

    if time:
        return createWordList(sentList), ceil(perf_counter() - timeS)
    else:
        return createWordList(sentList)

def parallelizeTextProcessing(sentList, perSize = 200, time = False):
    '''
    Multi-processing port

    #BULKY CODE ------------------------------------------------------------------------------------
    executor = ProcessPoolExecutor(ceil(sentList.__len__()/perSize))
    worklist = [sentList[i*perSize:(i+1)*perSize] for i in range(ceil(sentList.__len__()/perSize))]
    rFutures = [executor.submit(processData, chunk) for chunk in worklist]
    wait(rFutures, timeout=200, return_when=ALL_COMPLETED)
    if time:
        return [str for res in rFutures for str in res.result()], perf_counter() - timeS
    else:
        return [str for res in rFutures for str in res.result()]
    #----------------------------------------------------------------------------------------------

    '''

    timeS = perf_counter()
    with ProcessPoolExecutor(ceil(sentList.__len__()/perSize)) as executor:
        result = executor.map(processData, [sentList[i * perSize:(i + 1) * perSize] for i in range(ceil(sentList.__len__() / perSize))])
    if time:
        return [str for res in result for str in res], perf_counter() - timeS
    else:
        return [str for res in result for str in res]

def genSeq(wordList, vocab, subSize=7):
    '''
    Return a nested list of word sequences of size seqSize = 7
    NOTE: I shave articles here as well
    '''

    X = []
    Y = []
    for idx in tqdm(range(subSize, wordList.__len__())):
        if wordList[idx] in ['a', 'an', 'the', 'of', 'and']:
            if random() < 0.95:
                continue
        X_ = [[0]*vocab.__len__() for _ in range(subSize)]
        Y_ = [0]*vocab.__len__()
        for idxx, word in enumerate(wordList[idx - subSize: subSize]):
            X_[idxx][vocab[word]] = 1
        Y_[vocab[wordList[idx]]] = 1
        X.append(X_)
        Y.append(Y_)
    return array(X), array(Y)

def genSent(model, seed, vocab, size=5, textIterations = 10):
    def parse(X, vocab, size):
        X_ = [[0] * vocab.__len__() for _ in range(size)]
        for idxx, word in enumerate(X[: size]):
            X_[idxx][vocab[word]] = 1
        return np.array([X_])
    revDict = {vocab[key]:key for key in vocab.keys()}
    print('Seed Text: ',' '.join(seed),".....\n")
    for idxx in range(textIterations):
        X = seed[idxx: idxx+size]
        X = parse(X,vocab,size)
        Y = model.predict(X)[0]
        nIdx = sample(preds=Y, temperature=0.34)
        seed += revDict[nIdx]
    print('Gen Text:', ' '.join(seed))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def processAlphaData(sentList):
    '''
    Processing for alphabet sequence prediction
    '''
    def procPunct(sentence):
        '''
        Remove punctuations and special rules
        '''
        return sentence.translate(str.maketrans('', '', punctuation + digits + "\t\n\r\v\f"))

    def detectEnglish(sentence):
        '''
        Uses langdetect: detect
        '''
        return detect(sentence) == "en"

    data = []
    for idx, sentence in enumerate(sentList):
        if detectEnglish(sentence):
            data.extend(procPunct(sentence))

    return ''.join(data)