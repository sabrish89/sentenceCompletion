'''
1. Import dataset .txt
2. Append datasets .txt + .txt ....
'''

import re
from langdetect import detect
from roman import fromRoman
from re import sub

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

def processData(sentList):
    '''
    1. Clean list
    2. Combine list
    3. Return a string of data
    TODO: Translate roman numerals to english numbers ✔
    TODO: Language detection - only english ✔
    '''

    def procPunct(sentence):
        '''
        Remove punctuations and special rules
        '''
        return sub("(--)"," ",''.join([sub("(\n|\t|\r|\v|\f|,|\"|'|\*|\(|\)|\?)", "", word) for word in sentence]))

    def process(sentence, romanStops):
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

    def flattenAndCompress(sentence):
        '''
        Remove spaces and None
        '''
        return sentence.strip()

    romanStops = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
    for idx, sentence in enumerate(sentList): #idx first
        sentList[idx] = flattenAndCompress(process(sentence, romanStops))

    return sentList

print(processData(importData(".\\corpus\\edgar allen poe\\theFallOfTheHouseOfUsher.txt")))