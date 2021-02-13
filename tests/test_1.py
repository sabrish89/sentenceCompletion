'''
Test word sequences
'''

from accessories import importData, parallelizeTextProcessing, genSeq, genSent
from models.modeling.naiveRNN import naiveRNN, trainModel
from random import randint
from vocabulary import build

if __name__ == '__main__':
    sentList = importData("../corpus/edgar allen poe/theFallOfTheHouseOfUsher.txt")
    sentStringListP, timeP = parallelizeTextProcessing(sentList, time=True)  # pass sentList.copy() if use again
    vocab = build(sentStringListP)  # use a naive one-hot for sequences

    seqLength = 3

    X, Y = genSeq(sentStringListP, vocab, seqLength)

    # Build LSTM

    naiveLSTM = naiveRNN(vocab.__len__(), seqLength)
    naiveLSTM = trainModel(naiveLSTM, X, Y)

    # Make a random prediction
    randIdx = randint(0, sentStringListP.__len__())
    genSent(naiveLSTM, sentStringListP[randIdx: randIdx+seqLength], vocab, seqLength, textIterations=25)
