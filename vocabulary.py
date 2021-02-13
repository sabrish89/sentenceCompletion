from collections import Counter

def build(List):
    '''
    builds a vocabulary from List
    '''
    return {key: idx for idx, key in enumerate(Counter(List))}