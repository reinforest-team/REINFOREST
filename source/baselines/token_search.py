''' These are baseline searches that we use as a comparison
    we will use a bt25 search and a ast matching algorithm '''
import sys
from rank_bm25 import BM25Okapi
import nltk

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    new_tokens = []
    for token in tokens:
        new_tokens.extend(nltk.wordpunct_tokenize(token))
    return new_tokens

class BM25Search:
    def __init__(self):
        pass

    def get_scores(self, code, corpus):
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_code = tokenize(code)
        doc_scores = bm25.get_scores(tokenized_code)
        return doc_scores


class SubsetSearch:
    def __init__(self):
        pass

    def get_scores(self, code, corpus):
        tokenized_corpus = [set(tokenize(doc)) for doc in corpus]
        tokenized_code = set(tokenize(code))
        scores = [
            len(tokenized_code.intersection(doc)) / 
            len(tokenized_code.union(doc)) 
            for doc in tokenized_corpus
        ]
        return scores