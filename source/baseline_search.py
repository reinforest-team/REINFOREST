''' These are baseline searches that we use as a comparison
    we will use a bt25 search and a ast matching algorithm '''
import sys

from rank_bm25 import BM250kapi
def bt25Search(query, corpus):
    '''query: string, corpus: array of strings
        returns: array of scores {0,1} of same size as corpus
    '''
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    return doc_scores
def astMatchSearch(queary, corpus):
    ''''''
    pass
    
if __name__ == '__main__':
    pass
