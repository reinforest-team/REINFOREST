import json
import os
import sys
import nltk


if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    new_tokens = []
    for token in tokens:
        new_tokens.extend(nltk.wordpunct_tokenize(token))
    return new_tokens


def read_scores(lang):
    score_map = {}
    with open(f"{project_dir}/data/atcoder/semantic_data/{lang}/with_score.json") as f:
        data = json.load(f)
        for p in ['train', 'val', 'test']:
            problems = data[f"{p}_data"]
            for p in problems:
                src = " ".join(p['base_sample_code'])
                targets = p['positives'] + p['negatives']
                for t in targets:
                    tgt = " ".join(tokenize(t['code']))
                    try:
                        score = float(t['score'])
                        if score < 0:
                            score = 0
                    except:
                        score = 0
                    score_map[(src, tgt)] = score
    return score_map


class SemanticSearch:
    def __init__(self, lang, ignore_no_score):
        self.lang = lang
        self.score_map = read_scores(lang)
        self.ignore_no_score = ignore_no_score

    def get_scores(self, code, corpus):
        tokenized_code = " ".join(tokenize(code))
        tokenized_corpus = [" ".join(tokenize(doc)) for doc in corpus]
        doc_scores = []
        taken_count = 0
        for doc in tokenized_corpus:
            if (tokenized_code, doc) in self.score_map:
                doc_scores.append(self.score_map[(tokenized_code, doc)])
                taken_count += 1
            else:
                doc_scores.append(0)
        if self.ignore_no_score and taken_count == 0:
                return None
        return doc_scores