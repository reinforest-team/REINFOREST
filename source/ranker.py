import json
import os
import sys
import numpy as np
import torch.nn as nn
from typing import Type, Dict, Any, List, Callable, Tuple
from transformers.trainer_utils import EvalPrediction
from tqdm import tqdm


if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from source.codebert.data import CrossDataSetForCodeBERT
    from source.codex.data import CrossDataSetForCodex
    from source import util
    logger = util.get_logger(__file__)


def calculate_scores(vector, other_vectors):
    scores = []
    for o in other_vectors:
        scores.append(
            np.dot(vector, o) / (
                np.abs(np.linalg.norm(o, ord=2)) * \
                np.abs(np.linalg.norm(vector, ord=2))
            )
        )
    return np.array(scores)


def batchify(examples, batch_size=32):
    current_idx = 0
    batches = []
    while current_idx < len(examples):
        batches.append(examples[current_idx:current_idx+batch_size])
        current_idx += batch_size
    return batches


class Ranker:
    def __init__(
        self,
        data_class: type,
        model_class: type,
        additional_comp_for_ranker: Dict[str, Any],
    ):
        self.data_class = data_class
        self.model_class = model_class
        self.additional_comp_for_ranker = additional_comp_for_ranker
        assert (
            self.data_class == CrossDataSetForCodex or
            self.data_class == CrossDataSetForCodeBERT
        )
        if self.data_class == CrossDataSetForCodeBERT:
            assert "tokenizer" in self.additional_comp_for_ranker.keys()
        else:
            assert "model_name" in self.additional_comp_for_ranker.keys()
        self.cache = {}
        if "embedding_path" in self.additional_comp_for_ranker.keys():
            assert os.path.exists(
                self.additional_comp_for_ranker["embedding_path"]
            )
            logger.info(
                f'Loading from  {self.additional_comp_for_ranker["embedding_path"]}'
            )
            self.cache = json.load(
                open(self.additional_comp_for_ranker["embedding_path"], "r")
            )
            logger.info(f"Loaded {len(self.cache.keys())} codes from the cache")
            self.additional_comp_for_ranker["cache"] = self.cache
            self.additional_comp_for_ranker.pop("embedding_path")

    def rank(
        self,
        model: nn.Module,
        examples: List[Dict[str, Any]],
        metric_function: Callable[[EvalPrediction, bool], Dict[str, Any]],
        ignore_no_positives: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, List[float]], List[Dict[str, Any]]]:
        local_cache = {}
        all_codes = [examples[k]['code'] for k in range(len(examples))] + \
            [e['code'] for k in range(len(examples)) \
                for e in examples[k]['positives']] +\
            [e['code'] for k in range(len(examples)) \
                for e in examples[k]['negatives']]
        all_codes = list(set(all_codes))
        logger.info(f'Total Code to be cached : {len(all_codes)}')
        batches = batchify(all_codes)
        logger.info(
            f"Created {len(batches)} batches for computing the vectors")
        for batch in tqdm(batches):
            vectors = self.data_class.get_vector(
                model=model, texts=batch, 
                **self.additional_comp_for_ranker
            )
            for t, v in zip(batch, vectors):
                local_cache[t] = v
        logger.info(f"Inserted {len(local_cache.keys())} codes into the cache")
        results = {}
        bar = tqdm(examples, desc='rank_gap = 0.0000\t')
        for ex in bar:
            code = ex['code']
            if code in local_cache:
                code_vector = local_cache[code]
            else:
                code_vector = self.data_class.get_vector(
                    model=model, texts=code, **self.additional_comp_for_ranker
                )
            ex['code_vector'] = code_vector
            full_scores, full_labels = [], []
            for pid, p in enumerate(ex['positives']):
                if p['code'] in local_cache:
                    pv = local_cache[p['code']]
                else:
                    pv = self.data_class.get_vector(
                        model=model, texts=p['code'], 
                        **self.additional_comp_for_ranker
                    )
                local_cache[p['code']] = pv
                ex['positives'][pid]['code_vector'] = pv
                score = calculate_scores(
                    code_vector, [pv]
                ).tolist()[0]
                ex['positives'][pid]['similarity'] = score
                full_scores.append(score)
                full_labels.append(1)
                
            for nid, n in enumerate(ex['negatives']):
                if n['code'] in local_cache:
                    nv = local_cache[n['code']]
                else:
                    nv = self.data_class.get_vector(
                        model=model, texts=n['code'], 
                        **self.additional_comp_for_ranker
                    )
                local_cache[n['code']] = nv
                ex['negatives'][nid]['code_vector'] = nv
                score = calculate_scores(
                    code_vector, [nv]
                ).tolist()[0]
                ex['negatives'][nid]['similarity'] = score
                full_scores.append(score)
                full_labels.append(0)
            if ignore_no_positives and sum(full_labels) == 0:
                continue
            prediction = EvalPrediction(
                predictions=np.array([full_scores]),
                label_ids=np.array([full_labels])
            )
            current_ex_result = metric_function(prediction)
            bar.set_description(
                f'rank_gap = {round(current_ex_result["rank_gap"], 4)}\t'
            )
            for k in current_ex_result.keys():
                if k not in results.keys():
                    results[k] = []
                results[k].append(current_ex_result[k])

        aggr_result = {
            k: round(np.mean(results[k]), 4) for k in results.keys()
        }
        return aggr_result, results, examples    
            

if __name__ == '__main__':
    v = [2,]