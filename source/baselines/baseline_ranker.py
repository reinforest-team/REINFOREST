import copy
import json
import os
import sys
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple
from transformers.trainer_utils import EvalPrediction
from tqdm import tqdm
from multiprocessing import Pool


if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from source.baselines.token_search import BM25Search, SubsetSearch
    from source.baselines.ast_comparison_search import ASTSearch
    from source.baselines.semantic_search import SemanticSearch
    from source import util
    logger = util.get_logger(__file__)


def process_example(example):
    args = example['args']
    source_lang = args['source_lang']
    target_lang = args['target_lang']
    comparator_name = args['comparator_name']
    if comparator_name == 'bm25':
        comparator = BM25Search()
    elif comparator_name == 'subset':
        comparator = SubsetSearch()
    elif comparator_name == 'semantic':
        comparator = SemanticSearch(
            lang=source_lang,
            ignore_no_score=args['ignore_no_score'],
        )
    else:
        comparator = ASTSearch(
            source_lang=source_lang,
            target_lang=target_lang,
        )
    
    code = example['code']
    full_labels = []
    corpus_code = []
    for p in example['positives']:
        corpus_code.append(p['code'])
        full_labels.append(1)
    for n in example['negatives']:
        corpus_code.append(n['code'])
        full_labels.append(0)
    scores = comparator.get_scores(code, corpus_code)
    return scores, full_labels


class Ranker:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        comparator_name: str,
        use_multiprocessing: bool = True,
        workers: int = 20,
        maximum_examples: Optional[int] = None,
        ignore_no_score: bool = False,
    ):
        self.comparator_name = comparator_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        if comparator_name not in ['bm25', 'subset', 'ast', 'semantic']:
            raise ValueError(
                f"Comparator name {comparator_name} not supported")
        self.use_multiprocessing = use_multiprocessing
        if workers <= 0:
            workers = os.cpu_count()
        self.workers = workers if self.use_multiprocessing else 1
        self.args = {
            'source_lang': source_lang,
            'target_lang': target_lang,
            'comparator_name': comparator_name,
            'ignore_no_score': (
                comparator_name == 'semantic' and
                ignore_no_score
            ),
        }
        self.maximum_examples = maximum_examples

    def rank(
        self,
        examples: List[Dict[str, Any]],
        metric_function: Callable[[EvalPrediction, bool], Dict[str, Any]],
        ignore_no_positives: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, List[float]], List[Dict[str, Any]]]:
        for idx in range(len(examples)):
            examples[idx]['args'] = copy.copy(self.args)
        np.random.shuffle(examples)
        examples = examples[:self.maximum_examples]
        results = {}
        if self.workers > 1:
            logger.info(f'Using {self.workers} workers')
            pool = Pool(self.workers,)
            wrapped_examples = pool.imap(
                process_example, examples, chunksize=1
            )
        else:
            logger.info(f'Using single worker')
            wrapped_examples = map(
                process_example, examples
            )
        taken_example_count = 0
        bar = tqdm(wrapped_examples, total=len(examples))
        # bar = enumerate(wrapped_examples)
        for idx, result in enumerate(bar):
            # logger.info(f'Processing example {idx}')
            full_scores, full_labels = result
            bar.update()
            if full_scores is None:
                logger.info(f'Example {idx} failed to process')
                continue
            if ignore_no_positives and sum(full_labels) == 0:
                continue
            prediction = EvalPrediction(
                predictions=np.array([full_scores]),
                label_ids=np.array([full_labels])
            )
            current_ex_result = metric_function(prediction)
            for k in current_ex_result.keys():
                if k not in results.keys():
                    results[k] = []
                results[k].append(current_ex_result[k])
            taken_example_count += 1
            # if self.maximum_examples is not None \
            #     and idx >= (self.maximum_examples - 2):
            #     break
        aggr_result = {
            k: round(np.mean(results[k]), 4) for k in results.keys()
        }
        logger.info(f'Aggregated result: {aggr_result}')
        logger.info(f'Taken {taken_example_count} examples')
        return aggr_result, results, examples    
            
