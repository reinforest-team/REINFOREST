import argparse
import json
import os
import sys
from multiprocessing import Pool
import time

from tqdm import tqdm

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from source.baselines.baseline_ranker import Ranker
    from source.baselines.ast_comparison_search import ASTSearch
    from source.trainer import compute_metrics
    from source import util
    logger = util.get_logger(__file__)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name', '-e', help='Name of the experiment', required=True
    )
    parser.add_argument(
        '--ranker_name', '-r', help='Name of the ranker', required=True,
        choices=['bm25', 'subset', 'ast', 'semantic']
    )
    parser.add_argument(
        '--source_lang', '-s', help='Source language', required=True,
        choices=['java', 'python']
    )
    parser.add_argument(
        '--target_lang', '-t', help='Target language',
    )
    parser.add_argument(
        "--data_path", '-d', type=str,
        help="Base Directory of processed data", required=True
    )
    parser.add_argument(
        "--workers", '-w', help="Number of worker CPU", type=int, default=-1
    )
    parser.add_argument(
        '--rank_result_path', '-o', type=str, required=True,
        help='Path to store the ranked result'
    )
    parser.add_argument(
        '--parallel', '-p', action='store_true'
    )
    parser.add_argument(
        '--max_examples', '-m', type=int, default=99999999
    )
    parser.add_argument(
        '--ignore_no_scores', '-i', action='store_true'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line_args()
    logger.info(args)
    if args.source_lang == 'java':
        source_lang = 'java'
        target_lang = 'python'
    else:
        source_lang = 'python'
        target_lang = 'java'
    ranker = Ranker(
        comparator_name=args.ranker_name,
        use_multiprocessing=args.parallel,
        workers=args.workers,
        source_lang=source_lang,
        target_lang=target_lang,
        maximum_examples=args.max_examples,
        ignore_no_score=args.ignore_no_scores,
    )
    test_file = os.path.join(args.data_path, 'test.jsonl')
    logger.info(f'Ranking {test_file}')
    example = [json.loads(line) for line in open(test_file)]
    logger.info(f'Loaded {len(example)} examples')
    result, detailed_result, _ = ranker.rank(
        examples=example,
        ignore_no_positives=True,
        metric_function=compute_metrics
    )
    logger.info(
        '=' * 50,
        "\n" + json.dumps(result, indent=4),
        '=' * 50, sep="\n"
    )
    if args.rank_result_path is not None:
        os.makedirs(args.rank_result_path, exist_ok=True)
        result_file_name = os.path.join(
            args.rank_result_path, f'{args.ranker_name}.result'
        )
        result_file = open(
            result_file_name, 'w'
        )
        json.dump(result, result_file, indent=4)
        result_file.close()
        logger.info(f'Result saved to {result_file_name}')
        details_file = open(
            os.path.join(
                args.rank_result_path, f'{args.ranker_name}.details.json'
            ), 'w'
        )
        json.dump(detailed_result, details_file)
        details_file.close()


if __name__ == '__main__':
    main()
