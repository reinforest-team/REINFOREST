from __future__ import absolute_import, division, print_function
import datetime
from typing import List, Tuple
from model import Model
import multiprocessing

from tqdm import tqdm
import sys
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import torch
import numpy as np
import json
import random
import os
import argparse
from transformers.trainer_utils import EvalPrediction
from transformers import (
    AdamW, get_linear_schedule_with_warmup,
    BertConfig, BertForMaskedLM, BertTokenizer,
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    RobertaConfig, RobertaModel, RobertaTokenizer,
    DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
)

import warnings
warnings.filterwarnings("ignore")


if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from source.util import get_logger
    from source.trainer import compute_metrics
    from source.single_encoder_baselines.model import Model
    logger = get_logger(__file__)

cpu_cont = 16

global_cache = {}

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


def get_example(item):
    # global global_cache
    url1, url2, label, tokenizer, args, cache, url_to_code = item
    if url1 in cache:
        code1 = cache[url1].copy()
    else:
        try:
            code = ' '.join(url_to_code[url1].split())
        except:
            code = ""
        code1 = tokenizer.tokenize(code)
    # if url1 not in global_cache:
    #     global_cache[url1] = code1
    if url2 in cache:
        code2 = cache[url2].copy()
    else:
        try:
            code = ' '.join(url_to_code[url2].split())
        except:
            code = ""
        code2 = tokenizer.tokenize(code)
    # if url2 not in global_cache:
    #     global_cache[url2] = code2
    return convert_examples_to_features(
        code1, code2, label, url1, url2, tokenizer, args, cache
    )


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self, input_tokens, input_ids,
        label, url1, url2
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


def convert_examples_to_features(
    code1_tokens, code2_tokens, label, url1, url2, tokenizer, args, cache
):
    # source
    code1_tokens = code1_tokens[:args.block_size-2]
    code1_tokens = [tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens = code2_tokens[:args.block_size-2]
    code2_tokens = [tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id]*padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id]*padding_length

    source_tokens = code1_tokens+code2_tokens
    source_ids = code1_ids+code2_ids
    return InputFeatures(source_tokens, source_ids, label, url1, url2)


class TextDataset(Dataset):
    def __init__(
        self, tokenizer, args, file_path='train', block_size=512, pool=None,
        already_read_dataset=None
    ):
        data = []
        cache = {}
        url_to_code = {}
        if already_read_dataset is not None:
            if isinstance(already_read_dataset, list):
                dataset = already_read_dataset
            else:
                dataset = [already_read_dataset]
        else:
            dataset = []
            f = open(file_path, 'r')
            for line in f:
                dataset.append(json.loads(line.strip()))
        for d in dataset:
            code = d['code']
            url1 = d['base_sample_name']
            if url1 not in url_to_code:
                url_to_code[url1] = code
            positives = d['positives']
            negatives = d['negatives']
            for p in positives:
                scode = p['code']
                url2 = p['comparison_sample_name']
                if url2 not in url_to_code:
                    url_to_code[url2] = scode
                data.append(
                    (
                        url1, url2, 1, tokenizer,
                        args, cache, url_to_code
                    )
                )
            for p in negatives:
                scode = p['code']
                url2 = p['comparison_sample_name']
                if url2 not in url_to_code:
                    url_to_code[url2] = scode
                data.append(
                    (
                        url1, url2, 0, tokenizer,
                        args, cache, url_to_code
                    )
                )
        if pool is None:
            self.examples = [
                get_example(d) for d in data
            ]
        else:
            self.examples = pool.map(
                get_example, data
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (
            torch.tensor(self.examples[item].input_ids),
            torch.tensor(self.examples[item].label)
        )


def load_and_cache_examples(
    _args, _tokenizer, evaluate=False, test=False, pool=None,
    already_read_dataset=None
):
    global args, tokenizer
    if _args is not None:
        args = _args
    if _tokenizer is not None:
        tokenizer = _tokenizer
    dataset = TextDataset(
        tokenizer, args, file_path=args.test_data_file if test else (
            args.eval_data_file if evaluate else args.train_data_file
        ), block_size=args.block_size, pool=pool,
        already_read_dataset=already_read_dataset
    )
    return dataset


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, tokenizer, prefix="", pool=None, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(
        args, tokenizer, evaluate=True, pool=pool)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler,
        batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader):
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0
    best_f1 = 0
    for i in range(1, 100):
        threshold = i/100
        y_preds = logits[:, 1] > threshold
        from sklearn.metrics import recall_score
        recall = recall_score(y_trues, y_preds)
        from sklearn.metrics import precision_score
        precision = precision_score(y_trues, y_preds)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_trues, y_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }
    logger.info(f"***** Eval results {prefix} *****")
    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(round(result[key], 4))}")
    return result


def train(args, train_dataset, model, tokenizer, pool):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    args.max_steps = args.epoch*len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        pass
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info(f"***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = ",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = ",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = ", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_f1 = 0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    # Added here for reproducibility (even between python 2 and 3)
    set_seed(args.seed)
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                pass
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

        # if args.local_rank == -1 and args.evaluate_during_training:
        results = evaluate(
            args, model, tokenizer, pool=pool, eval_when_training=True)
        # Save model checkpoint
        if results['eval_f1'] > best_f1:
            best_f1 = results['eval_f1']
            logger.info("  " + "*" * 20)
            logger.info("  Best f1: ", round(best_f1, 4))
            logger.info("  " + "*" * 20)
            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(
                args.output_dir, '{}'.format(checkpoint_prefix)
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to ", output_dir)
    return global_step, tr_loss / global_step


args = None
model = None
tokenizer = None


class SimpleDataLoader:
    def batchify(self, batch: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """Gather a batch of individual examples into one batch."""
        batch = tuple(zip(*batch))
        return tuple(torch.stack(x, dim=0) for x in batch)

    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.idx = 0
        self.batches = []
        # logger.info(self.data[0][0].shape, self.data[0][1].shape)
        while self.idx < len(self.data):
            batch = []
            for i in range(self.batch_size):
                if self.idx + i >= len(self.data):
                    break
                batch.append(self.data[self.idx + i])
            batch = self.batchify(batch)
            self.idx += self.batch_size
            self.batches.append(batch)
        self.batch_idx = 0
        self.num_batches = len(self.batches)
        # logger.info(self.batches[0][0].shape, self.batches[0][1].shape)
    
    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration
        batch = self.batches[self.batch_idx]
        self.batch_idx += 1
        return batch


def test_one_example(data):
    args, model, tokenizer, d = data
    eval_dataset = load_and_cache_examples(
        args, tokenizer, test=True, pool=None,
        already_read_dataset=d
    )
    eval_dataloader = SimpleDataLoader(eval_dataset, batch_size=args.eval_batch_size)
    logits = []
    for batch in eval_dataloader.batches:
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            _, logit = model(inputs, labels)
            logits.append(logit.cpu().numpy())
    logits = np.concatenate(logits, 0)
    rank_score = {}
    for example, logit in zip(eval_dataset.examples, logits):
        if example.url1 not in rank_score:
            rank_score[example.url1] = {}
        rank_score[example.url1][example.url2] = logit[1]
    url1 = d['base_sample_name']
    scores, labels = [], []
    positives = d['positives']
    negatives = d['negatives']
    for p in positives:
        url2 = p['comparison_sample_name']
        try:
            score = rank_score[url1][url2]
        except:
            score = 0
        scores.append(score)
        labels.append(1)
    for n in negatives:
        url2 = n['comparison_sample_name']
        try:
            score = rank_score[url1][url2]
        except:
            score = 0
        scores.append(score)
        labels.append(0)
    prediction = EvalPrediction(
        predictions=np.array([scores]),
        label_ids=np.array([labels]),
    )
    current_ex_result = compute_metrics(prediction)
    return current_ex_result


def test(args, model, tokenizer, prefix=""):
    test_file = args.test_data_file
    results = {}
    f = open(test_file, 'r', encoding='utf-8')
    test_data = [json.loads(i) for i in f.readlines()]
    f.close()
    np.random.seed(datetime.datetime.now().microsecond)
    np.random.shuffle(test_data)
    test_data = test_data[:args.num_test_examples]

    model.eval()
    logger.info(f"***** Running Test {prefix} *****")
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    t_data = [
        (args, model, tokenizer, d) for d in test_data
    ]
    
    if args.distributed_testing:
        multiprocessing.set_start_method('spawn', force=True)
        mp_pool = multiprocessing.Pool(
            processes=multiprocessing.cpu_count() - 2
        )
        mapper = mp_pool.imap(
            func=test_one_example, iterable=t_data, chunksize=1
        )
    else:
        mapper = map(
            test_one_example, t_data
        )
    progress_bar = tqdm(
        mapper, total=len(t_data)
    )
    for didx, current_ex_result in enumerate(progress_bar):
        if didx % 100 == 0:
            logger.info(
                f"***** Running Test {prefix} {didx} *****")
            logger.info("\n" + json.dumps(current_ex_result, indent=4))
        for k in current_ex_result.keys():
            if k not in results.keys():
                results[k] = []
            results[k].append(current_ex_result[k])
        progress_bar.update()
    aggr_result = {
        k: round(np.mean(results[k]), 4) for k in results.keys()
    }
    logger.info(
        '=' * 50,
        "\n" + json.dumps(aggr_result, indent=4),
        '=' * 50, sep="\n"
    )
    if args.rank_result_path is not None:
        os.makedirs(args.rank_result_path, exist_ok=True)
        result_file_name = os.path.join(
            args.rank_result_path, f'summary.result'
        )
        result_file = open(
            result_file_name, 'w'
        )
        json.dump(aggr_result, result_file, indent=4)
        result_file.close()
        logger.info(f'Result saved to {result_file_name}')
        details_file = open(
            os.path.join(
                args.rank_result_path, f'summary.details.json'
            ), 'w'
        )
        json.dump(results, details_file)
        details_file.close()
    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str, required=True,
                        help="The input evaluation data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--test_data_file", default=None, type=str, required=True,
                        help="An input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port', type=str,
                        default='', help="For distant debugging.")
    parser.add_argument(
        '--num_test_examples', type=int, default=9999999
    )
    parser.add_argument(
        '--rank_result_path', type=str, default='rank_result'
    )
    parser.add_argument(
        '--distributed_testing', action='store_true'
    )

    args = parser.parse_args()
    if args.do_train:
        pool = multiprocessing.Pool(cpu_cont)
    else:
        pool = None

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        pass

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" % (
            args.local_rank, device, args.n_gpu, bool(
                args.local_rank != -1), args.fp16)
    )

    set_seed(args.seed)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(
            checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(
            checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = config.max_position_embeddings - 2
    args.block_size = min(
        args.block_size, config.max_position_embeddings - 2)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(
                '.ckpt' in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        train_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=False, pool=pool)

        if args.local_rank == 0:
            torch.distributed.barrier()
        train(args, train_dataset, model, tokenizer, pool)

    if args.do_test and args.local_rank in [-1, 0]:
        if pool is not None:
            pool.close()
            pool = None
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(
            args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)


if __name__ == "__main__":
    main()
