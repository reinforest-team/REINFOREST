from dataclasses import dataclass
from torch import nn as nn
from torch.utils.data import Dataset as TorchDS
from datasets import load_dataset
from typing import List, Optional, Dict, Any, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DefaultDataCollator
import torch
import os
import sys

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from source.codebert.models import CodeBERTBasedModel


class CrossDataSetForCodeBERT(TorchDS):
    def __init__(
        self,
        path: str,
        data_files: List[str],
        name: str,
        tokenizer: PreTrainedTokenizer,
        training_arguments: TrainingArguments,
        cache_dir: Optional[str],
        num_workers: Optional[int] = 16,
        load_from_cache: Optional[bool] = True,
        max_positive_examples: Optional[int] = 3,
        max_negative_examples: Optional[int] = 3,
        *args, **kwargs
    ):
        super().__init__()
        self.max_positive_examples = max_positive_examples
        self.max_negative_examples = max_negative_examples
        self.training_args = training_arguments
        self.tokenizer = tokenizer
        self.data = load_dataset(
            path=path,
            data_dir=path,
            data_files=data_files,
            split="train",
            cache_dir=cache_dir,
            name=name
        )
        columns = self.data.column_names

        def prepare_features(examples):
            inputs = examples["code"]
            positives = examples["positives"]
            positives = sorted(positives, key=lambda x:x['score'], reverse=True)
            positive_codes = [p['code'] for p in positives]
            positive_scores = [p['score'] if p['score'] != -1 else 0. for p in positives]
            assert len(positive_codes) == len(positive_scores)
            if len(positive_codes) > max_positive_examples:
                positive_codes = positive_codes[:max_positive_examples]
                positive_scores = positive_scores[:max_positive_examples]
            elif len(positive_codes) < max_positive_examples:
                positive_codes.extend(
                    [""] * (max_positive_examples - len(positive_codes))
                )
                positive_scores.extend(
                    [0.] * (max_positive_examples - len(positive_scores))
                )
            negatives = examples['negatives']
            negatives = sorted(negatives, key=lambda x:x['score'], reverse=True)
            negative_codes = [p['code'] for p in negatives]
            negative_scores = [p['score'] if p['score'] != -1 else 0. for p in negatives]
            assert len(negative_codes) == len(negative_scores)
            if len(negative_codes) > max_negative_examples:
                negative_codes = negative_codes[:max_negative_examples]
                negative_scores = negative_scores[:max_negative_examples]
            elif len(negative_codes) < max_negative_examples:
                negative_codes.extend(
                    [""] * (max_negative_examples - len(negative_codes))
                )
                negative_scores.extend(
                    [0.] * (max_negative_examples - len(negative_scores))
                )
            tokenizer_output = self.tokenizer(
                inputs, max_length=tokenizer.model_max_length,
                padding="max_length", truncation=True
            )
            input_ids, attention_mask = tokenizer_output.input_ids, \
                tokenizer_output.attention_mask
            pos_input_ids, pos_attn_masks = [], []
            for p in positive_codes:
                tokenizer_output = self.tokenizer(
                    p, max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True
                )
                pos_input_ids.append(tokenizer_output.input_ids)
                pos_attn_masks.append(tokenizer_output.attention_mask)
            neg_input_ids, neg_attn_masks = [], []
            for n in negative_codes:
                tokenizer_output = self.tokenizer(
                    n, max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True
                )
                neg_input_ids.append(tokenizer_output.input_ids)
                neg_attn_masks.append(tokenizer_output.attention_mask)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pos_input_ids": pos_input_ids,
                "pos_attn_mask": pos_attn_masks,
                "pos_semantic_scores": positive_scores,
                "neg_input_ids": neg_input_ids,
                "neg_attn_mask": neg_attn_masks,
                "neg_semantic_scores": negative_scores
            }
        with self.training_args.main_process_first(desc=f"dataset map pre-processing {name}"):
            self.data = self.data.map(
                prepare_features,
                batched=False,  # Do not do batched processing, it will not work
                num_proc=num_workers,
                remove_columns=columns,
                load_from_cache_file=load_from_cache,
                desc=f"dataset map pre-processing {name}"
            )

    @classmethod
    def get_vector(
        cls,
        tokenizer: PreTrainedTokenizer,
        model: CodeBERTBasedModel,
        texts: Union[str, List[str]],
        no_train_rank: bool = False
    ):
        assert isinstance(model, CodeBERTBasedModel)
        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]
        tokenizer_output = tokenizer(
            texts, max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors='pt'
        )
        input_ids, attention_mask = tokenizer_output.input_ids, \
            tokenizer_output.attention_mask
        assert isinstance(input_ids, torch.LongTensor) \
            and isinstance(attention_mask, torch.LongTensor)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        vector = model.get_vector(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        if not batched:
            vector = vector.squeeze(0)
        return vector.cpu().numpy().tolist()

    def get_vector_from_dataset(
        self,
        model: CodeBERTBasedModel,
        texts: Union[str, List[str]]
    ):
        return CrossDataSetForCodeBERT.get_vector(
            tokenizer=self.tokenizer,
            model=model,
            texts=texts
        )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@dataclass
class CrossLangSearchDataCollatorforCodeBERT(DefaultDataCollator):
    def __call__(
        self,
        features: List[Dict[str, Any]],
        return_tensors=None
    ) -> Dict[str, Any]:
        batch = {}
        first = features[0]
        for k, v in first.items():
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                try:
                    batch[k] = torch.tensor([f[k] for f in features])
                except Exception as e:
                    print(k)
                    for f in features:
                        print(torch.tensor(f[k]).shape)
                    exit()
        return batch
