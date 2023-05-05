from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils import logging
from transformers.training_args import TrainingArguments
import os
import json
import sys

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    from source.codebert.data import CrossDataSetForCodeBERT


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    data_dir = f"{os.environ['HOME']}/REINFOREST/data/atcoder/semantic_match_data"
    data_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")
    ])
    print(data_files)
    arguments = TrainingArguments(output_dir="/tmp/output")
    data = CrossDataSetForCodeBERT(
        path=data_dir,
        data_files=data_files,
        name="dataloading-v1",
        tokenizer=tokenizer,
        cache_dir=os.path.join(data_dir, "cached"),
        num_workers=16,
        training_arguments=arguments,
        load_from_cache=True
    )
    # print(json.dumps(data[0], indent=4))
    # print(len(data))

