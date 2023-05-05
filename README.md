# REINFOREST

## Requirements
```
python = 3.9;
openai;
tiktoken;
nltk==3.8.1;
pytorch==1.13.1 
pytorch-cuda=11.6;
transformers==4.16.2;
datasets==1.18.3;
scikit-learn==1.2.1;
```

Install Anaconda from [here.](https://www.anaconda.com/download/)

# Setting up the repository
### Step 1: Setup environment
```bash
conda create --name reinforest python=3.6;
conda activate reinforest;
```
### Step 2: Setup dependencies
```
bash setup.sh
```

# Get the data
```
cd data/atcoder/semantic_data;
bash download.sh;
cd ../../..;
```

# To train a new model for search
For now, we support RoBERTa stule models such as CodeBERT, GraphCodeBERT, RoBERTa, etc.,as well as pre-trained embeddings from four different Codex mdoels -- `ada`, `babbage`, `curie`, `davinci`. The training and evaluation scripts are inside [`scripts/` direstory](scripts).

1. To train a RoBERTa model (such as CodeBERT), from inside the [`scripts/`](scripts) directory.
 ```sh
    bash train.sh \
      <experiment_name> \
      <query_language: java or python> \
      <alpha_for_SSS: put this parameter 0 to ignore SSS> \
      <initial_model: which can be one of "codebert", "graphcodebert", "roberta-base", or "codex"> \
      <codex_model_name: This is optional, if initial_model = codex, choose one of "ada", "babbage", "curie", "davinci">
  ```
  Note that, [`train.sh`](scripts/train.sh) will also run the ranker after training. If you just want to run the ranker (on an already trained model), try [`rank.sh`](scripts/rank.sh) with the similar parameters as [`train.sh`](scripts/train.sh).

2. To run a ranker on untrained model, For example, just using the pretrained embedding from codex model, run [`rank_wo_training.sh`](scripts/rank_wo_training.sh) from the [`scripts/`](scripts) directory using similar parameters as [`train.sh`](scripts/train.sh). 


