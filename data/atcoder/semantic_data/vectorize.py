import openai
import time
from tqdm import tqdm
import json
import os

import sys
language = sys.argv[1]

def get_embedding(text, codex_model='babbage-code-search-text'):
    sleep_time = 1
    while True:
        try:
            return openai.Embedding.create(
                input=[text], model=codex_model
            )['data'][0]['embedding']
        except openai.error.RateLimitError as e:
            # print(f"Exception in get_codex_response_with_retries {e}")
            # print(type(e), e)
            if 'Please reduce' in str(e):
                text = text[:int(.9 * len(text))]
            time.sleep(sleep_time)
        except openai.error.OpenAIError as e:
            # print(f"Exception in get_codex_response_with_retries {e}")
            # print(type(e), e)
            if 'Please reduce' in str(e):
                text = text[:int(.9 * len(text))]
            time.sleep(sleep_time)
        except Exception as e:
            # print(f"Exception in get_codex_response_with_retries {e}")
            # print(type(e), e)
            if 'Please reduce' in str(e):
                text = text[:int(.9 * len(text))]
            time.sleep(sleep_time)

data_source = f"{language}/full_score"
codes = set()

for file in tqdm(['train.jsonl', 'valid.jsonl', 'test.jsonl']):
    fn = f"{data_source}/{file}"
    with open(fn) as f:
        data = [json.loads(line.strip()) for line in f]
        for d in data:
            codes.add(d['code'])
            positives = d['positives']
            for p in positives:
                codes.add(p['code'])
            negatives = d['negatives']
            for n in negatives:
                codes.add(n['code'])

codes = list(codes)
print(len(codes))

models = {
    # 'ada': 'ada-code-search-code',
    # 'babbage': 'babbage-code-search-code',
    # 'curie': 'curie-similarity',
    # 'davinci': 'davinci-similarity',
}

for model_name, model in models.items():
    output = f"{data_source}/embeddings_{model_name}.json"
    if os.path.exists(output):
        continue
    embeddings = {}
    bar = tqdm(codes, total=len(codes), desc=f"Embedding {language} with {model_name}")
    for code in bar:
        embeddings[code] = get_embedding(code, model)
    with open(f"{data_source}/embeddings_{model_name}.json", 'w') as f:
        json.dump(embeddings, f)
