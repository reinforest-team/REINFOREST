#!/usr/bin/env bash

if [[ $# -lt 4 ]]; then
    echo "Must provide at least four arguments";
    echo "bash rank.sh <exp_name> <dataset java/python> <alpha "0" for no SSS> <initial_model> [codex_model_name]";
    exit;
fi

gpus=0;
export CUDA_VISIBLE_DEVICES=$gpus;
exp_name=$1;
dataset=$2;
datatype=$3;
initial_model=$4;
codex_model=$5;

if [[ $codex_model = "ada" ]]; then
    model_initial="ada";
    codex_model="ada-code-search-code";
elif [[ $codex_model = "babbage" ]]; then
    model_initial="babbage";
    codex_model="babbage-code-search-code";
elif [[ $codex_model = "curie" ]]; then
    model_initial="curie";
    codex_model="curie-similarity";
elif [[ $codex_model = "davinci" ]]; then
    model_initial="davinci";
    codex_model="davinci-similarity";
else
    model_initial="none";
    codex_model="none";
fi

# if [[ $relaod_path != "" ]]; then
#     reload_args="--ckpt_path_from_other_exp $relaod_path";
# fi

exp_name="$exp_name-$datatype";

if [[ $datatype = 0 ]]; then
    datatype="wo";
    sm_factor="0.0";
else
    sm_factor=$datatype;
    datatype="with";
fi

cdir=`pwd`;
project_dir=`realpath ..`;
data_path="${project_dir}/data/atcoder/semantic_data/${dataset}/full_score";
embedding_path="${data_path}/embeddings_${model_initial}.json";
data_cache_path="${data_path}/${exp_name}-cached";
output_path="${project_dir}/models/atcoder/semantic_data/${dataset}/${datatype}_score/${exp_name}";
ranking_output_path="${project_dir}/models/atcoder/semantic_data/${dataset}/${datatype}_score/${exp_name}/ranking_result";
config_path="${project_dir}/configs/default.json";
log_dir="$project_dir/logs";
log_file="$log_dir/$exp_name-${dataset}-${sm_factor}-ranking.log";
seed=4000;
raw_data_path="${project_dir}/data/atcoder/raw_data";

if [[ $initial_model = "codex" ]]; then
    codex_specific_args="--embedding_path $embedding_path";
else
    codex_specific_args="";
fi

export PYTHONPATH=$project_dir:$PYTHONPATH;
mkdir -p $data_cache_path;
mkdir -p $output_path;
mkdir -p $ranking_output_path;
mkdir -p $log_dir;

python $project_dir/source/main.py \
    --exp_name $exp_name \
    --training_config $config_path \
    --data_path $data_path \
    --output_dir $output_path \
    --initial_model $initial_model \
    --raw_data $raw_data_path \
    --data_cache_path $data_cache_path \
    --semantic_match_factor $sm_factor --do_rank \
    --codex_model $codex_model \
    $codex_specific_args \
    --rank_result_path $ranking_output_path \
    --seed $seed 2>&1 | tee $log_file;
