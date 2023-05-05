#!/usr/bin/env bash

if [[ $# -lt 5 ]]; then
    echo "Must provide at least four arguments";
    echo "bash train.sh <dataset java/python> <data_type with/wo semantic> <codex_model_name> <max_positive> <max_negative>";
    exit;
fi

gpus=0;
export CUDA_VISIBLE_DEVICES=$gpus;
# exp_name="";
dataset=$1;
datatype=$2;
initial_model="codex";
codex_model=$3;
max_positive=$4;
max_negative=$5;

exp_name="${codex_model}_sm-${datatype}_mp-${max_positive}_mn-${max_negative}";

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
elif [[ $codex_model = "ada2" ]]; then
    model_initial="ada_002";
    codex_model="text-embedding-ada-002";
else
    model_initial="none";
    codex_model="none";
    echo "MUST PROVIDE A VALID CODEX MODEL NAME from ada, babbage, curie, davinci";
    exit;
fi

# if [[ $relaod_path != "" ]]; then
#     reload_args="--ckpt_path_from_other_exp $relaod_path";
# fi

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
config_path="${project_dir}/configs/default.json";
log_dir="$project_dir/logs";
log_file="$log_dir/$exp_name-${dataset}-${sm_factor}.log";
seed=4000;
raw_data_path="${project_dir}/data/atcoder/raw_data";
ranking_output_path="${project_dir}/models/atcoder/semantic_data/${dataset}/${datatype}_score/${exp_name}/ranking_result";

if [[ $initial_model = "codex" ]]; then
    codex_specific_args="--embedding_path $embedding_path";
else
    codex_specific_args="";
fi

export PYTHONPATH=$project_dir:$PYTHONPATH;
mkdir -p $data_cache_path;
mkdir -p $output_path;
mkdir -p $log_dir;
mkdir -p $ranking_output_path;

# echo "python $project_dir/source/main.py --exp_name $exp_name --training_config $config_path --data_path $data_path --output_dir $output_path --initial_model $initial_model --data_cache_path $data_cache_path --seed $seed $reload_args 2>&1 | tee $log_file;"
python $project_dir/source/main.py \
    --exp_name $exp_name \
    --training_config $config_path \
    --max_positive_examples $max_positive \
    --max_negative_examples $max_negative \
    --data_path $data_path \
    --output_dir $output_path \
    --initial_model $initial_model \
    --raw_data $raw_data_path \
    --data_cache_path $data_cache_path \
    --semantic_match_factor $sm_factor --do_train \
    --do_rank \
    --codex_model $codex_model \
    $codex_specific_args \
    --rank_result_path $ranking_output_path \
    --seed $seed 2>&1 | tee $log_file;
