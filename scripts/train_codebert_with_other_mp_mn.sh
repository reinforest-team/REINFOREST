#!/usr/bin/env bash

if [[ $# -lt 4 ]]; then
    echo "Must provide at least four arguments";
    echo "bash train.sh <dataset java/python> <semantic_factor> <max_positive> <max_negative>";
    exit;
fi

gpus=0;
export CUDA_VISIBLE_DEVICES=$gpus;
# exp_name="";
dataset=$1;
datatype=$2;
initial_model="codebert";
max_positive=$3;
max_negative=$4;

exp_name="codebert_sm-${datatype}_mp-${max_positive}_mn-${max_negative}";

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
data_cache_path="${data_path}/${exp_name}-cached";
output_path="${project_dir}/models/atcoder/semantic_data/${dataset}/${datatype}_score/${exp_name}";
config_path="${project_dir}/configs/codebert_default.json";
log_dir="$project_dir/logs";
log_file="$log_dir/$exp_name.log";
seed=4000;
ranking_output_path="${project_dir}/models/atcoder/semantic_data/${dataset}/${datatype}_score/${exp_name}/ranking_result";

export PYTHONPATH=$project_dir:$PYTHONPATH;
mkdir -p $data_cache_path;
mkdir -p $output_path;
mkdir -p $log_dir;
mkdir -p $ranking_output_path;

python $project_dir/source/main.py \
    --exp_name $exp_name \
    --training_config $config_path \
    --max_positive_examples $max_positive \
    --max_negative_examples $max_negative \
    --data_path $data_path \
    --output_dir $output_path \
    --initial_model $initial_model \
    --data_cache_path $data_cache_path \
    --semantic_match_factor $sm_factor --do_train \
    --do_rank \
    --rank_result_path $ranking_output_path \
    --seed $seed 2>&1 | tee $log_file;
