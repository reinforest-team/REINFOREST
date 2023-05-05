language=$1;
# model="initial_models/concord-codebert"
# model="initial_models/concord-graphcodebert"
# model="microsoft/graphcodebert-base"
mdl=$2;
bs=16;
gs=2;
if [ $mdl == "cb" ]; then
    model="microsoft/codebert-base"
elif [ $mdl == "gcb" ]; then
    model="microsoft/graphcodebert-base"
elif [ $mdl == "unix" ]; then
    model="microsoft/unixcoder-base"
    bs=8;
    gs=4;
elif [ $mdl == "c-cb" ]; then
    model="./initial_models/concord-codebert"
elif [ $mdl == "c-gcb" ]; then
    model="./initial_models/concord-graphcodebert"
else
    echo "Invalid model name"
    exit 1
fi

project_base=`realpath ../../`;
data_dir="$project_base/data/atcoder/semantic_data/$language/full_score"
model_dir="$project_base/models/atcoder/semantic_data/$language/single_encoder_baselines/$model"
mkdir -p $model_dir;

ckpt_file="${model_dir}/checkpoint-best-f1/model.bin";
if [[ -f $ckpt_file ]]; then
    train_args=""
else
    train_args="--do_train"
fi

python run.py \
    --train_data_file $data_dir/train.jsonl \
    --output_dir $model_dir \
    --eval_data_file $data_dir/valid.jsonl \
    --test_data_file $data_dir/test.jsonl \
    --model_type roberta \
    --model_name_or_path $model \
    --tokenizer_name $model \
    $train_args \
    --do_test \
    --evaluate_during_training \
    --train_batch_size $bs \
    --eval_batch_size $bs \
    --gradient_accumulation_steps $gs \
    --epoch 5 \
    --save_total_limit 1 \
    --rank_result_path $model_dir/ranking_result \
    --num_test_examples 200 \
    --distributed_testing
