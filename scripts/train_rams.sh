if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/rams/$SEED/$LR
mkdir -p $work_path

python -u engine.py \
    --model_type=epig_eae \
    --dataset_type=rams \
    --model_name_or_path=bart-base \
    --role_path=./data/dset_meta/description_rams.csv \
    --prompt_path=./data/prompts/prompts_rams_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --batch_size 1 \
    --max_steps=30000 \
    --max_enc_seq_length 500 \
    --max_prompt_seq_length 64 \
    --bipartite
