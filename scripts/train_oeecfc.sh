if [ $# == 0 ] 
then
    SEED=21
    LR=5e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/ace05/$SEED/$LR
mkdir -p $work_path

python engine.py \
    --model_type=epig_eae \
    --dataset_type=oeecfc \
    --model_name_or_path=bart-base-chinese \
    --role_path=./data/dset_meta/description_OEE_CFC.csv \
    --prompt_path=./data/prompts/prompts_OEE_CFC_specific_continuous.csv \
    --seed=$SEED \
    --output_dir=$work_path  \
    --learning_rate=$LR \
    --batch_size 4 \
    --infer_batch_size 32 \
    --max_steps=10000 \
    --max_enc_seq_length 600 \
    --max_prompt_seq_length 300 \
    --bipartite
