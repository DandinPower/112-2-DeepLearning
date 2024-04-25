function run_model {
    MODEL_NAME=$1
    DATASET_NAME=DandinPower/ZH-Reading-Comprehension
    SPLIT=train
    SAMPLE_SIZE=100
    OUTPUT_DIR=zero_shot_evaluation

    python src/zero_shot_evaluation.py \
        --model_name $MODEL_NAME \
        --dataset $DATASET_NAME \
        --split $SPLIT \
        --sample_size $SAMPLE_SIZE \
        --output_dir $OUTPUT_DIR
}

# MODELS=("MediaTek-Research/Breeze-7B-Instruct-v1_0" "google/gemma-1.1-7b-it" "meta-llama/Meta-Llama-3-8B-Instruct")
MODELS=("MediaTek-Research/Breeze-7B-Instruct-v1_0")

for MODEL in "${MODELS[@]}"
do
    run_model $MODEL
done