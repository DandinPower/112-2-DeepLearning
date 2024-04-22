function run_model {
    MODEL_NAME=$1
    DATASET_NAME=DandinPower/ZH-Reading-Comprehension
    SPLIT=train
    SAMPLE_SIZE=20
    OUTPUT_DIR=zero_shot_evaluation

    python src/zero_shot_evaluation.py \
        --model_name $MODEL_NAME \
        --dataset $DATASET_NAME \
        --split $SPLIT \
        --sample_size $SAMPLE_SIZE \
        --output_dir $OUTPUT_DIR
}

# List of all models
# MODELS=("meta-llama/Llama-2-13b-chat-hf" "taide/TAIDE-LX-7B-Chat" "yentinglin/Taiwan-LLM-7B-v2.1-chat", "yentinglin/Taiwan-LLM-13B-v2.0-chat", "MediaTek-Research/Breeze-7B-Instruct-v1_0")
MODELS=("MediaTek-Research/Breeze-7B-Instruct-v1_0")


for MODEL in "${MODELS[@]}"
do
    run_model $MODEL
done