DATASET_NAME="DandinPower/Taiwanese_ASR"
MODEL_NAME="DandinPower/whisper-medium-tawiwanese-asr"
OUTPUT_PATH="submission/whisper-medium-tawiwanese-asr.csv"
INFERENCE_SPLIT="test"
SAMPLE_RATE=16000

python inference.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_path $OUTPUT_PATH \
    --inference_split $INFERENCE_SPLIT \
    --sample_rate $SAMPLE_RATE