MODEL_NAME_OR_PATH=MediaTek-Research/Breeze-7B-Instruct-v1_0
ADAPTER_NAME_OR_PATH=DandinPower/breeze_7b_lora
DATASET_NAME_OR_PATH=DandinPower/ZH-Reading-Comprehension-Breeze-Instruct
SPLIT=test
OUTPUT_CSV_PATH=test.csv

python src/inference.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --adapter_name_or_path $ADAPTER_NAME_OR_PATH \
    --dataset_name_or_path $DATASET_NAME_OR_PATH \
    --split $SPLIT \
    --output_csv_path $OUTPUT_CSV_PATH