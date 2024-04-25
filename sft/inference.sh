MODEL_NAME_OR_PATH=MediaTek-Research/Breeze-7B-Instruct-v1_0
ADAPTER_NAME_OR_PATH=DandinPower/breeze_7b_lora_completion_only
DATASET_NAME_OR_PATH=DandinPower/ZH-Reading-Comprehension-Breeze-Instruct
TEST_SPLIT=test
VALID_SPLIT=validation
OUTPUT_CSV_PATH=predict/breeze_7b_lora_completion_only.csv

MAX_NEW_TOKENS=3
TEMPERATURE=0.7
TOP_K=10
TOP_P=0.95

python src/inference.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --adapter_name_or_path $ADAPTER_NAME_OR_PATH \
    --dataset_name_or_path $DATASET_NAME_OR_PATH \
    --test_split $TEST_SPLIT \
    --valid_split $VALID_SPLIT \
    --output_csv_path $OUTPUT_CSV_PATH \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --do_valid \
    --do_test \
    # --verbose \