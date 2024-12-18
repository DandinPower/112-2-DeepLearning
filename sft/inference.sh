ADAPTER_NAME_OR_PATH=DandinPower/taiwanllm_7b_lora_completion_only
OUTPUT_CSV_PATH=predict/taiwanllm_7b_lora_completion_only.csv
DATASET_NAME_OR_PATH=DandinPower/ZH-Reading-Comprehension-TaiwanLlama-chat
# Available dataset
# 1. DandinPower/ZH-Reading-Comprehension-Llama-Instruct
# 2. DandinPower/ZH-Reading-Comprehension-Breeze-Instruct
# 3. DandinPower/ZH-Reading-Comprehension-gemma-it
# 4. DandinPower/ZH-Reading-Comprehension-TAIDE-Chat
# 5. DandinPower/ZH-Reading-Comprehension-Mistral-Instruct
# 6. DandinPower/ZH-Reading-Comprehension-TaiwanLlama-chat

MODEL_NAME_OR_PATH=yentinglin/Taiwan-LLM-7B-v2.1-chat
# Available model
# 1. meta-llama/Meta-Llama-3-8B-Instruct
# 2. MediaTek-Research/Breeze-7B-Instruct-v1_0
# 3. google/gemma-1.1-7b-it
# 4. taide/TAIDE-LX-7B-Chat
# 5. mistralai/Mistral-7B-Instruct-v0.2
# 6. taide/Llama3-TAIDE-LX-8B-Chat-Alpha1
# 7. yentinglin/Taiwan-LLM-7B-v2.1-chat

TEST_SPLIT=test
VALID_SPLIT=validation

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
    --do_sample \
    --do_valid \
    --do_test \
    --verbose \