VOTE_FILE="predict/breeze_7b_lora_completion_only.csv predict/breeze_7b_lora_completion_only_5_epochs.csv predict/llama_3_8b_lora_completion_only.csv"
WEIGHTS="0.890994 0.858999 0.87985"
OUTPUT_FILE="predict/llama_breeze_breeze_5.csv"

python src/vote.py \
    --vote_files $VOTE_FILE \
    --weights $WEIGHTS \
    --output_file $OUTPUT_FILE