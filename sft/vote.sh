VOTE_FILE="predict/llama_3_8b_lora_completion_only.csv predict/breeze_7b_lora_completion_only.csv predict/breeze_7b_lora_completion_only_5_epochs.csv"
OUTPUT_FILE="predict/llama_breeze_breeze5.csv"

python src/vote.py \
    --vote_files $VOTE_FILE \
    --output_file $OUTPUT_FILE