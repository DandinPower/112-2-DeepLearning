VOTE_FILE="predict/llama_breeze_breeze5.csv predict/llama_breeze_breeze5_gemma_weighted.csv predict/llama_breeze_breeze5_gemma_taide_mistral_weighted.csv predict/llama_breeze_gemma_taide.csv predict/llama_breeze_gemma_taide_mistral.csv predict/gemma_7b_lora_completion_only.csv predict/taide_7b_lora_completion_only.csv predict/mistral_7b_lora_completion_only.csv"
WEIGHTS="0.918996 0.916992 0.912999 0.910996 0.91 0.899996 0.846999 0.794998 0.750998"
OUTPUT_FILE="predict/ultimate_vote_v3.csv"

python src/vote.py \
    --vote_files $VOTE_FILE \
    --weights $WEIGHTS \
    --output_file $OUTPUT_FILE