VOTE_FILE="predict/llama_breeze_taide3_breeze5_gemma_weight.csv predict/llama_breeze_taide3.csv predict/llama_breeze_breeze5.csv predict/llama_breeze_breeze5_gemma_weighted.csv predict/llama_breeze_breeze5_gemma_taide_mistral_weighted.csv"
WEIGHTS="0.922995 0.920997 0.918996 0.916992 0.912999"
OUTPUT_FILE="predict/vote_of_blt3b5gw_blt3_bb5l_bb5lgw_bb5lgtmw_weight.csv"

python src/vote.py \
    --vote_files $VOTE_FILE \
    --weights $WEIGHTS \
    --output_file $OUTPUT_FILE