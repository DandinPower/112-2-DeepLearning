INPUT_DIRECTORY="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/test_images"
OUTPUT_CSV="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/sentences/test_extract_sentences.csv"

python ../utils/extract_subtitle_by_openai.py \
    --input_directory $INPUT_DIRECTORY \
    --output_csv $OUTPUT_CSV