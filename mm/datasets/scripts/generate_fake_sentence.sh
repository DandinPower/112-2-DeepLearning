ORIGINAL_FILE='/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/sentences/test_fake_sentences_v3.txt'
OUTPUT_FILE='/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/sentences/test_fake_sentences_v4.txt'
TYPE='txt' # csv or txt
BATCH_SIZE=10

python ../utils/generate_fake_sentence.py \
    --original_file $ORIGINAL_FILE \
    --output_file $OUTPUT_FILE \
    --type $TYPE \
    --batch_size $BATCH_SIZE