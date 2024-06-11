OUTPUT_DIR="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/synth_images_v2"
NUM_IMAGES=1000
TEXT_FILE="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/sentences/test_fake_sentences_v3.txt"
IMAGE_DIR="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/non_subtitle_images"
OUTPUT_FILE="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/synth_images_subtitles_v2.txt"
FONT_DIR="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/fit_video_chinese_font"

python ../utils/generate_synth_images.py \
    --output_dir $OUTPUT_DIR \
    --num_images $NUM_IMAGES \
    --text_file $TEXT_FILE \
    --image_dir $IMAGE_DIR \
    --output_file $OUTPUT_FILE \
    --font_dir $FONT_DIR