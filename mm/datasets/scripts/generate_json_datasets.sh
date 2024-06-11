MODEL_TYPE='tinyllava' # 'minicpm' or 'tinyllava'
TRAIN_RATE=1 # tinyllava: 1, minicpm: 0.8
SEED=42
IMAGES_FOLDER='/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/synth_images'
SUBTITLES_FILE='/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/synth_images_subtitles.txt'
TRAIN_DATASETS_FILE='/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/json_datasets/tinyllava_train.json'
VALIDATION_DATASETS_FILE='/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/json_datasets/tinyllava_validation.json'

python ../utils/generate_json_datasets.py \
    --model_type $MODEL_TYPE \
    --train_rate $TRAIN_RATE \
    --images_folder $IMAGES_FOLDER \
    --subtitles_file $SUBTITLES_FILE \
    --train_datasets_file $TRAIN_DATASETS_FILE \
    --validation_datasets_file $VALIDATION_DATASETS_FILE \
    --seed $SEED