TEXT_COLUMN_NAME="transcription"
LANGUAGE="chinese"

TRAIN_SPLIT_NAME="train"
EVAL_SPLIT_NAME="validation"

MODEL_NAME_OR_PATH="openai/whisper-large"
DATASET_NAME="DandinPower/Taiwanese_ASR"

MAX_STEPS="5000"
PER_DEVICE_TRAIN_BATCH_SIZE="4"
PER_DEVICE_EVAL_BATCH_SIZE="4"
LEARNING_RATE="1e-5"
WARMUP_STEPS="500"
GRADIENT_ACCUMULATION_STEPS="8"

EVAL_STEPS="200"
SAVE_STEPS="200"
GENERATION_MAX_LENGTH="225"

MODEL_INDEX_NAME=whisper-large-taiwanese-asr
OUTPUT_DIR="./exp/$MODEL_INDEX_NAME"

# Clear out the output directory
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# Train the model
python run_speech_recognition_seq2seq_streaming.py \
	--model_index_name $MODEL_INDEX_NAME \
	--model_name_or_path $MODEL_NAME_OR_PATH --dataset_name $DATASET_NAME \
	--text_column_name $TEXT_COLUMN_NAME \
	--language $LANGUAGE \
	--train_split_name $TRAIN_SPLIT_NAME --eval_split_name $EVAL_SPLIT_NAME \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
	--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_steps $MAX_STEPS --learning_rate $LEARNING_RATE --warmup_steps $WARMUP_STEPS --generation_max_length $GENERATION_MAX_LENGTH \
	--output_dir $OUTPUT_DIR \
	--logging_steps="25" \
	--evaluation_strategy="steps" \
	--eval_steps $EVAL_STEPS \
	--save_steps $SAVE_STEPS \
	--save_strategy="steps" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--use_auth_token \
	--push_to_hub \
	--gradient_checkpointing \
	--optim="adamw_bnb_8bit" \
	# --dataset_config_name $DATASET_CONFIG_NAME \
	# --streaming \