NNODES=1
NPROC_PER_NODE=2

DATASET_NAME_OR_PATH=DandinPower/ZH-Reading-Comprehension-Llama-Instruct
# Available dataset
# 1. DandinPower/ZH-Reading-Comprehension-Llama-Instruct
# 2. DandinPower/ZH-Reading-Comprehension-Breeze-Instruct
# 3. DandinPower/ZH-Reading-Comprehension-gemma-it
# 4. DandinPower/ZH-Reading-Comprehension-TAIDE-Chat
# 5. DandinPower/ZH-Reading-Comprehension-Mistral-Instruct

MODEL_NAME_OR_PATH=taide/Llama3-TAIDE-LX-8B-Chat-Alpha1
# Available model
# 1. meta-llama/Meta-Llama-3-8B-Instruct
# 2. MediaTek-Research/Breeze-7B-Instruct-v1_0
# 3. google/gemma-1.1-7b-it
# 4. taide/TAIDE-LX-7B-Chat
# 5. mistralai/Mistral-7B-Instruct-v0.2
# 6. taide/Llama3-TAIDE-LX-8B-Chat-Alpha1

OUTPUT_DIR=/mnt/nvme0n1p1/supervised_fine_tuning/taide_llama3_8b_lora_completion_only
TRAIN_SPLIT=train
VAL_SPLIT=validation

LORA_ALPHA=128
LORA_DROPOUT=0.1
LORA_RANK=64

MAX_SEQ_LENGTH=4096
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
WARMUP_STEPS=700
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
EVALUATION_STRATEGY=steps
EVALUATION_STEPS=250
SAVE_STRATEGY=steps
SAVE_STEPS=250
LOGGING_STEPS=50
NUM_TRAIN_EPOCHS=3

DEEPSPEED_CONFIG=config/ds_config.json

torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE src/training.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name_or_path $DATASET_NAME_OR_PATH \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --output_dir $OUTPUT_DIR \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_rank $LORA_RANK \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --evaluation_strategy $EVALUATION_STRATEGY \
    --eval_steps $EVALUATION_STEPS \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --deepspeed $DEEPSPEED_CONFIG \
    --overwrite_output_dir \
    --push_to_hub \
    --completion_only_training \