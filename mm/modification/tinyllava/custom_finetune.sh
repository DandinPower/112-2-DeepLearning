DATA_PATH="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/json_datasets/tinyllava_train.json"
IMAGE_PATH="/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/synth_images"
MODEL_MAX_LENGTH=1024
OUTPUT_DIR="/home/liaw/Desktop/112-2-DeepLearning/mm/TinyLLaVA_Factory/exp/tinyllava_images_latest_sentence_v3_total_1000_epoch_5-lora"

deepspeed --include localhost:0,1 --master_port 29501 /home/liaw/Desktop/112-2-DeepLearning/mm/TinyLLaVA_Factory/tinyllava/train/custom_finetune.py \
    --deepspeed /home/liaw/Desktop/112-2-DeepLearning/mm/TinyLLaVA_Factory/scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version phi \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 True \
    --training_recipe lora \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --pretrained_model_path "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name tinyllava_images_latest_sentence_v3_total_1000_epoch_5-lora
