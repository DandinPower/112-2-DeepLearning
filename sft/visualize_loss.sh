TRAIN_STATE_JSON_PATH=/mnt/nvme0n1p1/supervised_fine_tuning/taiwanllm_7b_lora_completion_only/checkpoint-2000/trainer_state.json
IMAGE_PATH=images/taiwanllm_7b_lora_completion_only.png

python src/visualize_loss.py \
    --trainer_state_json_path $TRAIN_STATE_JSON_PATH \
    --image_path $IMAGE_PATH