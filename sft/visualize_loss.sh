# parser.add_argument("--trainer_state_json_path", type=str, required=True)
#     parser.add_argument("--image_path", type=str, required=True)

TRAIN_STATE_JSON_PATH=/mnt/nvme0n1p1/supervised_fine_tuning/breeze_7b_lora_completion_only/checkpoint-1750/trainer_state.json
IMAGE_PATH=images/breeze_7b_lora_completion_only.png

python src/visualize_loss.py \
    --trainer_state_json_path $TRAIN_STATE_JSON_PATH \
    --image_path $IMAGE_PATH