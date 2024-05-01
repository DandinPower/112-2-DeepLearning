# Whisper-Finetuning

## System Prerequisites

1. ffmpeg
    ```bash
    sudo apt-get install ffmpeg
    ```
2. sox
    ```bash
    sudo apt-get install sox
    ```

## Installation

1. Following the instructions
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login # use your huggingface account WRITE token
    ```

## Training

1. Modify config of `training.sh`

    - `MODEL_NAME_OR_PATH` : The pretrained model name you want to finetune.
    - `DATASET_NAME` : The dataset name you want to finetune.
    - `MAX_STEPS` : The maximum steps of training.
    - `PER_DEVICE_TRAIN_BATCH_SIZE` : The batch size of training.
    - `PER_DEVICE_EVAL_BATCH_SIZE` : The batch size of evaluation.
    - `LEARNING_RATE` : The learning rate of training.
    - `WARMUP_STEPS` : The warmup steps of training.
    - `EVAL_STEPS` : The evaluation steps of training.
    - `SAVE_STEPS` : The model saving steps of training.
    - `GENERATION_MAX_LENGTH` : The maximum length of text generation.
    - `MODEL_INDEX_NAME` : The model name you want to push to huggingface hub.

2. Run the following command to train the model
    ```bash
    bash training.sh
    ```

## Decoding

1. Modify config of `inference.sh`

    - `MODEL_NAME` : The model name you want to decode, which is the same as the model you push to huggingface hub.
    - `OUTPUT_PATH` : The output path of the decoded text.

2. Run the following command to decode the model
    ```bash
    bash inference.sh
    ```

## Reference

1. [openai whisper](https://github.com/openai/whisper)
2. [Whisper Finetuning event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)