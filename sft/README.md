## Installation

1. Python Virtual Environment
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Setup Huggingface CLI
    ```bash
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login # login by your WRITE token
    ```

3. Unsloth Installation (Optional)
    - go to the github repository for your computer configuration
    - [link](https://github.com/unslothai/unsloth)

## Usage

### Zero Shot Evaluation

- Because this task is a reading comprehension task, we can use the zero-shot evaluation to evaluate the model before training.
- So we can analyze the model's performance before training.
- Modify the `zero_shot_evaluation.sh` file to evaluate the model.
- Run the following command to evaluate the model.
    ```bash
    bash zero_shot_evaluation.sh
    ```

### Training

- We use huggingface library and integrate with the deepspeed library to train the model.
- Modify the `training.sh` file to train the model.
    - Set `NPROC_PER_NODE` to the number of GPUs you want to use.
    - `DATASET_NAME_OR_PATH` choose the dataset match with the model.
    - `MODEL_NAME_OR_PATH` choose the model you want to train.
    - `OUTPUT_DIR` choose the output directory.
    - Other hyperparameters can be set by the need you want.
- Run the following command to train the model.
    ```bash
    bash training.sh
    ```

### Visualize Training Result

- After training, we can visualize the training result by specifying the output directory which contain the `trainer_state.json` file.
- Modify the `visualize_loss.sh` file to visualize the training result.
- Run the following command to visualize the training result.
    ```bash
    bash visualize_loss.sh
    ```

### Inference

- After training, we can use the model to predict the answer for the question.
- Modify the `inference.sh` file to predict the answer.
    - Set `ADAPTER_NAME_OR_PATH` to the path of the previous training output directory or the huggingface model name.
    - Set `MODEL_NAME_OR_PATH` to the path pretrained model which is used to train the adapter.
    - Other hyperparameters can be set by the need you want.
    - You can enable `--do_valid` to evaluate the model on the validation set.
    - You can enable `--do_test` to evaluate the model on the test set and save the result to the csv file.
    - You can enable `--verbose` to print the model prediction.
- Run the following command to predict the answer.
    ```bash
    bash inference.sh
    ```

### Vote

- After inference, you may want to ensemble the model prediction.
- We provide the voting script to ensemble the model prediction.
- Modify `vote.sh` file to specify the model prediction file.
- Run the following command to ensemble the model prediction.
    ```bash
    bash vote.sh
    ```