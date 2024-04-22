# 112-2-DeepLearning

## Installation

### Prequisites (Stage1)

1. Ensure your system already install python3, cuda and cudnn

    ```bash
    nvcc -V # need to show the cuda version
    ```

2. Prepare dataset by downloading dataset from kaggle

    - [Dataset](https://www.kaggle.com/competitions/nycu-iass-dl2024-taiwanese-asr/data)

### Install ESPnet (Stage2)

1. Reference: [ESPnet](https://espnet.github.io/espnet/installation.html#step-2-installation-espnet)

2. Installation Step

    ```bash
    # Git clone the repository
    git submodule add https://github.com/espnet/espnet.git # if you first time clone the repository
    git submodule update --init # if not, using this command to clone and update the submodule

    # Install System Dependencies
    sudo apt-get install cmake
    sudo apt-get install sox
    sudo apt-get install flac

    # cd <espnet-root>/tools
    ./setup_venv.sh $(command -v python3)

    # Install espnet
    make

    # Check the installation
    chmod 777 activate_python.sh
    chmod 777 extra_path.sh
    bash -c ". ./activate_python.sh; . ./extra_path.sh; python3 check_install.py"

    # Activate the virtual environment
    . activate_python.sh
    ```

### Run Example Recipe (Optional)

1. Reference: [Recipes using ESPnet2](https://espnet.github.io/espnet/espnet2_tutorial.html)

2. Running egs2/aishell/asr1 recipe

    ```bash
    cd egs2/aishell/asr1
    
    # Download the unique dependencies tools
    cd ../../../tools && make kenlm.done # if you encounter error, you can try to run `make kenlm.done` again

    ./run.sh
    ```
    
    - ensure that you have right config setting in run.sh

    - In aishell original dataset, if i activate `speed perturbation related`, it will cause error in stage 11, showing some encoding error. So I comment out the `speed perturbation related` in `run.sh` file. But maybe it stiil have functionality in future usage.

## Run Homework Dataset

### Transform Dataset (Stage3)

1. using sox to transform original dataset to 16k sample rate

2. modify dataset path in `transform_wav.sh`

3. run the script

    ```bash
    chmod 777 transform_wav.sh
    ./transform_wav.sh
    ```

### Warmup folder structure by adding noise to dataset (Stage4)

#### Step

1. install dependencies

    ```bash
    pip install -r additional_requirements.txt
    ```

2. download noise dataset by running 

    - modify path inside `download_noise.sh`

    - run the script
        ```bash
        chmod 777 download_noise.sh
        ./download_noise.sh
        ```

3. modify dataset path in `add_noise.py`

4. run the script

    ```bash
    python3 add_noise.py
    ```

#### Reference

1. [AudioAugment Library](https://github.com/iver56/audiomentations)
2. [BackgroundNoiseDataset](https://github.com/karolpiczak/ESC-50#download)
3. [ImpulseResponseDataset](http://www.echothief.com/)

### Generate espnet format dataset (Stage5)

1. modify `generate_dataset.py` path

2. run the script

    ```bash
    python3 generate_dataset.py
    ```

### Setup ESPNET for Training (Stage6)

1. put the generated dataset in `espnet/egs2/aishell/asr1` folder

2. go to `espnet/egs2/aishell/asr1` folder

3. modify `run.sh` file

    - modify `token_type=char` into `token_type=word`, otherwise the sequence length will be too long
    - remove `speed perturbation related` in `run.sh` file, because we already modify speed in noise adding process

4. modify `conf/train_asr_branchformer.yaml` for training setting

    - lower the `batch_bins` if you encounter `CUDA Out of memory` error
    - modify `epoch` to higher value if you want to train more epochs

### Train Model without re-noise(Stage7)

1. run the script to train the model

    ```bash
    ./run.sh --stage 2 --stop_stage 11 # because we don't need to run original dataset download and preprocess
    ```

#### Monitor Training Process

1. using tensorboard to monitor the training process

    ```bash
    tensorboard --logdir exp
    ```

#### Resume Training

1. you can run the script to resume training

    ```bash
    ./run.sh --stage 11 --stop_stage 11
    ```

### Run Multiple noise in one training process (Stage7_v2)

1. download dependencies

    ```bash
    sudo snap install yq
    ```

2. modify `multiple_renoise_training.sh` path

3. run the script

    ```bash
    ./multiple_renoise_training.sh
    ```

#### Monitor Training Process

1. monitor the training process by using tensorboard

    ```bash
    tensorboard --logdir espnet/egs2/aishell_test/asr1/exp/
    ```

2.  monitor the training process by using `cat`
    
    ```bash
    cat espnet/egs2/aishell_test/asr1/exp/asr_train_asr_branchformer_raw_zh_word/train.log
    ```

### Decode Model (Stage8)

1. after training, run the script to decode the model

    - [Reference](https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/README_multitask.md#how-to-run)

    - use cpu for inference
        ```bash
        ./run.sh --stage 12 --stop_stage 12 --test_sets test
        ```

    - use gpu for inference
        ```bash
        ./run.sh --stage 12 --stop_stage 12 --test_sets test --gpu_inference true
        ```

2. the result will be in `exp/asr_train_asr_branchformer_raw_zh_word/decode_asr_branchformer_asr_model_valid.acc.ave/test/text`

3. you can run `generate_submission.py` to generate submission file, before running this script, you need to modify the path in `generate_submission.py`

    ```bash
    python3 generate_submission.py
    ```

#### Merge Training and Decoding together

- you can also run whole training and infernce process by running, but using this command will not activate re-noise process

    ```bash
    ./run.sh --stage 2 --stop_stage 12 --test_sets test
    ```

### Run S3prl

- first go to espnet/tools/installers folder

- run the script to install s3prl

    ```bash
    ./install_s3prl.sh
    ```

- go to espnet/egs2/aishell/asr1 folder

- use the `configs\train_asr_conformer_s3prl.yaml` as the training config

    - ensure `extract_feats_in_collect_stats` is true, otherwise the training process will not work
    - open this setting will require more memory, so you need to lower the `nj` in `run.sh` file to lower the number of parallel process

- run the script to train the model

    ```bash
    ./run.sh --stage 2 --stop_stage 11
    ```