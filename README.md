# 112-2-DeepLearning

## Installation

### Prequisites

1. ensure your system already install python3, cuda and cudnn

### Install ESPnet

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

### Run Example Recipe

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

### New Task

1. Reference: [New Task](https://espnet.github.io/espnet/notebook/espnet2_new_task_tutorial_CMU_11751_18781_Fall2022.html)