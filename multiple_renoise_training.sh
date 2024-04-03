EPOCHS=500
UPDATE_FREQ=5
CURRENT_MAX_EPOCH=0

TRAIN_SHELL_FOLDER=espnet/egs2/aishell_test/asr1
TRAIN_RECORD_FOLDER=espnet/egs2/aishell_test/asr1/exp
CONFIG_FILE=espnet/egs2/aishell_test/asr1/conf/train_asr_branchformer.yaml
DATA_PREP_SCRIPT="./run.sh --stage 2 --stop_stage 10"
TRAINING_SCRIPT="./run.sh --stage 11 --stop_stage 11"

rm -rf $TRAIN_RECORD_FOLDER

echo "Add noise"
python add_noise.py
(cd $TRAIN_SHELL_FOLDER && $DATA_PREP_SCRIPT)

for i in $(seq 1 $UPDATE_FREQ $EPOCHS); do
    CURRENT_MAX_EPOCH=$((CURRENT_MAX_EPOCH + UPDATE_FREQ))
    yq eval ".max_epoch = $CURRENT_MAX_EPOCH" $CONFIG_FILE -i
    (cd $TRAIN_SHELL_FOLDER && $TRAINING_SCRIPT)
    echo "End training from epoch $((CURRENT_MAX_EPOCH - UPDATE_FREQ)) to $CURRENT_MAX_EPOCH"
    echo "Add noise"
    python add_noise.py
    (cd $TRAIN_SHELL_FOLDER && $DATA_PREP_SCRIPT)
done