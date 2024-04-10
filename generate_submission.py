import os

DECODE_TEXT_PATH = "espnet/egs2/aishell_test/asr1/exp/asr_train_asr_conformer_s3prl_raw_zh_char/decode_asr_branchformer_asr_model_valid.acc.ave/test/text"
SUBMISSION_PATH = "output/submission/conformer_hubert_base.csv"

def check_path():
    if not os.path.exists(DECODE_TEXT_PATH):
        raise FileNotFoundError(f"File not found: {DECODE_TEXT_PATH}")

    if not os.path.exists(os.path.dirname(SUBMISSION_PATH)):
        os.makedirs(os.path.dirname(SUBMISSION_PATH))

def main():
    check_path()

    with open(DECODE_TEXT_PATH, "r") as f:
        lines = f.readlines()
    
    with open(SUBMISSION_PATH, "w") as f:
        f.write("id,sentence\n")
        for i, line in enumerate(lines):
            id = line.split()[0]
            line = " ".join(line.split()[1:])
            f.write(f"{id},{line.strip()}\n")

if __name__ == "__main__":
    main()