import pandas as pd
import os
from dataclasses import dataclass
import random

RANDOM_SEED = 0

RAW_TRAIN_DATASET_FOLDER = 'datasets/noisy/train'
RAW_TEST_DATASET_FOLDER = 'datasets/transform/test'
ANSWER_FILE = 'datasets/downloads/train/train-toneless.csv'
SPK_ID = 'S01'

FINAL_FOLDER = 'datasets/data'
FINAL_TRAIN_FOLDER = f'{FINAL_FOLDER}/train'
FINAL_VALID_FOLDER = f'{FINAL_FOLDER}/dev'
FINAL_TEST_FOLDER = f'{FINAL_FOLDER}/test'

@dataclass
class WavData:
    wav_path: str
    id: str
    text: str

def data_augmentation():
    pass

def get_train_id(index: str, fill_number: int) -> str:
    index = str(index).zfill(fill_number)  # Fill with zeros if index is less than 4 digits
    return f'{index}' 

def get_train_wav_path(id: str) -> str:    
    return os.path.abspath(f'{RAW_TRAIN_DATASET_FOLDER}/{id}.wav')

def load_train_dataset() -> list[WavData]:
    datas:list[WavData] = []
    with open(ANSWER_FILE, 'r+') as csvfile:
        for i, line in enumerate(csvfile):
            if i == 0: continue
            index, text = line.split(',')
            text = text.replace('\n', '')
            id = get_train_id(index, 0)
            wav_path = get_train_wav_path(id)
            datas.append(WavData(wav_path, id, text))

    return datas

def load_predict_dataset(folder: str) -> list[WavData]:
    # traverse folder to get all wav files
    datas: list[WavData] = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav'):
                id = file.split('.')[0]
                wav_path = os.path.abspath(f'{folder}/{file}')
                datas.append(WavData(wav_path, id, ''))

    datas.sort(key=lambda x: x.id)
    
    return datas

def generate_token_list(datas: list[WavData], path: str):
    dir_path = os.path.dirname(path)
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

    tokens = set()
    for data in datas:
        for token in data.text.split(' '):
            tokens.add(token)
    
    with open(path, 'w+') as file:
        for token in tokens:
            file.write(f"{token}\n")

def generate_spk2utt(datas: list[WavData], path: str):
    dir_path = os.path.dirname(path)
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

    text = f"{SPK_ID}"
    for data in datas:
        text += f" {data.id}"
    
    with open(path, 'w+') as file:
        file.write(text)

def generate_utt2spk(datas: list[WavData], path: str):
    dir_path = os.path.dirname(path)
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)
    
    text = ""
    for data in datas:
        text += f"{data.id} {SPK_ID}\n"
    
    with open(path, 'w+') as file:
        file.write(text)

def generate_text(datas: list[WavData], path: str):
    dir_path = os.path.dirname(path)
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)
    
    text = ""
    for data in datas:
        text += f"{data.id} {data.text}\n"
    
    with open(path, 'w+') as file:
        file.write(text)

def generate_wav_scp(datas: list[WavData], path: str):
    dir_path = os.path.dirname(path)
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)
    
    text = ""
    for data in datas:
        text += f"{data.id} {data.wav_path}\n"
    
    with open(path, 'w+') as file:
        file.write(text)

def train_valid_split(datas: list[WavData], train_ratio: float, valid_ratio: float, random_seed: int = 42):
    assert train_ratio + valid_ratio == 1.0, "train, valid ratio must sum up to 1.0"
    random.seed(random_seed)
    random.shuffle(datas)
    train_size = int(len(datas) * train_ratio)
    valid_size = len(datas) - train_size
    assert train_size + valid_size == len(datas), "train, valid size must sum up to total data size"
    train, valid = datas[:train_size], datas[train_size:]
    train.sort(key=lambda x: x.id)
    valid.sort(key=lambda x: x.id)
    return train, valid

def main():
    # remove data folder
    if os.path.exists(FINAL_FOLDER):
        os.system(f'rm -rf {FINAL_FOLDER}')
    
    train_datas: list[WavData] = load_train_dataset()
    train, valid = train_valid_split(train_datas, 0.85, 0.15, RANDOM_SEED)
    
    generate_spk2utt(train, f'{FINAL_TRAIN_FOLDER}/spk2utt')
    generate_spk2utt(valid, f'{FINAL_VALID_FOLDER}/spk2utt')

    generate_utt2spk(train, f'{FINAL_TRAIN_FOLDER}/utt2spk')
    generate_utt2spk(valid, f'{FINAL_VALID_FOLDER}/utt2spk') 

    generate_text(train, f'{FINAL_TRAIN_FOLDER}/text')
    generate_text(valid, f'{FINAL_VALID_FOLDER}/text')

    generate_wav_scp(train, f'{FINAL_TRAIN_FOLDER}/wav.scp')
    generate_wav_scp(valid, f'{FINAL_VALID_FOLDER}/wav.scp')

    test_datas: list[WavData] = load_predict_dataset(RAW_TEST_DATASET_FOLDER)
    
    generate_spk2utt(test_datas, f'{FINAL_TEST_FOLDER}/spk2utt')
    generate_utt2spk(test_datas, f'{FINAL_TEST_FOLDER}/utt2spk')
    generate_wav_scp(test_datas, f'{FINAL_TEST_FOLDER}/wav.scp')

if __name__ == "__main__":
    main()