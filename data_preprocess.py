import pandas as pd
import os
from dataclasses import dataclass
import random

RAW_TRAIN_DATASET_FOLDER = 'transform/train'
RAW_TEST_DATASET_FOLDER = 'transform/test'
ANSWER_FILE = 'downloads/train/train-toneless.csv'
SPK_ID = 'S01'

FINAL_TRAIN_FOLDER = 'data/train'
FINAL_VALID_FOLDER = 'data/dev'
FINAL_TEST_FOLDER = 'data/test'

@dataclass
class WavData:
    wav_path: str
    id: str
    text: str

def data_augmentation():
    pass

def get_id(index: str):
    index = str(index).zfill(4)  # Fill with zeros if index is less than 4 digits
    return f'B{index}' 

def get_wav_path(id: str):    
    return os.path.abspath(f'{RAW_TRAIN_DATASET_FOLDER}/{id}.wav')

def load_text_data():
    datas:list[WavData] = []
    with open(ANSWER_FILE, 'r+') as csvfile:
        for i, line in enumerate(csvfile):
            if i == 0: continue
            index, text = line.split(',')
            text = text.replace('\n', '')
            id = get_id(index)
            wav_path = get_wav_path(id)
            datas.append(WavData(wav_path, id, text))
    
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

def train_valid_test_split(datas: list[WavData], train_ratio: float, valid_ratio: float, test_ratio: float, random_seed: int = 42):
    assert train_ratio + valid_ratio + test_ratio == 1.0, "train, valid, test ratio must sum up to 1.0"
    random.seed(random_seed)
    random.shuffle(datas)
    train_size = int(len(datas) * train_ratio)
    valid_size = int(len(datas) * valid_ratio)
    test_size = len(datas) - train_size - valid_size
    assert train_size + valid_size + test_size == len(datas), "train, valid, test size must sum up to total data size"
    train, valid, test = datas[:train_size], datas[train_size:train_size+valid_size], datas[train_size+valid_size:]
    train.sort(key=lambda x: x.id)
    valid.sort(key=lambda x: x.id)
    test.sort(key=lambda x: x.id)
    return train, valid, test

def main():
    # remove data folder
    if os.path.exists('data'):
        os.system('rm -rf data')
    datas: list[WavData] = load_text_data()
    
    generate_token_list(datas, 'data/zh_token_list/char/tokens.txt')
    
    train, valid, test = train_valid_test_split(datas, 0.8, 0.1, 0.1)
    
    generate_spk2utt(train, f'{FINAL_TRAIN_FOLDER}/spk2utt')
    generate_spk2utt(valid, f'{FINAL_VALID_FOLDER}/spk2utt')
    generate_spk2utt(test, f'{FINAL_TEST_FOLDER}/spk2utt')

    generate_utt2spk(train, f'{FINAL_TRAIN_FOLDER}/utt2spk')
    generate_utt2spk(valid, f'{FINAL_VALID_FOLDER}/utt2spk')
    generate_utt2spk(test, f'{FINAL_TEST_FOLDER}/utt2spk') 

    generate_text(train, f'{FINAL_TRAIN_FOLDER}/text')
    generate_text(valid, f'{FINAL_VALID_FOLDER}/text')
    generate_text(test, f'{FINAL_TEST_FOLDER}/text')

    generate_wav_scp(train, f'{FINAL_TRAIN_FOLDER}/wav.scp')
    generate_wav_scp(valid, f'{FINAL_VALID_FOLDER}/wav.scp')
    generate_wav_scp(test, f'{FINAL_TEST_FOLDER}/wav.scp')

if __name__ == "__main__":
    main()

# stage 1 data augmentation

# load text data

# prepare token list

# split data into train, valid and test

# create espnet format