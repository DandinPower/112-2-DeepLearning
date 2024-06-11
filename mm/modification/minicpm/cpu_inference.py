import torch
from PIL import Image
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import random
import numpy as np
import os
from tqdm import tqdm

TEST_IMAGES_FOLDER='/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/test_images'
TEST_RESULTS_PATH='/home/liaw/Desktop/112-2-DeepLearning/mm/predicts/food_soldier_1000_fake3-lora-checkpoint-500.csv'
ADAPTER_PATH='/home/liaw/Desktop/112-2-DeepLearning/mm/MiniCPM-V/finetune/output/food_soldier_1000_fake3-lora-checkpoint-500'
PROMPT = "Extract the subtitles from the following TV drama screenshot and output the subtitle content. Capture only the subtitles that appear at the bottom center of the screen, characterized by white text with a black border. Ignore any other text on the screen, such as titles, descriptions, captions, floating text, or TV station names. Ensure that the generated subtitle content is in Traditional Chinese with full-width punctuation."
SEED = 42

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_model():
    model = AutoPeftModelForCausalLM.from_pretrained(
        # path to the output directory
        ADAPTER_PATH,
        device_map='cpu',  
        trust_remote_code=True  
    )
    return model

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH, 
        trust_remote_code=True
    )
    return tokenizer

def get_subtitle(model, tokenizer, image_path) -> str:
    response = model.chat(
        image=Image.open(image_path).convert("RGB"),
        msgs=[
            {
                "role": "user",
                "content": PROMPT
            }
        ],
        tokenizer=tokenizer
    )
    return response

def main():
    set_seed(SEED)
    set_seed(SEED)
    model = get_model()
    tokenizer = get_tokenizer()
    torch.set_grad_enabled(False)
    model.eval()

    # Write the header of the CSV file
    with open(TEST_RESULTS_PATH, 'w') as f:
        f.write('id,text\n')

    # Iterate through all the test images
    all_files = os.listdir(TEST_IMAGES_FOLDER)
    for file in tqdm(all_files, total=len(all_files)):
        if file.endswith('.jpg'):
            image_path = os.path.join(TEST_IMAGES_FOLDER, file)
            image_id = file.split('.')[0]
            subtitle = get_subtitle(model, tokenizer, image_path)
            print(f'Image: {image_path}, ID: {image_id}, Subtitle: {subtitle}')
            with open(TEST_RESULTS_PATH, 'a') as f:
                f.write(f'{image_id},{subtitle}\n')