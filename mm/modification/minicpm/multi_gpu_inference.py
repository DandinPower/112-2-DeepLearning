from PIL import Image
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
import random
import numpy as np
import os
from tqdm import tqdm

TEST_IMAGES_FOLDER='/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/test_images'
TEST_RESULTS_PATH='/home/liaw/Desktop/112-2-DeepLearning/mm/predicts/food_soldier_1000_fake3-lora-checkpoint-500.csv'
MERGE_MODEL_PATH='/home/liaw/Desktop/112-2-DeepLearning/mm/MiniCPM-V/finetune/output/food_soldier_1000_fake3-lora-checkpoint-500'
ADAPTER_PATH='/home/liaw/Desktop/112-2-DeepLearning/mm/MiniCPM-V/finetune/output/food_soldier_1000_fake3-lora-checkpoint-500'
PROMPT = "Extract the subtitles from the following TV drama screenshot and output the subtitle content. Capture only the subtitles that appear at the bottom center of the screen, characterized by white text with a black border. Ignore any other text on the screen, such as titles, descriptions, captions, floating text, or TV station names. Ensure that the generated subtitle content is in Traditional Chinese with full-width punctuation."
SEED = 42

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_model():
    max_memory_each_gpu = '10GiB' # Define the maximum memory to use on each gpu, here we suggest using a balanced value, because the weight is not everything, the intermediate activation value also uses GPU memory (10GiB < 16GiB)
    gpu_device_ids = [0, 1] # Define which gpu to use (now we have two GPUs, each has 16GiB memory)
    no_split_module_classes = ["LlamaDecoderLayer"]
    max_memory = {
        device_id: max_memory_each_gpu for device_id in gpu_device_ids
    }
    config = AutoConfig.from_pretrained(
        MERGE_MODEL_PATH, 
        trust_remote_code=True
    )

    with init_empty_weights():
        model = AutoModel.from_config(
            config, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    # Here we want to make sure the input and output layer are all on the first gpu to avoid any modifications to original inference script.
    device_map["llm.model.embed_tokens"] = 0
    device_map["llm.model.layers.0"] = 0
    device_map["llm.lm_head"] = 0
    device_map["vpm"] = 0
    device_map["resampler"] = 0

    print("modified device_map", device_map)

    load_checkpoint_in_model(
        model, 
        MERGE_MODEL_PATH, 
        device_map=device_map)

    model = dispatch_model(
        model, 
        device_map=device_map
    )

def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True
    )

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

if __name__ == '__main__':
    main()