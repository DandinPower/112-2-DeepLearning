import os
import json
import random

from argparse import ArgumentParser, Namespace

PROMPT = "Extract the subtitles from the following TV drama screenshot and output the subtitle content. Capture only the subtitles that appear at the bottom center of the screen, characterized by white text with a black border. Ignore any other text on the screen, such as titles, descriptions, captions, floating text, or TV station names. Ensure that the generated subtitle content is in Traditional Chinese with full-width punctuation."

def set_random_seed(seed: int):
    random.seed(seed)

def format_prompt_by_model_type(model_type: str, prompt: str) -> str:
    if model_type == 'minicpm':
        return prompt
    elif model_type == 'tinyllava':
        return f'<image>\n{prompt}'
    else:
        raise ValueError(f'Invalid model type: {model_type}')

def format_conversations_by_model_type(model_type: str, prompt: str, subtitle: str) -> list[dict]:
    if model_type == 'minicpm':
        conversation = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": subtitle
            }
        ]
        return conversation
    elif model_type == 'tinyllava':
        conversation = [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": subtitle
            }
        ]
        return conversation
    else:
        raise ValueError(f'Invalid model type: {model_type}')

def main(args: Namespace) -> None:
    set_random_seed(args.seed)

    # Read the subtitle data from the CSV file
    subtitles = {}
    with open(args.subtitles_file, 'r', encoding='utf-8') as file:
        for line in file:
            video_id, subtitle = line.strip().split('\t')
            subtitles[video_id] = subtitle

    # Create the dataset
    dataset = []
    prompt = format_prompt_by_model_type(args.model_type, PROMPT)
    for filename in os.listdir(args.images_folder):
        if filename.endswith('.jpg'):
            video_id = filename
            image_path = f'{args.images_folder}/{filename}'
            
            if video_id in subtitles:
                subtitle = subtitles[video_id]

                conversation = format_conversations_by_model_type(args.model_type, prompt, subtitle)
                
                data_entry = {
                    "id": video_id,
                    "image": image_path,
                    "conversations": conversation
                }
                
                dataset.append(data_entry)

    # Shuffle the dataset
    random.shuffle(dataset)

    # Split the dataset into training and validation sets
    split_index = int(len(dataset) * args.train_rate)
    train_dataset = dataset[:split_index]
    validation_dataset = dataset[split_index:]

    # Write the training dataset to a JSON file
    with open(args.train_datasets_file, 'w', encoding='utf-8') as file:
        json.dump(train_dataset, file, ensure_ascii=False, indent=4)

    print(f'Training dataset has been written to {args.train_datasets_file}')

    # Write the validation dataset to a JSON file
    with open(args.validation_datasets_file, 'w', encoding='utf-8') as file:
        json.dump(validation_dataset, file, ensure_ascii=False, indent=4)

    print(f'Validation dataset has been written to {args.validation_datasets_file}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--train_rate", type=float, required=True)
    parser.add_argument("--images_folder", type=str, required=True)
    parser.add_argument("--subtitles_file", type=str, required=True)
    parser.add_argument("--train_datasets_file", type=str, required=True)
    parser.add_argument("--validation_datasets_file", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(args)

MODEL_TYPE = 'minicpm'  # 'minicpm' or 'tinyllava'
TRAIN_RATE = 0.8
SEED = 42
IMAGES_FOLDER = '/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/synth_images'
SUBTITLES_FILE = '/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/synth_images_subtitles.txt'
TRAIN_DATASETS_FILE = '/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/json_datasets/cpm_train.json'
VALIDATION_DATASETS_FILE = '/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/json_datasets/cpm_validation.json'