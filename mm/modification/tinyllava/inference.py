
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

import torch

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

OUTPUT_FILE="tinyllava_images_latest_sentence_v3_total_1000_epoch_5-lora.csv"
MODEL_PATH = "/home/liaw/Desktop/112-2-DeepLearning/mm/TinyLLaVA_Factory/exp/tinyllava_images_latest_sentence_v3_total_1000_epoch_5-lora"
IMAGE_FOLDER= "/home/liaw/Desktop/112-2-DeepLearning/mm/datasets/assets/test_images"
TEXT_PROMPT = "<image>\nExtract the subtitles from the following TV drama screenshot and output the subtitle content. Capture only the subtitles that appear at the bottom center of the screen, characterized by white text with a black border. Ignore any other text on the screen, such as titles, descriptions, captions, floating text, or TV station names. Ensure that the generated subtitle content is in Traditional Chinese with full-width punctuation."
CONV_MODE = "phi" # or llama, gemma, etc
SEP = ","
TEMPERATURE = 0
TOP_P = None
NUM_BEAMS = 1
MAX_NEW_TOKENS = 512

def image_parser(image_file):
    out = image_file.split(SEP)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def main():
    disable_torch_init()
    model, tokenizer, image_processor, context_len = load_pretrained_model(MODEL_PATH)

    tokenizer = model.tokenizer
    image_processor = model.vision_tower._image_processor
    text_processor = TextPreprocess(tokenizer, CONV_MODE)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    model.cuda()

    def eval(image_file) -> str:
        qs = TEXT_PROMPT
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        msg = Message()
        msg.add_message(qs)
        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        prompt = result['prompt']
        input_ids = input_ids.unsqueeze(0).cuda()
        image_files = image_parser(image_file)
        images = load_images(image_files)[0]
        images_tensor = image_processor(images)
        images_tensor = images_tensor.unsqueeze(0).half().cuda()
        stop_str = text_processor.template.separator.apply()[1]
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if TEMPERATURE > 0 else False,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_beams=NUM_BEAMS,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs

    output_file = OUTPUT_FILE
    with open(output_file, 'w') as f:
        f.write('id,text\n')

    all_files = os.listdir(IMAGE_FOLDER)
    for file in tqdm(all_files, total=len(all_files)):
        if file.endswith('.jpg'):
            image_path = os.path.join(IMAGE_FOLDER, file)
            image_id = file.split('.')[0]
            subtitle = eval(image_path)
            print(f'Image: {image_path}, ID: {image_id}, Subtitle: {subtitle}')
            with open(output_file, 'a') as f:
                f.write(f'{image_id},{subtitle}\n')

if __name__ == "__main__":
    main()