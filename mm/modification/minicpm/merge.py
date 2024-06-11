import torch
from peft import AutoPeftModelForCausalLM
import random
import numpy as np

SEED = 42
ADAPER_NAME_OR_PATH = '/home/liaw/Desktop/112-2-DeepLearning/mm/MiniCPM-V/finetune/output/food_soldier_1000_fake3-lora/checkpoint-500'
MERGE_MODEL_SAVE_PATH = '/home/liaw/Desktop/112-2-DeepLearning/mm/MiniCPM-V/finetune/output/food_soldier_1000_fake3-lora-checkpoint-500'

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_peft_model():
    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPER_NAME_OR_PATH,
        device_map='cpu',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    vpm_resampler_embedtokens_weight = torch.load(f"{ADAPER_NAME_OR_PATH}/vpm_resampler_embedtokens.pt")
    msg = model.load_state_dict(vpm_resampler_embedtokens_weight, strict=False)
    return model

def merge_model(model):
    return model.merge_and_unload()

def save_model(model):
    model.save_pretrained("/home/liaw/Desktop/112-2-DeepLearning/mm/MiniCPM-V/finetune/output/food_soldier_1000_fake3-lora-checkpoint-500", safe_serialization=False)

def main():
    set_seed(SEED)
    model = get_peft_model()
    model = merge_model(model)
    save_model(model)

if __name__ == '__main__':
    main()