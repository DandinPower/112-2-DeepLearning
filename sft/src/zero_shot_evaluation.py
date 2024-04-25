# Load model directly
import os
import torch

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm


@dataclass
class Data:
    id: str
    instruction: str
    output: str
    model_output: str


def prompt_format(instruction, tokenizer):
    messages = [
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    return text


def write_tsv(data: list[Data], path: str):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('id\toutput\tmodel_output\tinstruction\n')
        for d in data:
            f.write(f'{d.id}\t{d.output}\t{d.model_output}\t{d.instruction}\n')


def get_model_name(model_name: str) -> str:
    return model_name.split('/')[-1]


def only_get_model_generated_text(original_text: str, model_generate_text: str) -> str:
    return model_generate_text[len(original_text):]


def get_model_pipeline(model_name: str):
    if model_name == "taide/TAIDE-LX-7B-Chat":
        return pipeline(model=model_name, device_map="auto", use_fast=False)
    elif model_name == "microsoft/Phi-3-mini-4k-instruct":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    elif model_name == "MediaTek-Research/Breeze-7B-Instruct-v1_0":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype="auto",
            # attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipeline(model=model_name, device_map="auto")


def main(args: Namespace):
    torch.random.manual_seed(42)
    print(f'Loading model: {args.model_name}')
    pipe = get_model_pipeline(args.model_name)

    print(f'Loading dataset: {args.dataset_name}')
    dataset: Dataset = load_dataset(args.dataset_name, split=args.split)

    data_list: list[Data] = []

    print(f'Sampling {args.sample_size} examples')
    for i in tqdm(range(args.sample_size)):
        instruction = clean_text(dataset["instruction"][i])
        prompt = prompt_format(instruction, pipe.tokenizer)
        outputs = pipe(prompt, max_new_tokens=128, do_sample=True,
                       temperature=0.7, top_k=50, top_p=0.95)
        model_output = clean_text(outputs[0]["generated_text"])
        data_list.append(Data(
            id=dataset["id"][i],
            instruction=instruction,
            output=dataset["output"][i],
            model_output=only_get_model_generated_text(
                prompt, model_output)
        ))

    write_tsv(
        data_list, f'{args.output_dir}/sample_{args.sample_size}/{get_model_name(args.model_name)}_{args.sample_size}.tsv')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
