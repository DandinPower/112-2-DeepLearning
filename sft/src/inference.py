import re
import os

from argparse import ArgumentParser, Namespace
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm


MODEL_TYPE_CHOISES = ["MediaTek-Research/Breeze-7B-Instruct-v1_0"]


@dataclass
class TestData:
    id: str
    text: str
    answer: int


@dataclass
class ValidData:
    text: str
    output: int
    answer: int


def print_verbose(message: str, verbose: bool):
    if verbose:
        print(message)


def get_model_and_tokenizer(model_name_or_path: str, adapter_name_or_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype="auto", device_map="auto")
    model.load_adapter(adapter_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def get_valid_input_text(tokenizer, data: dict) -> str:
    chat = [
        {"role": "user", "content": data["instruction"]},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False)


def write_test_csv(path: str, test_datas: list[TestData]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        f.write("id,answer\n")
        for test_data in test_datas:
            if test_data.answer is None:
                f.write(f"{test_data.id},{-1}\n")
            else:
                f.write(f"{test_data.id},{test_data.answer}\n")


class FindAnswerOutputParser:
    def __init__(self, model_type: str):
        assert model_type in MODEL_TYPE_CHOISES, "Invalid model_type"
        self.model_type = model_type

    def parse(self, model_output: str) -> str:
        if self.model_type == "MediaTek-Research/Breeze-7B-Instruct-v1_0":
            return model_output.split("/INST]")[1]
        else:
            raise NotImplementedError


def test(model, tokenizer, args: Namespace):
    find_answer_output_parser = FindAnswerOutputParser(args.model_name_or_path)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    dataset: Dataset = load_dataset(
        args.dataset_name_or_path, split=args.test_split)

    test_datas: list[TestData] = []

    for data in dataset:
        test_datas.append(TestData(data["id"], data["text"], None, None))

    not_found = 0

    for test_data in tqdm(test_datas):
        outputs = pipe(test_data.text, max_new_tokens=args.max_new_tokens, do_sample=True,
                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        model_output = outputs[0]["generated_text"]
        answer_output = find_answer_output_parser.parse(model_output)

        answer = re.findall(r"\d", answer_output)
        if len(answer) == 0:
            not_found += 1
        else:
            test_data.answer = int(answer[0])
        print_verbose(
            f"NotFound: {not_found}, ID: {test_data.id}, AnswerOutput: {answer_output}, ModelOutput: {model_output}", args.verbose)

    write_test_csv(args.output_csv_path, test_datas)


def valid(model, tokenizer, args: Namespace):
    find_answer_output_parser = FindAnswerOutputParser(args.model_name_or_path)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    dataset: Dataset = load_dataset(
        args.dataset_name_or_path, split=args.valid_split)

    valid_datas: list[ValidData] = []

    for data in dataset:
        text = get_valid_input_text(tokenizer, data)
        valid_datas.append(ValidData(text, int(data["output"]), None))

    not_found = 0

    for valid_data in tqdm(valid_datas):
        outputs = pipe(valid_data.text, max_new_tokens=args.max_new_tokens, do_sample=True,
                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        model_output = outputs[0]["generated_text"]
        answer_output = find_answer_output_parser.parse(model_output)

        answer = re.findall(r"\d", answer_output)
        if len(answer) == 0:
            not_found += 1
            valid_data.answer = -1

        else:
            valid_data.answer = int(answer[0])

        print_verbose(
            f"NotFound: {not_found}, AnswerOutput: {answer_output}, ModelOutput: {model_output}", args.verbose)

    correct = 0
    for valid_data in valid_datas:
        if valid_data.answer == valid_data.output:
            correct += 1

    length = len(valid_datas)
    print(f"Valid Accuracy: {correct / length}")


def main(args: Namespace):
    model, tokenizer = get_model_and_tokenizer(
        args.model_name_or_path, args.adapter_name_or_path)

    if args.do_valid:
        valid(model, tokenizer, args)

    if args.do_test:
        test(model, tokenizer, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--test_split", type=str, required=True)
    parser.add_argument("--valid_split", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--do_valid", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    args = parser.parse_args()
    main(args)
