import re

from argparse import ArgumentParser, Namespace
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm


@dataclass
class Data:
    id: int
    text: str
    output: str
    answer: int


def get_model_and_tokenizer(model_name_or_path: str, adapter_name_or_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype="auto", device_map="auto")
    model.load_adapter(adapter_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


MODEL_TYPE_CHOISES = ["MediaTek-Research/Breeze-7B-Instruct-v1_0",
                      "meta-llama/Meta-Llama-3-8B-Instruct", "TAIDE-LX-7B-Chat", "google/gemma-1.1-7b-it"]


class FindAnswerOutputParser:
    def __init__(self, model_type: str):
        assert model_type in MODEL_TYPE_CHOISES, "Invalid model_type"
        self.model_type = model_type

    def parse(self, model_output: str) -> str:
        if self.model_type == "MediaTek-Research/Breeze-7B-Instruct-v1_0":
            return model_output.split("/INST]")[1]
        else:
            raise NotImplementedError


def main(args: Namespace):

    find_answer_output_parser = FindAnswerOutputParser(args.model_name_or_path)

    model, tokenizer = get_model_and_tokenizer(
        args.model_name_or_path, args.adapter_name_or_path)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    dataset: Dataset = load_dataset(
        args.dataset_name_or_path, split=args.split)

    test_datas: list[Data] = []

    for data in dataset:
        test_datas.append(Data(data["id"], data["text"], None, None))

    not_found = 0

    for test_data in tqdm(test_datas):
        outputs = pipe(test_data.text, max_new_tokens=5, do_sample=True,
                       temperature=0.7, top_k=10, top_p=0.95)
        model_output = outputs[0]["generated_text"]
        test_data.output = model_output
        answer_output = find_answer_output_parser.parse(model_output)

        answer = re.findall(r"\d", answer_output)
        if len(answer) == 0:
            not_found += 1
            print(
                f"NotFound: {not_found}, ID: {test_data.id}, AnswerOutput: {answer_output}, ModelOutput: {model_output}")
            continue
        else:
            print(
                f"NotFound: {not_found}, ID: {test_data.id}, AnswerOutput: {answer_output}, ModelOutput: {model_output}")

        test_data.answer = int(answer[0])

    with open(args.output_csv_path, "w") as f:
        f.write("id,answer\n")
        for test_data in test_datas:
            if test_data.answer is None:
                f.write(f"{test_data.id},{-1}\n")
            else:
                f.write(f"{test_data.id},{test_data.answer}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
