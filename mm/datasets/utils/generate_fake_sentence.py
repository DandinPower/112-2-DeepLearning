import os 
import requests
import json
import pandas as pd
from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def llm_call(prompt: str) -> list[str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o",
        "response_format":{ "type": "json_object" },
        "messages": [
            {
                "role": "user",
                "content": [{
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }

    json_answer = None

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())  
        json_answer = json.loads(response.json().get('choices')[0].get('message').get('content'))
        return json_answer['sentences']
    except Exception as e:
        print(f'Error: {e}, LLM return: {json_answer}')
        return None

def generate_fake_sentence(original_sentences: list[str]) -> list[str]:
    prompt_template="""我會提供你 <NUMBER> 個繁體中文的句子，請幫我重新撰寫對應數量的句子，請符合以下的規定\n0. 綜合所有原句子的所有單詞、文字以及標點符號來組成素材集\n1. 盡量只能使用素材集來撰寫\n2. 盡量最後撰寫完所有句子後有使用到所有素材集的素材 \n3. 不能跟原先的任一句子重複或太過相似\n4. 撰寫的句子間不要重複或太過相似\n5. 句子的長度要原先的句子一樣不能比較多\n6. 撰寫的句子要合乎繁體中文的語法\n7. 使用 json 的格式來回覆，舉例來說 {"sentences": ["句子", "句子", ..., "句子"]}\n以下是提供給你的句子:\n<SENTENCES>"""
    original_sentences_text = ""
    for index, sentence in enumerate(original_sentences):
        original_sentences_text += str(index+1) + ". " + sentence + "\n"
    number = len(original_sentences)
    prompt = prompt_template.replace("<NUMBER>", str(number)).replace("<SENTENCES>", original_sentences_text)
    return llm_call(prompt)

def generate_rewrite_sentence(original_sentences: list[str]) -> list[str]:
    prompt_template="""我會提供你 <NUMBER> 個繁體中文的句子，請幫我重新撰寫對應數量的句子，請符合以下的規定\n0. 改寫原版的句子使其合理通順並換句話說\n1. 句子的長度要原先的句子一樣不能比較多\n2. 撰寫的句子為繁體中文並使用全形符號\n3. 使用 json 的格式來回覆，舉例來說 {"sentences": ["句子", "句子", ..., "句子"]}\n以下是提供給你的句子:\n<SENTENCES>"""
    original_sentences_text = ""
    for index, sentence in enumerate(original_sentences):
        original_sentences_text += str(index+1) + ". " + sentence + "\n"
    number = len(original_sentences)
    prompt = prompt_template.replace("<NUMBER>", str(number)).replace("<SENTENCES>", original_sentences_text)
    return llm_call(prompt)

def parse_original_sentences_by_csv(csv_file: str) -> list[str]:
    df = pd.read_csv(csv_file)
    return df["text"].tolist()

def parse_original_sentences_by_txt(txt_file: str) -> list[str]:
    with open(txt_file, "r") as f:
        return [line.replace("\n", "") for line in f.readlines()]

def get_parse_function(type: str) -> callable:
    if type == "csv":
        return parse_original_sentences_by_csv
    elif type == "txt":
        return parse_original_sentences_by_txt
    else:
        raise ValueError("type should be csv or txt")

def main(args: Namespace) -> None:
    parse_function = get_parse_function(args.type)
    original_sentences = parse_function(args.original_file)

    # clear the output file
    with open(args.output_file, "w") as f:
        pass

    for batch in range(0, len(original_sentences), args.batch_size):
        generated_sentences = generate_fake_sentence(original_sentences[batch:batch+args.batch_size])
        if generated_sentences is not None:
            with open(args.output_file, "a") as f:
                for sentence in generated_sentences:
                    print(f'batch: {batch}, sentence: {sentence}')
                    f.write(sentence + "\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_file", type=str, required=True, help="The file contains the original sentences")
    parser.add_argument("--output_file", type=str, required=True, help="The file contains the generated sentences")
    parser.add_argument("--type", type=str, required=True, help="The type of the original file, csv or txt")
    parser.add_argument("--batch_size", type=int, required=True, help="The batch size of the original sentences")
    args = parser.parse_args()
    main(args)