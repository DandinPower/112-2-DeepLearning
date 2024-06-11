import base64
import requests
import json
import csv
import os
from argparse import ArgumentParser, Namespace
from dotenv import load_dotenv

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_transcript(image_path):
    base64_image = encode_image(image_path)

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
                        "text": "請將以下電視劇截圖中的中文字幕提取出來。請勿包含出現在其他地方的文字，如標題、或任何漂浮在畫面中的其他文字、電視台名稱等。請確保使用繁體中文撰寫、標點符號使用全形符號(，。·)，並以 JSON 格式回答。例如：{\n\"subtitle\": \"這是一個簡單的例子\"\n}"
                    },{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        json_answer = json.loads(response.json().get('choices')[0].get('message').get('content'))
        return json_answer['subtitle']
    except Exception as e:
        return "ERROR!"

def main(args: Namespace):
    with open(args.output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['image_name', 'transcript']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for filename in os.listdir(args.input_directory):
            if filename.endswith('.jpg'):
                image_path = os.path.join(args.input_directory, filename)
                transcript = get_transcript(image_path)
                writer.writerow({'image_name': filename, 'transcript': transcript})
                print(f"Processed {filename}, transcript: {transcript}")

if __name__ == "__main__":
    load_dotenv()
    parser = ArgumentParser()
    parser.add_argument("--input_directory", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    main(parser.parse_args())