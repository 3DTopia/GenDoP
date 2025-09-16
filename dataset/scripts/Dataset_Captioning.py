import os
import base64
import random
from openai import OpenAI
import json
import numpy as np
from time import sleep
from tqdm import tqdm

data_dir = "./DATA"
api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4_v(user_prompt, user_img_path, max_tokens=700):
    # global conversation_history
    base64_image = encode_image(user_img_path)
    conversation_history = [({"role": "user",
                                 "content": [
                                     {"type": "text", "text": user_prompt},
                                     {"type": "image_url",
                                      "image_url": {
                                          "url": f"data:image/jpeg;base64,{base64_image}"
                                      }
                                      }
                                 ]
                                 }
                                )]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        max_tokens=max_tokens,
    )
    return response


def show_content(response):
    print(response.choices[0].message.content)

def save_content(response, file):
    with open(file, 'w') as f:
        f.write(response.choices[0].message.content)

def single_test(foldername, prompt_text, name):
    image_path = f"{data_dir}/Captions/{name}.png"
    caption_path = f"{foldername}/{name}_caption.txt"
    relationship_path = f"{foldername}/{name}_relationship.txt"

    if os.path.exists(relationship_path):
        print(f"Skipping {name}...")
        return
    
    with open(caption_path, 'r') as f:
        caption = f.read().strip()
    
    prompt_text = prompt_text + "\n\nMovement: " + caption
    for k in range(3):
        try:
            response = call_gpt4_v(prompt_text, image_path)
            para_len = response.choices[0].message.content.split("\n")
            assert len(para_len) == 3, f"\nLen Error: {para_len}, {response.choices[0].message.content}\n"
            break
        except Exception as e:
            print("Error: ", e)
            continue
    # show_content(response)
    save_content(response, relationship_path)

if __name__ == "__main__":
    prompt_path = "./scripts/configs/captioning/llm/relationship+image.json"
    prompts = json.load(open(prompt_path, 'r'))
    prompt_text = prompts['context'] + prompts['instruction'] + prompts['constraint'] + prompts['format']
    tag_dir = f"{data_dir}/Tagging/cam_segments"
    for folder in tqdm(sorted(os.listdir(tag_dir))):
        folder_path = os.path.join(tag_dir, folder)
        files = os.listdir(folder_path)
        txts = [f for f in files if f.lower().endswith('_caption.txt')]
        for txt in sorted(txts):
            name = f"{folder}/{txt.replace('_caption.txt', '')}"
            relationship_path = f"{tag_dir}/{name}_relationship.txt"
            if not os.path.exists(relationship_path):
                print(f"\nProcessing {name}...")
                single_test(tag_dir, prompt_text, name)