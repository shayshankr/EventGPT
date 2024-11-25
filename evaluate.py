import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
import re
from typing import Dict, Optional, Sequence, List
import transformers
import os
from model.EventChatModel import EventChatModel
import json
from PIL import Image
import requests
from tqdm import tqdm
from io import BytesIO
import numpy as np
from dataset.conversation import conv_templates, SeparatorStyle
from dataset.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER,DEFAULT_IMAGE_PATCH_TOKEN

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_event_images_list(event_npy, n):
    x, y, p, t = event_npy['x'], event_npy['y'], event_npy['p'], event_npy['t']
    total_events = len(t)
    events_per_image = total_events // n  
    event_image_list = [] 

    for i in range(n):
        start_idx = i * events_per_image
        end_idx = (i + 1) * events_per_image if i < n - 1 else total_events  
        x_part = x[start_idx:end_idx]
        y_part = y[start_idx:end_idx]
        p_part = p[start_idx:end_idx]
        event_img = generate_event_image(x_part, y_part, p_part)
        event_image_list.append(event_img) 
    return event_image_list  

def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def generate_event_image(x, y, p):
    height, width = 480, 640
    event_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    for x_, y_, p_ in zip(x, y, p):
        if p_ == 0:
            event_image[y_, x_] = np.array([0, 0, 255])  
        else:
            event_image[y_, x_] = np.array([255, 0, 0])  
    return event_image

def generate_eventchat_response(model, model_path, query, image_file, event_frame, sptial_temporal=True):
    #config = AutoConfig.from_pretrained("path/EventChat_checkpoints/")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #model = EventChatModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config)

    image_size = [480, 640]

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    vision_tower = model.get_visual_tower()
    image_processor = vision_tower.image_processor
    context_len = 2048
                
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if sptial_temporal:
        event_npy = np.load(event_frame, allow_pickle=True)
        event_npy = np.array(event_npy).item()
        event_img_list = get_event_images_list(event_npy, 3)
        event_list = []
        for event in event_img_list:
            event = image_processor(event, return_tensors='pt')['pixel_values'][0]
            event = event.to(device , dtype=torch.bfloat16)
            event_list.append(event)
        event_tensor = event_list
        print("####")
    else:
        event_npy = np.load(event_frame, allow_pickle=True)
        event_npy = np.array(event_npy).item()
        event_img = generate_event_image(event_npy['x'], event_npy['y'], event_npy['p'])
        event_tensor = image_processor(event_img, return_tensors='pt')['pixel_values']
        event_tensor = event_tensor.to(device , dtype=torch.bfloat16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, 
            images=event_tensor,
            image_sizes=image_size,
            do_sample=True,
            temperature=0.6,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True
        )
    print(output_ids.shape)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

eval_data_path = "path/EventChat/custom_data/instruction_tuning/"
image_dir = "path/EventChat_Datasets/"
model_path = "path/EventChat_checkpoints/"
result_file_path = "path/EventChat/eval/"

results_template = {
    "query": "",
    "pred": "",
    "answer": ""
}

config = AutoConfig.from_pretrained(model_path)
model = EventChatModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config)

with open(eval_data_path, "r") as f:
    data = json.load(f)

if not os.path.exists(result_file_path):
    with open(result_file_path, 'w') as f:
        f.write("[")

with open(result_file_path, 'r+') as f:
    f.seek(0, os.SEEK_END)
    file_size = f.tell()
    if file_size > 1:
        f.seek(file_size - 1)
        f.write(",")

for index, item in enumerate(tqdm(data, desc="Processing items")):
    image_file = item["image"]
    image_path = os.path.join(image_dir, image_file)
    
    query = item['conversations'][0]['value'].split('\n')[1]
    event_frame = image_path.split('.')[0] + '.npy'

    pred = generate_eventchat_response(model, model_path, query, image_path, event_frame, False)

    reslut_template = results_template.copy()
    reslut_template["query"] = query
    reslut_template["pred"] = pred
    reslut_template["answer"] = item['conversations'][1]['value']

    with open(result_file_path, 'a') as f:
        json.dump(reslut_template, f, indent=4)
        f.write(",\n" if index < len(data) - 1 else "\n")

with open(result_file_path, 'a') as f:
    f.write("]")





