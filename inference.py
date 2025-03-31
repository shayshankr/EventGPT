import torch
from transformers import AutoConfig
import argparse
from transformers import AutoTokenizer
from model.EventChatModel import EventChatModel
from common.common import tokenizer_event_token, process_event_data
import numpy as np
from dataset.conversation import conv_templates, prepare_event_prompt
from dataset.constants import EVENT_TOKEN_INDEX, DEFAULT_EVENT_TOKEN, DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN, EVENT_PLACEHOLDER, DEFAULT_EVENT_PATCH_TOKEN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default='eventgpt_v1')
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--context_len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--spatial_temporal_encoder", type=bool, default=True)
    parser.add_argument("--event_frame", type=str, required=True)
  
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = EventChatModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, config=config)

    event_processor = None
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_EVENT_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_EV_START_TOKEN, DEFAULT_EV_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_visual_tower()
    event_processor = vision_tower.event_processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompt = prepare_event_prompt(args.query, args.conv_mode)
    event_image_size, event_tensor = process_event_data(args.event_frame, event_processor, device)
    input_ids = tokenizer_event_token(prompt, tokenizer, EVENT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            event_tensors=event_tensor,
            event_image_sizes=event_image_size,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)