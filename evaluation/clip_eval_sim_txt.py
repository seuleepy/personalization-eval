"""
Evaluation script for computing similarity between generated images and text prompts using CLIP model.
Only the average similarity per text prompt and the overall average across all prompts are recorded from gen_img_path; individual sample scores are not saved.

This script:
- Load generated images and text prompts.
- Compute the similarity score between the generated images and the text prompts using CLIP model.

Output (JSON format):
{
    "gen_img_path": "path/to/generated/images",
    "model": "RN50",
    "text_scores": {
        "0": 0.1234,
        "1": 0.5678,
        ...
        },
    "avg_score": 0.1234
}
"""

import os
import json
import argparse
import glob
from tqdm.auto import tqdm
from PIL import Image

import torch
import clip

import sys

from utils.constant import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen_img_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        required=True,
        choices=[
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "RN50x64",
            "ViT-B/32",
            "ViT-B/16",
            "ViT-L/14",
            "ViT-L/14@336px",
        ],
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    return args


def load_and_preprocess(image_path_list, preprocess, device):
    images = []
    for path in image_path_list:
        image = Image.open(path)
        image = preprocess(image).unsqueeze(0).to(device)
        images.append(image)
        return torch.cat(images)


def append_to_json(output_path, new_data):
    MAX_BRACE_COUNT = 30
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_data = json.load(f)
            open_braces = existing_data.count("{")
            close_braces = existing_data.count("}")
            if open_braces != close_braces:
                raise ValueError("The existing json file is not properly formatted.")
            if open_braces >= MAX_BRACE_COUNT:
                raise ValueError("The existing json file has too many braces.")
    else:
        existing_data = []

    existing_data.append(new_data)

    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=4)


def main(args):

    # Create output dir if it does not exist
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 Created directory: {output_dir}")

    # Prepare text prompts
    file_name = os.path.basename(args.gen_img_path)
    class_type = TARGET2CLASS_MAPPING[file_name]
    is_object = CLASS2OBJECT_MAPPING[class_type]
    if is_object:
        prompt_ls = OBJECT_PROMPT_LIST
    else:
        prompt_ls = LIVE_PROMPT_LIST
    prompt_ls = list(enumerate(prompt_ls))
    del file_name, is_object

    # Load clip model
    clip_model, preprocess = clip.load(args.clip_model, device=args.device)

    text_scores = {}

    # Extract the text features
    with torch.no_grad():
        for txt_idx, txt in prompt_ls:
            txt = txt.format(f"{class_type}")
            token = clip.tokenize(txt).to(args.device)
            txt_ftr = clip_model.encode_text(token)
            txt_ftr /= txt_ftr.norm(dim=-1, keepdim=True)
            del txt, token
            torch.cuda.empty_cache()

            # Extract the generated images features corresponding to the text
            gen_img_path = f"{args.gen_img_path}/prompt_{txt_idx:02d}*"
            gen_img_path_ls = glob.glob(gen_img_path)
            gen_img = load_and_preprocess(gen_img_path_ls, preprocess, args.device)
            gen_ftr = clip_model.encode_image(gen_img)
            gen_ftr /= gen_ftr.norm(dim=-1, keepdim=True)
            del gen_img_path, gen_img_path_ls, gen_img
            torch.cuda.empty_cache()

            sim = (txt_ftr @ gen_ftr.T).squeeze().mean().item()
            text_scores[txt_idx] = sim
            del sim, txt_ftr, gen_ftr
            torch.cuda.empty_cache()

    avg_score = sum(text_scores.values()) / len(text_scores)
    result = {
        "gen_img_path": args.gen_img_path,
        "model": args.clip_model,
        "text_scores": text_scores,
        "avg_score": avg_score,
    }
    append_to_json(args.output_path, result)


if __name__ == "__main__":
    args = parse_args()
    main(args)
