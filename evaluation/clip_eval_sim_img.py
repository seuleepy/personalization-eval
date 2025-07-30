"""
Evaluation script for computing similarity between generated images and a reference image using CLIP model.
Only the average similarity score per gen_img_path is recorded in the output; individual sample scores are not saved.

This script:
- Load generated images and a reference image.
- Compute the similarity score between the generated images and the reference image using CLIP model.

Output (JSON format):
{
    "gen_img_path": "path/to/generated/images",
    "model": "RN50",
    "real_img_file": "reference_image.jpg",
    "batch": 50,
    "score": 0.1234,
}
"""

import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import json
import random

import torch
import clip
from accelerate import Accelerator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen_img_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--real_img_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch",
        type=int,
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
    parser.add_argument(
        "--real_img_file",
        type=str,
        help="Enter the file name of the reference image for eval. \
            If you don't provide it, a random selection will be made.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return args


def load_and_preprocess(image_path_list, preprocess, device, batch):
    images = []
    for path in image_path_list:
        image = Image.open(path)
        image = preprocess(image).unsqueeze(0).to(device)
        images.append(image)

        if len(images) == batch:
            yield torch.cat(images)
            images = []

    # Rest images
    if images:
        yield torch.cat(images)


def append_to_json(output_file, new_data):
    MAX_BRACE_COUNT = 30
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
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

    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=4)


def main(args):

    # Create the output_dir if it does not exist
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📂 Created directory: {output_dir}")

    # Prepare accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load clip model
    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model = accelerator.prepare(clip_model)

    # Set the real_img_path
    if args.real_img_file is None:
        if args.seed is not None:
            random.seed(args.seed)
        real_img_file_ls = os.listdir(args.real_img_path)
        real_img_file = random.choice(real_img_file_ls)
        real_img_path = f"{args.real_img_path}/{real_img_file}"
        del real_img_file, real_img_file_ls
    else:
        real_img_path = f"{args.real_img_path}/{args.real_img_file}"

    # Extract the real image feature
    real_img = load_and_preprocess([real_img_path], preprocess, device, batch=1)
    real_ftr = clip_model.module.encode_image(next(real_img))
    real_ftr /= real_ftr.norm(dim=-1, keepdim=True)
    del real_img
    torch.cuda.empty_cache()

    # Set the gen_img_path
    gen_img_path_ls = [
        f"{args.gen_img_path}/{file}" for file in os.listdir(args.gen_img_path)
    ]

    progress_bar = tqdm(
        total=len(gen_img_path_ls), disable=not accelerator.is_local_main_process
    )
    scores = torch.tensor(0.0, device=device)
    score_num = torch.tensor(0, device=device)

    with accelerator.split_between_processes(gen_img_path_ls) as input:

        for batch in load_and_preprocess(input, preprocess, device, args.batch):

            # Extract the gen image feature
            with torch.no_grad():
                gen_ftr = clip_model.module.encode_image(batch)
                gen_ftr /= gen_ftr.norm(dim=-1, keepdim=True)

            # Calculate scores
            sim = (real_ftr @ gen_ftr.T).squeeze().cpu().detach()
            scores += sim.sum()
            score_num += sim.numel()

            progress_bar.update(
                sum(
                    accelerator.gather(torch.tensor([len(batch)], device=device))
                ).item()
            )

            del gen_ftr
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    progress_bar.close()

    # Calculate score per gpu
    score = scores / score_num

    gather_score = accelerator.gather(score)
    if accelerator.is_local_main_process:
        avg_score = gather_score.mean().cpu().item()

        # Generate output json file
        result = {
            "gen_img_path": args.gen_img_path,
            "model": args.clip_model,
            "real_img_file": args.real_img_file if args.real_img_file else "random",
            "batch": args.batch,
            "score": avg_score,
        }
        append_to_json(args.output_path, result)


if __name__ == "__main__":
    args = parse_args()
    main(args)
