"""
Evaluation script for computing similarity between generated images and a reference image using DINO model.
All similarity scores for individual samples are recorded in the output.

This script:
- Load generated images and a reference image.
- Compute the similarity score between the generated images and the reference image using DINO model.

Output (JSON format):
{
    "gen_img_path": "path/to/generated/images",
    "real_img_file": "reference_image.jpg",
    "batch": 50,
    "score": 0.1234,
    "per_samples_score": [0.1234, 0.5678, ...]
    "per_samples_name": ["00_00", "00_01", ...]
}

"""

import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import json
import random

import torch
from transformers import ViTImageProcessor, ViTModel
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


def load_and_preprocess(image_path_list, processor, device, batch):
    images = []
    paths = []

    for path in image_path_list:
        image = Image.open(path)
        image = processor(images=image, return_tensors="pt").to(device)
        images.append(image["pixel_values"])

        path_name = os.path.splitext(os.path.basename(path))[0].replace("prompt_", "")
        paths.append(path_name)

        if len(images) == batch:
            yield torch.cat(images), paths
            images = []
            paths = []

    # rest images
    if images:
        yield torch.cat(images), paths


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
    model = ViTModel.from_pretrained("facebook/dino-vits16").to(device)
    processor = ViTImageProcessor.from_pretrained("facebook/dino-vits16")

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
    real_img, _ = next(load_and_preprocess([real_img_path], processor, device, batch=1))
    real_ftr = model(real_img).last_hidden_state[:, 0]
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

    img_scores = torch.tensor([], device=device)
    img_names = torch.tensor([], device=device)

    with accelerator.split_between_processes(gen_img_path_ls) as input:

        for batch_images, batch_paths in load_and_preprocess(
            input, processor, device, args.batch
        ):

            # Extract the gen image feature
            with torch.no_grad():
                gen_ftr = model(batch_images).last_hidden_state[:, 0]
                gen_ftr /= gen_ftr.norm(dim=-1, keepdim=True)

            # Calculate scores
            sim = (real_ftr @ gen_ftr.T).squeeze()
            scores += sim.sum()
            score_num += sim.numel()

            img_scores = torch.cat([img_scores, sim])

            # Encode image names to UTF-8 byte tensors and pad to same length
            # This is required because accelerator.gather does not support string tensors
            encoded_names = [name.encode("utf-8") for name in batch_paths]
            max_len = max(len(n) for n in encoded_names)
            name_tensor = torch.zeros(
                (len(encoded_names), max_len), dtype=torch.uint8, device=device
            )
            for i, n in enumerate(encoded_names):
                name_tensor[i, : len(n)] = torch.tensor(list(n), dtype=torch.uint8)
            img_names = torch.cat([img_names, name_tensor], dim=0)

            progress_bar.update(
                sum(
                    accelerator.gather(torch.tensor([len(batch_images)], device=device))
                ).item()
            )

            del gen_ftr
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    progress_bar.close()

    # Calculate score per gpu
    score = scores / score_num

    gather_score = accelerator.gather(score)
    gathered_sample_scores = accelerator.gather(img_scores)
    gathered_sample_names = accelerator.gather(img_names)

    if accelerator.is_local_main_process:
        avg_score = gather_score.mean().cpu().item()

        decoded_names = []
        for row in gathered_sample_names.cpu():
            name_bytes = bytes([int(b) for b in row if int(b) != 0])
            decoded_names.append(name_bytes.decode("utf-8"))

        # Generate output json file
        result = {
            "gen_img_path": args.gen_img_path,
            "real_img_file": args.real_img_file if args.real_img_file else "random",
            "batch": args.batch,
            "score": avg_score,
            "per_samples_score": gathered_sample_scores.cpu().tolist(),
            "per_samples_name": decoded_names,
        }
        append_to_json(args.output_path, result)


if __name__ == "__main__":
    args = parse_args()
    main(args)
