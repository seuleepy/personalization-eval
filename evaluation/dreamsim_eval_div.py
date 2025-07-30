"""
Evaluation script for computing intra-cluster diversity of generated images using DreamSim model.
Since the model measures distance, the values can be directly used as diversity scores.

This script:
- Load generated images grouped by prompts. (e.g., prompt_01, prompt_02, ...)
- Compute the distance between images using DreamSim model.

Output (JSON format):
{
    "gen_img_path": "path/to/generated/images",
    "average_score": 0.1234,
    "cluster": ["prompt_00", "prompt_01", ...],
    "per_cluster_score": [0.1234, 0.5678, ...]
}
"""

import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import json

from collections import defaultdict
import re

import torch
from dreamsim import dreamsim


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen_img_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
    )
    args = parser.parse_args()

    return args


def group_images_by_prompt(gen_img_path):
    pattern = re.compile(r"prompt_(\d+)_\d+\.jpg")
    cluster_dict = defaultdict(list)

    for filename in os.listdir(gen_img_path):
        match = pattern.search(filename)
        if match:
            prompt_id = match.group(1).zfill(2)
            cluster_key = f"prompt_{prompt_id}"

            cluster_dict[cluster_key].append(os.path.join(gen_img_path, filename))

    return cluster_dict


def compute_intra_cluster_diversity(image_paths, model, processor, device):
    """
    Compute the intra-cluster diversity of a set of images.
    """
    import itertools

    imgs = load_and_preprocess(image_paths, processor, device, batch=len(image_paths))

    sims = []
    with torch.no_grad():
        for i, j in itertools.combinations(range(len(imgs)), 2):  # all pairs
            img1 = imgs[i]
            img2 = imgs[j]
            distance = model(img1, img2)
            sims.append(distance)

    return torch.stack(sims).mean()


def load_and_preprocess(image_path_list, processor, device, batch):
    images = []
    for path in image_path_list:
        image = Image.open(path)
        image = processor(image).to(device)
        images.append(image)
    return images


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
    device = torch.device(args.device)

    # Load model
    model, processor = dreamsim(pretrained=True, device=device)

    # Cluster
    cluster_dict = group_images_by_prompt(args.gen_img_path)
    cluster_keys = sorted(cluster_dict.keys())

    pbar = tqdm(cluster_keys, desc="Clusters")

    scores = []
    for key in pbar:
        paths = cluster_dict[key]
        if len(paths) < 2:
            continue
        score = compute_intra_cluster_diversity(paths, model, processor, device)
        scores.append((key, score))
        pbar.set_postfix({key: f"{score:.4f}"})

    # Calculate average score
    avg_score = sum(s for _, s in scores) / len(scores) if scores else 0.0

    cluster_names = [k for k, _ in scores]
    cluster_scores = [s.item() for _, s in scores]

    # JSON output
    result = {
        "gen_img_path": args.gen_img_path,
        "average_score": avg_score.item(),
        "cluster": cluster_names,
        "per_cluster_score": cluster_scores,
    }
    append_to_json(args.output_path, result)
    print(f"Done! avg intra-cluster diversity = {avg_score:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
