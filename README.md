# Personalization Evaluation

## Features
* Text-Image Similarity using CLIP
* Image-Image Similarity using CLIP and DINO
* Intra-cluster Diversity using DINO and DreamSim
* Inference Efficiency benchmarking (latency, memory)

## 🧩 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Text-Image Similarity**  | Computes similarity between prompts and generated images |
| **Image-Image Similarity** | Computes similarity between reference images and generated images |
| **Diversity**              | Measures variation across generated samples |
| **Inference Efficiency**   | Tracks latency and peak memory usage during generation |

## Usage

This repository uses  [OpenAI's CLIP model](https://github.com/openai/CLIP) for similarity evaluation.
Please make sure to install CLIP first:

```bash
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
### Similarity and Diversity
You can evaluate **similarity** and **diversity** using the `scripts/run.sh` bash script.
#### Notes:
* `WORKING_DIR`: Must be updated to your path before running the script.
* `gen_folders`: These are the **top-level folders** under `WORKING_DIR`, such as different experiment versions.
* If you only have one folder structure, you can remove this loop.
* `filenames`: These are subdirectory names **within each `gen_folder`** and represent the names of DreamBooth datsets. Replace them with the actual folder names you want to evaluate.
* Change `eval_file_name.py` to any other file you want to run.
* If the evaluation file **does not support `accelerate`**, simply run it using `python` instead.
  
  ```bash
  # Replace this:
  accelerate launch evaluation/eval_file_name.py --args

  # With this:
  python evaluation/eval_file_name.py --args
  ```
* Make sure --args match the script's `argparse` config.

### Inference Efficiency
We provide two Jupyter notebooks for measuring inference efficiency, including latency and memory usage.

| Notebook | Description |
|----------|-------------|
| `inference_efficiency.ipynb` | Evaluates baseline models using the `StableDiffusionPipeline` from 🤗 Diffusers |
| `inference_efficiency_mindiff.ipynb` | Evaluates models with MINDiff using `StableDiffusionMaskPipeline` |

Each notebook runs multiple forward passes and reports:

- **Average latency** (in milliseconds)
- **Peak memory usage** (in MB)

Make sure to update the model paths and prompts according to your own evaluation setup.
