#!/usr/bin/env python
"""
Download a small, balanced sample of the FaceForensics++-style dataset
`subhan1501/fake-face-detection-dataset` (HuggingFace, MIT license per
dataset card) for research prototyping.

Downloads N original (real) + N Deepfakes (fake) mp4 videos into
data/raw/videos/{real,fake}/.

Source: https://huggingface.co/datasets/subhan1501/fake-face-detection-dataset
Underlying videos originate from FaceForensics++ (Rossler et al., 2019),
sourced from YouTube. See docs/DATASET_GUIDE.md for full provenance.
"""
import argparse
import sys
import urllib.request
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download
from loguru import logger

REPO = "subhan1501/fake-face-detection-dataset"


def list_dir(subdir: str):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{subdir}"
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.load(r)


def main():
    parser = argparse.ArgumentParser(description="Download FF++-style sample videos")
    parser.add_argument("--num-per-class", type=int, default=40)
    parser.add_argument("--stride", type=int, default=25,
                        help="Sampling stride for identity diversity")
    parser.add_argument("--out", default="data/raw/videos")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)

    original = sorted(x["path"] for x in list_dir("raw/original") if x["path"].endswith(".mp4"))
    fakes = sorted(x["path"] for x in list_dir("raw/Deepfakes") if x["path"].endswith(".mp4"))
    logger.info(f"Repo has {len(original)} original + {len(fakes)} Deepfakes videos")

    sel_real = [original[i] for i in range(0, len(original), args.stride)][:args.num_per_class]
    sel_fake = [fakes[i] for i in range(0, len(fakes), args.stride)][:args.num_per_class]

    for remote in sel_real:
        dest = out / "real" / Path(remote).name
        if dest.exists():
            logger.info(f"Exists, skip: {dest.name}")
            continue
        p = hf_hub_download(REPO, remote, repo_type="dataset", local_dir=str(out.parent / "hf_cache"))
        # hf_hub_download mirrors repo structure; copy into place
        import shutil
        shutil.copy2(p, dest)
        logger.info(f"Downloaded real: {dest.name} ({dest.stat().st_size // 1024} KB)")

    for remote in sel_fake:
        dest = out / "fake" / Path(remote).name
        if dest.exists():
            logger.info(f"Exists, skip: {dest.name}")
            continue
        p = hf_hub_download(REPO, remote, repo_type="dataset", local_dir=str(out.parent / "hf_cache"))
        import shutil
        shutil.copy2(p, dest)
        logger.info(f"Downloaded fake: {dest.name} ({dest.stat().st_size // 1024} KB)")

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
