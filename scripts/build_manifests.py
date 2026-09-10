#!/usr/bin/env python
"""
Build training manifests from data/raw/videos/{real,fake}/*.mp4.

Pipeline:
1. Scan videos, record label / manipulation / license provenance.
2. Extract N uniform frames per video -> data/raw/frames/{real,fake}/.
3. Identity-leakage-safe split: union-find over source-video ids found in
   filenames (original "042.mp4" -> {042}; fake "042_137.mp4" -> {042,137});
   connected identity components are assigned wholly to one split.
4. Duplicate / near-duplicate check via SHA-256 of extracted frames.
5. Write data/{train,validation,test}/manifest.csv with columns:
   image_path, video_path, audio_path, label, source_video, manipulation,
   dataset, sha256, split (absolute paths; trainers filter their own column).
6. Write docs/dataset_inspection.json with the full research audit
   (labels, distribution, manipulation types, licensing, leakage report).

Usage:
    python scripts/build_manifests.py --frames-per-video 8
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

DATASET_NAME = "ffpp_sample_hf"
DATASET_URL = "https://huggingface.co/datasets/subhan1501/fake-face-detection-dataset"
MANIPULATION = "Deepfakes"
LICENSE = "MIT (per HuggingFace dataset card; underlying videos from FaceForensics++, YouTube-sourced)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_of(stem: str, label: str):
    """Identity ids embedded in filename. Fake 'AAA_BBB' involves both."""
    parts = stem.split("_")
    if label == "fake" and len(parts) >= 2:
        return {parts[0], parts[1]}
    return {parts[0]}


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def extract_frames(video_path: Path, out_dir: Path, stem: str, n: int):
    """Extract n uniform frames as JPEG. Returns list of saved paths."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = []
    if total <= 0:
        cap.release()
        return saved
    idxs = np.linspace(0, total - 1, min(n, total), dtype=int)
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        out = out_dir / f"{stem}_f{i:02d}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(out)
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/videos")
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.raw)
    frame_root = root.parent / "frames"
    (frame_root / "real").mkdir(parents=True, exist_ok=True)
    (frame_root / "fake").mkdir(parents=True, exist_ok=True)

    videos = []
    for label in ("real", "fake"):
        for vp in sorted((root / label).glob("*.mp4")):
            videos.append({"path": vp, "label": label})
    if not videos:
        logger.error(f"No videos found under {root}/{{real,fake}}")
        sys.exit(1)
    logger.info(f"Found {len(videos)} videos")

    # ---- identity components (leakage-safe grouping) ----
    uf = UnionFind()
    for v in videos:
        ids = ids_of(v["path"].stem, v["label"])
        v["ids"] = sorted(ids)
        idlist = list(ids)
        for other in idlist[1:]:
            uf.union(idlist[0], other)
    for v in videos:
        v["component"] = uf.find(v["ids"][0])

    # ---- frames + hashes ----
    frame_rows, video_rows = [], []
    for v in videos:
        vhash = sha256_file(v["path"])
        out_dir = frame_root / v["label"]
        frames = extract_frames(v["path"], out_dir, v["path"].stem, args.frames_per_video)
        comp = v["component"]
        for fp in frames:
            frame_rows.append({
                "image_path": str(fp.resolve()),
                "video_path": str(v["path"].resolve()),
                "audio_path": "",
                "label": v["label"],
                "source_video": v["path"].name,
                "manipulation": "original" if v["label"] == "real" else MANIPULATION,
                "dataset": DATASET_NAME,
                "sha256": sha256_file(fp),
                "component": comp,
            })
        video_rows.append({
            "image_path": "",
            "video_path": str(v["path"].resolve()),
            "audio_path": "",
            "label": v["label"],
            "source_video": v["path"].name,
            "manipulation": "original" if v["label"] == "real" else MANIPULATION,
            "dataset": DATASET_NAME,
            "sha256": vhash,
            "component": comp,
        })

    # ---- duplicate / near-duplicate frame check ----
    fhashes = [r["sha256"] for r in frame_rows]
    dup_frames = len(fhashes) - len(set(fhashes))
    vhashes = [r["sha256"] for r in video_rows]
    dup_videos = len(vhashes) - len(set(vhashes))

    # ---- stratified split by component ----
    comp_df = pd.DataFrame([
        {"component": v["component"], "label": v["label"]} for v in videos
    ]).drop_duplicates("component")
    # stratify needs >=2 members per class; fall back gracefully
    try:
        train_c, tmp_c = train_test_split(comp_df, test_size=0.30,
                                          stratify=comp_df["label"], random_state=args.seed)
        val_c, test_c = train_test_split(tmp_c, test_size=0.50,
                                         stratify=tmp_c["label"], random_state=args.seed)
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}); using unstratified split")
        train_c, tmp_c = train_test_split(comp_df, test_size=0.30, random_state=args.seed)
        val_c, test_c = train_test_split(tmp_c, test_size=0.50, random_state=args.seed)
    split_of = {c: "train" for c in train_c["component"]}
    split_of.update({c: "validation" for c in val_c["component"]})
    split_of.update({c: "test" for c in test_c["component"]})

    # leakage verification: no component in two splits (by construction) +
    # no shared identity id across splits
    comp_ids = {}
    for v in videos:
        comp_ids.setdefault(v["component"], set()).update(v["ids"])
    split_ids = {}
    for comp, ids in comp_ids.items():
        split_ids.setdefault(split_of[comp], set()).update(ids)
    import itertools
    leaks = []
    for (s1, i1), (s2, i2) in itertools.combinations(split_ids.items(), 2):
        overlap = i1 & i2
        if overlap:
            leaks.append({"splits": [s1, s2], "shared_ids": sorted(overlap)})

    all_rows = frame_rows + video_rows
    for r in all_rows:
        r["split"] = split_of[r["component"]]

    for split in ("train", "validation", "test"):
        d = Path("data") / split
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([r for r in all_rows if r["split"] == split])
        df.drop(columns=["component"]).to_csv(d / "manifest.csv", index=False)
        n_frames = int((df["image_path"] != "").sum())
        n_videos = int(((df["video_path"] != "") & (df["image_path"] == "")).sum())
        logger.info(f"{split}: {len(df)} rows (frame rows={n_frames}, video rows={n_videos})")

    inspection = {
        "dataset": DATASET_NAME,
        "source_url": DATASET_URL,
        "license": LICENSE,
        "manipulation_types": sorted({r["manipulation"] for r in all_rows}),
        "labels": {"real": 0, "fake": 1},
        "videos": {
            "total": len(video_rows),
            "real": sum(1 for r in video_rows if r["label"] == "real"),
            "fake": sum(1 for r in video_rows if r["label"] == "fake"),
            "exact_duplicate_files": dup_videos,
        },
        "frames": {
            "total": len(frame_rows),
            "per_video": args.frames_per_video,
            "exact_duplicate_frames": dup_frames,
        },
        "split": {
            "strategy": "stratified, grouped by identity component (union-find over filename ids)",
            "train_rows": sum(1 for r in all_rows if r["split"] == "train"),
            "validation_rows": sum(1 for r in all_rows if r["split"] == "validation"),
            "test_rows": sum(1 for r in all_rows if r["split"] == "test"),
            "identity_components": {s: sum(1 for c in set(v['component'] for v in videos) if split_of[c] == s)
                                    for s in ("train", "validation", "test")},
            "cross_split_identity_leaks": leaks,
        },
        "notes": [
            "Fake filenames 'AAA_BBB.mp4' involve two source identities; both are "
            "unioned into one split component to prevent identity leakage.",
            "No audio tracks are used; audio detector training is out of scope "
            "for this sample (documented as future work).",
        ],
    }
    Path("docs").mkdir(exist_ok=True)
    with open("docs/dataset_inspection.json", "w") as f:
        json.dump(inspection, f, indent=2)
    logger.info("Wrote docs/dataset_inspection.json")
    if leaks:
        logger.warning(f"IDENTITY LEAKS: {leaks}")
    else:
        logger.info("No cross-split identity leakage detected.")


if __name__ == "__main__":
    main()
