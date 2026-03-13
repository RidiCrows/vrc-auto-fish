"""
训练数据集共享工具
==================
抽取路径、计数、统计和 ROI 读取等重复逻辑。
"""

from __future__ import annotations

import json
import os

import config
from trainer_common.profiles import TrainerProfile


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def build_dataset_paths(profile: TrainerProfile) -> dict[str, str]:
    base = profile.dataset_root
    return {
        "BASE": base,
        "UNLABELED": os.path.join(base, "images", "unlabeled"),
        "TRAIN_IMG": os.path.join(base, "images", "train"),
        "TRAIN_LBL": os.path.join(base, "labels", "train"),
        "VAL_IMG": os.path.join(base, "images", "val"),
        "VAL_LBL": os.path.join(base, "labels", "val"),
        "DATA_YAML": profile.data_yaml,
        "RUNS_DIR": profile.runs_root,
    }


def ensure_dataset_dirs(profile: TrainerProfile):
    paths = build_dataset_paths(profile)
    for key in ("UNLABELED", "TRAIN_IMG", "TRAIN_LBL", "VAL_IMG", "VAL_LBL"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def count_images(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for name in os.listdir(path) if name.lower().endswith(IMAGE_EXTS))


def count_labels(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for name in os.listdir(path) if name.lower().endswith(".txt"))


def load_saved_roi():
    try:
        with open(config.SETTINGS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None

    if isinstance(data, dict):
        if isinstance(data.get("current"), dict):
            return data["current"].get("DETECT_ROI")
        return data.get("DETECT_ROI")
    return None


def get_dataset_stats(profile: TrainerProfile) -> dict[str, int]:
    paths = ensure_dataset_dirs(profile)
    stats = {
        "train_images": count_images(paths["TRAIN_IMG"]),
        "train_labels": count_labels(paths["TRAIN_LBL"]),
        "val_images": count_images(paths["VAL_IMG"]),
        "val_labels": count_labels(paths["VAL_LBL"]),
        "unlabeled_images": count_images(paths["UNLABELED"]),
    }
    stats["labeled_pairs"] = min(
        stats["train_images"],
        stats["train_labels"],
    ) + min(
        stats["val_images"],
        stats["val_labels"],
    )
    return stats
