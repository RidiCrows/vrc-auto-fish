"""
共享标注工具辅助
================
抽取 yolo 与 fish_trainer 标注脚本共同使用的标签读写和文件枚举逻辑。
"""

from __future__ import annotations

import argparse
import os
import random
import shutil

from trainer_common.dataset import IMAGE_EXTS


def build_label_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--split", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--relabel", action="store_true", help="重新标注已有 train/val 数据")
    return parser


def list_relabel_entries(train_img, train_lbl, val_img, val_lbl):
    entries = []
    for img_dir, lbl_dir in ((train_img, train_lbl), (val_img, val_lbl)):
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            if name.lower().endswith(IMAGE_EXTS):
                entries.append({
                    "img_path": os.path.join(img_dir, name),
                    "lbl_path": os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"),
                })
    return entries


def list_unlabeled_entries(unlabeled_dir):
    if not os.path.isdir(unlabeled_dir):
        return []
    return [
        {"img_path": os.path.join(unlabeled_dir, name), "lbl_path": None}
        for name in sorted(os.listdir(unlabeled_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]


def load_existing_labels(lbl_path, img_w, img_h):
    loaded = []
    if not os.path.exists(lbl_path):
        return loaded
    with open(lbl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            loaded.append((cls, x1, y1, x2, y2))
    return loaded


def write_yolo_labels(lbl_path, image_shape, boxes):
    h, w = image_shape[:2]
    os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
    with open(lbl_path, "w", encoding="utf-8") as handle:
        for cls, x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            handle.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def save_new_labeled_entry(
    entry,
    image_shape,
    boxes,
    split_ratio,
    train_img,
    train_lbl,
    val_img,
    val_lbl,
    printer,
):
    is_val = random.random() < split_ratio
    dst_img_dir = val_img if is_val else train_img
    dst_lbl_dir = val_lbl if is_val else train_lbl
    img_path = entry["img_path"]
    name = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(dst_lbl_dir, name + ".txt")
    write_yolo_labels(lbl_path, image_shape, boxes)
    dst_img_path = os.path.join(dst_img_dir, os.path.basename(img_path))
    shutil.move(img_path, dst_img_path)
    printer(f"      -> {'val' if is_val else 'train'}/")
    return dst_img_path, lbl_path
