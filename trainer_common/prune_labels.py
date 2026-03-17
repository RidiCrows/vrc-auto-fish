"""
共享已标注数据清理工具
====================
按类别批量删除 train/val 中已有标签，必要时把删空标签的图片移回 unlabeled。
"""

from __future__ import annotations

import argparse
import os
import shutil

from trainer_common.dataset import IMAGE_EXTS, ensure_dataset_dirs
from trainer_common.profiles import TrainerProfile


def build_parser(profile: TrainerProfile) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"清理 {profile.name} 已标注数据中的指定类别"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="要删除的类别名或类别 id，可传多个，例如 fish_clover fish_relic 3",
    )
    parser.add_argument(
        "--delete-empty-images",
        action="store_true",
        help="当一张图删完后没有剩余标签时，连图片一起删除；默认移回 unlabeled",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览会删除哪些标签，不实际写回文件",
    )
    return parser


def _resolve_target_ids(
    class_names: dict[int, str],
    raw_targets: list[str],
    aliases: dict[str, str] | None = None,
) -> tuple[set[int], list[str]]:
    aliases = aliases or {}
    name_to_id = {name: cls_id for cls_id, name in class_names.items()}

    target_ids: set[int] = set()
    resolved_names: list[str] = []
    unknown: list[str] = []

    for raw in raw_targets:
        value = raw.strip()
        if not value:
            continue
        if value.isdigit():
            cls_id = int(value)
            if cls_id in class_names:
                target_ids.add(cls_id)
                resolved_names.append(class_names[cls_id])
            else:
                unknown.append(value)
            continue
        mapped = aliases.get(value, value)
        cls_id = name_to_id.get(mapped)
        if cls_id is None:
            unknown.append(value)
            continue
        target_ids.add(cls_id)
        resolved_names.append(class_names[cls_id])

    if unknown:
        raise SystemExit(f"未知类别: {', '.join(unknown)}")
    return target_ids, sorted(set(resolved_names))


def _matching_image_path(img_dir: str, stem: str) -> str | None:
    for ext in IMAGE_EXTS:
        path = os.path.join(img_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def _unique_unlabeled_path(unlabeled_dir: str, image_name: str) -> str:
    base, ext = os.path.splitext(image_name)
    dst = os.path.join(unlabeled_dir, image_name)
    if not os.path.exists(dst):
        return dst
    idx = 1
    while True:
        candidate = os.path.join(unlabeled_dir, f"{base}.relabel{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def run_prune(
    profile: TrainerProfile,
    class_names: dict[int, str],
    printer,
    argv=None,
    aliases: dict[str, str] | None = None,
):
    paths = ensure_dataset_dirs(profile)
    parser = build_parser(profile)
    args = parser.parse_args(argv)

    target_ids, target_names = _resolve_target_ids(class_names, args.classes, aliases)
    printer(f"[清理] profile={profile.name} targets={', '.join(target_names)}")
    if args.dry_run:
        printer("[清理] dry-run 模式，不会修改文件")

    stats = {
        "labels_scanned": 0,
        "labels_modified": 0,
        "boxes_removed": 0,
        "images_moved": 0,
        "images_deleted": 0,
    }
    changed_files: list[str] = []

    for split_name, img_dir, lbl_dir in (
        ("train", paths["TRAIN_IMG"], paths["TRAIN_LBL"]),
        ("val", paths["VAL_IMG"], paths["VAL_LBL"]),
    ):
        if not os.path.isdir(lbl_dir):
            continue
        for name in sorted(os.listdir(lbl_dir)):
            if not name.lower().endswith(".txt"):
                continue
            stats["labels_scanned"] += 1
            lbl_path = os.path.join(lbl_dir, name)
            stem = os.path.splitext(name)[0]
            img_path = _matching_image_path(img_dir, stem)

            with open(lbl_path, "r", encoding="utf-8") as handle:
                original_lines = handle.readlines()

            kept_lines: list[str] = []
            removed_here = 0
            for line in original_lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    kept_lines.append(line)
                    continue
                if cls_id in target_ids:
                    removed_here += 1
                else:
                    kept_lines.append(line)

            if removed_here == 0:
                continue

            stats["labels_modified"] += 1
            stats["boxes_removed"] += removed_here
            changed_files.append(os.path.join(split_name, name))
            printer(f"[清理] {split_name}/{name}: 删除 {removed_here} 个框")

            if args.dry_run:
                continue

            if kept_lines:
                with open(lbl_path, "w", encoding="utf-8") as handle:
                    handle.writelines(kept_lines)
                continue

            os.remove(lbl_path)
            if img_path and os.path.exists(img_path):
                if args.delete_empty_images:
                    os.remove(img_path)
                    stats["images_deleted"] += 1
                    printer(f"        -> 删除空标签图片: {os.path.basename(img_path)}")
                else:
                    dst = _unique_unlabeled_path(paths["UNLABELED"], os.path.basename(img_path))
                    shutil.move(img_path, dst)
                    stats["images_moved"] += 1
                    printer(f"        -> 移回 unlabeled/: {os.path.basename(dst)}")

    if not changed_files:
        printer("[清理] 没有匹配到需要删除的已标注框")
        return

    printer(
        "[清理] 完成: "
        f"扫描标签 {stats['labels_scanned']} 个, "
        f"修改 {stats['labels_modified']} 个, "
        f"删除框 {stats['boxes_removed']} 个, "
        f"移回图片 {stats['images_moved']} 张, "
        f"删除空图片 {stats['images_deleted']} 张"
    )
