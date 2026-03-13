"""
共享训练入口
============
"""

from __future__ import annotations

import argparse
import os

from trainer_common.dataset import build_dataset_paths, count_images, ensure_dataset_dirs
from trainer_common.profiles import TrainerProfile


def build_parser(profile: TrainerProfile):
    parser = argparse.ArgumentParser(description=profile.train_description)
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="基础模型")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=-1, help="batch size")
    parser.add_argument("--resume", action="store_true", help="从上次中断继续训练")
    return parser


def run_train(profile: TrainerProfile, printer, argv=None):
    parser = build_parser(profile)
    args = parser.parse_args(argv)
    paths = ensure_dataset_dirs(profile)
    n_train = count_images(paths["TRAIN_IMG"])
    n_val = count_images(paths["VAL_IMG"])

    printer("=" * 50)
    printer(f"  {profile.train_banner}")
    printer("=" * 50)
    printer(f"  训练集: {n_train} 张")
    printer(f"  验证集: {n_val} 张")
    printer(f"  模型: {args.model}")
    printer(f"  数据配置: {paths['DATA_YAML']}")

    if n_train < 10:
        printer("[错误] 训练集图片不足，至少建议 10 张")
        return

    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        printer("[错误] 缺少 ultralytics 或 torch，请先安装依赖")
        return

    weights_dir = os.path.join(paths["RUNS_DIR"], profile.train_run_name, "weights")
    last_pt = os.path.join(weights_dir, "last.pt")
    if args.resume and os.path.exists(last_pt):
        model = YOLO(last_pt)
        printer(f"[继续训练] {last_pt}")
    else:
        model = YOLO(args.model)

    device = 0 if torch.cuda.is_available() else "cpu"
    model.train(
        data=paths["DATA_YAML"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=paths["RUNS_DIR"],
        name=profile.train_run_name,
        exist_ok=True,
        device=device,
        workers=4,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    best_pt = os.path.join(weights_dir, "best.pt")
    if os.path.exists(best_pt):
        printer(f"[OK] 训练完成: {best_pt}")
    else:
        printer("[警告] 未找到 best.pt，请检查训练日志")
