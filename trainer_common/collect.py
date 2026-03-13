"""
共享训练采集逻辑
================
"""

from __future__ import annotations

import argparse
import os
import time

import cv2

import config
from core.screen import ScreenCapture
from core.window import WindowManager
from trainer_common.dataset import build_dataset_paths, ensure_dataset_dirs, load_saved_roi
from trainer_common.profiles import TrainerProfile


def build_parser(profile: TrainerProfile):
    parser = argparse.ArgumentParser(description=profile.collect_description)
    parser.add_argument("--fps", type=float, default=2.0, help="每秒截图数")
    parser.add_argument("--roi", action="store_true", help="只截取已保存的 ROI")
    parser.add_argument("--max", type=int, default=0, help="最大截图数量，0 表示无限")
    return parser


def run_collect(profile: TrainerProfile, printer, argv=None):
    parser = build_parser(profile)
    args = parser.parse_args(argv)
    paths = ensure_dataset_dirs(profile)

    window = WindowManager(config.WINDOW_TITLE)
    screen = ScreenCapture()
    if not window.find():
        printer("[错误] 未找到 VRChat 窗口，请确保游戏正在运行")
        return

    roi = load_saved_roi() if args.roi else None
    interval = 1.0 / max(args.fps, 0.1)
    count = 0

    printer(f"[OK] 已连接: {window.title} (HWND={window.hwnd})")
    printer(f"[保存] {paths['UNLABELED']}")
    printer(f"[设置] 截图间隔: {interval:.2f}s | ROI: {'是' if roi else '否'}")
    if roi:
        printer(f"[ROI] X={roi[0]} Y={roi[1]} {roi[2]}x{roi[3]}")

    try:
        while True:
            if not window.is_valid() and not window.find():
                printer("[等待] VRChat 窗口未找到，5 秒后重试...")
                time.sleep(5)
                continue

            img, _ = screen.grab_window(window)
            if img is None:
                time.sleep(0.5)
                continue

            if roi:
                rx, ry, rw, rh = roi
                img = img[ry:ry + rh, rx:rx + rw]

            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((time.time() % 1) * 1000)
            name = f"{ts}_{ms:03d}.png"
            cv2.imwrite(os.path.join(paths["UNLABELED"], name), img)
            count += 1
            h, w = img.shape[:2]
            printer(f"  [{count}] {name} ({w}x{h})", end="\r")

            if args.max > 0 and count >= args.max:
                printer(f"\n[完成] 已采集 {count} 张截图")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        printer(f"\n[停止] 共采集 {count} 张截图 -> {paths['UNLABELED']}")
