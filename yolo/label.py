"""
YOLO 标注辅助工具
================
用 OpenCV 窗口打开未标注的截图，让用户画框 + 选择类别。
标注结果保存为 YOLO 格式 (.txt)，完成后自动移到 train/ 目录。

类别:
  F = fish_generic
  1-9 = 多颜色鱼
  B = bar
  T = track
  P = progress

操作:
  鼠标拖拽  = 画框
  F/1-9/B/T/P = 设置类别
  N/M       = 下一个/上一个类别
  Z         = 撤销上一个框
  X         = 清空当前图片标注
  H         = 显示帮助
  S / Enter = 保存并下一张
  D         = 删除当前图片 (跳过)
  Q / Esc   = 退出

用法:
    python -m yolo.label                 # 标注 unlabeled/ 中的新图
    python -m yolo.label --split 0.2     # 20% 分到 val/
    python -m yolo.label --relabel       # 重新标注已有 train/val 图片 (补标新类别)
"""

import os
import sys
import random
import shutil
import argparse
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

BASE = os.path.join(config.BASE_DIR, "yolo", "dataset")
UNLABELED = os.path.join(BASE, "images", "unlabeled")
TRAIN_IMG = os.path.join(BASE, "images", "train")
TRAIN_LBL = os.path.join(BASE, "labels", "train")
VAL_IMG = os.path.join(BASE, "images", "val")
VAL_LBL = os.path.join(BASE, "labels", "val")

# 兼容旧版 yolo/dataset 中 bar/track/progress 的类 ID，不破坏历史标签。
CLASS_NAMES = {
    0: "fish_generic",
    1: "bar",
    2: "track",
    3: "progress",
    4: "fish_white",
    5: "fish_copper",
    6: "fish_green",
    7: "fish_blue",
    8: "fish_purple",
    9: "fish_golden",
    10: "fish_red",
    11: "fish_pink",
    12: "fish_rainbow",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 255, 255),
    2: (255, 100, 0),
    3: (0, 200, 255),
    4: (255, 255, 255),
    5: (60, 140, 200),
    6: (0, 220, 0),
    7: (255, 140, 0),
    8: (220, 80, 220),
    9: (0, 215, 255),
    10: (0, 0, 255),
    11: (200, 120, 255),
    12: (0, 255, 255),
}

DISPLAY_NAMES = {
    0: "通用鱼",
    1: "白条",
    2: "轨道",
    3: "进度条",
    4: "白鱼",
    5: "铜鱼",
    6: "绿鱼",
    7: "蓝鱼",
    8: "紫鱼",
    9: "金鱼",
    10: "红鱼",
    11: "粉鱼",
    12: "彩鱼",
}

OVERLAY_NAMES = dict(CLASS_NAMES)

KEY_TO_CLASS = {
    ord("f"): 0,
    ord("b"): 1,
    ord("t"): 2,
    ord("p"): 3,
    ord("1"): 4,
    ord("2"): 5,
    ord("3"): 6,
    ord("4"): 7,
    ord("5"): 8,
    ord("6"): 9,
    ord("7"): 10,
    ord("8"): 11,
    ord("9"): 12,
}

CLASS_SHORTCUTS = {
    0: "F",
    1: "B",
    2: "T",
    3: "P",
    4: "1",
    5: "2",
    6: "3",
    7: "4",
    8: "5",
    9: "6",
    10: "7",
    11: "8",
    12: "9",
}

drawing = False
ix, iy = 0, 0
boxes = []          # [(class_id, x1, y1, x2, y2), ...]
current_class = 0
img_display = None
img_orig = None


def short_help():
    return (
        "[F]=generic [1-9]=fish colors [B]=bar [T]=track [P]=progress "
        "[N/M]=prev/next [Z]=undo [X]=clear [H]=help [S]=save [D]=skip [Q]=quit"
    )


def print_help():
    print("=" * 72)
    print("  YOLO 多颜色鱼标注快捷键")
    print("=" * 72)
    for cls_id in sorted(CLASS_NAMES):
        print(
            f"  [{CLASS_SHORTCUTS.get(cls_id, '?')}] "
            f"{DISPLAY_NAMES.get(cls_id, CLASS_NAMES[cls_id])} ({CLASS_NAMES[cls_id]})"
        )
    print("  [N]/[M] 下一个/上一个类别")
    print("  [Z] 撤销  [X] 清空当前图片标注")
    print("  [S]/[Enter] 保存  [D] 跳过  [Q]/[Esc] 退出")
    print("=" * 72)


def draw_overlay():
    global img_display
    img_display = img_orig.copy()
    h, w = img_display.shape[:2]
    legend_y = 20

    for cls, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(cls, (128, 128, 128))
        label = OVERLAY_NAMES.get(cls, CLASS_NAMES.get(cls, "?"))
        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img_display, f"{label} ({cls})", (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    for cls_id in sorted(CLASS_NAMES):
        color = CLASS_COLORS.get(cls_id, (128, 128, 128))
        marker = ">" if cls_id == current_class else " "
        legend = f"{marker}[{CLASS_SHORTCUTS.get(cls_id, '?')}] {OVERLAY_NAMES.get(cls_id)}"
        cv2.putText(
            img_display, legend, (8, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1
        )
        legend_y += 18

    info = (
        f"Class: {OVERLAY_NAMES.get(current_class, '?')} | "
        f"Boxes: {len(boxes)} | {short_help()}"
    )
    cv2.putText(
        img_display, info, (5, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1
    )


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = img_display.copy()
        color = CLASS_COLORS.get(current_class, (128, 128, 128))
        cv2.rectangle(tmp, (ix, iy), (x, y), color, 2)
        cv2.imshow("Label Tool", tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        if x2 - x1 > 5 and y2 - y1 > 5:
            boxes.append((current_class, x1, y1, x2, y2))
            draw_overlay()
            cv2.imshow("Label Tool", img_display)


def save_annotation(img_path, dst_img_dir, dst_lbl_dir):
    """保存 YOLO 格式标注并移动图片"""
    h, w = img_orig.shape[:2]
    name = os.path.splitext(os.path.basename(img_path))[0]

    lbl_path = os.path.join(dst_lbl_dir, name + ".txt")
    with open(lbl_path, "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    dst_path = os.path.join(dst_img_dir, os.path.basename(img_path))
    shutil.move(img_path, dst_path)
    return lbl_path, dst_path


def load_existing_labels(lbl_path, img_w, img_h):
    """加载 YOLO 格式标注文件，转换回像素坐标的 boxes 列表"""
    loaded = []
    if not os.path.exists(lbl_path):
        return loaded
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            loaded.append((cls, x1, y1, x2, y2))
    return loaded


def save_annotation_inplace(lbl_path):
    """直接覆盖写入标注文件 (用于 relabel 模式)"""
    h, w = img_orig.shape[:2]
    with open(lbl_path, "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw_n = (x2 - x1) / w
            bh_n = (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}\n")


def _label_loop(files_with_paths, save_func, mode_name="标注"):
    """
    通用标注循环。

    files_with_paths: [(img_path, lbl_path_or_None), ...]
    save_func: callable(img_path, lbl_path) -> 保存后打印信息的回调
    """
    global current_class, boxes, img_orig, img_display

    cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Label Tool", mouse_cb)
    print_help()

    labeled = 0
    for i, (fpath, existing_lbl) in enumerate(files_with_paths):
        img_orig = cv2.imread(fpath)
        if img_orig is None:
            continue

        h, w = img_orig.shape[:2]

        if existing_lbl and os.path.exists(existing_lbl):
            boxes = load_existing_labels(existing_lbl, w, h)
        else:
            boxes = []

        current_class = 3 if existing_lbl else 0

        dw = min(w, 1280)
        dh = int(h * dw / w)
        cv2.resizeWindow("Label Tool", dw, dh)

        draw_overlay()
        cv2.imshow("Label Tool", img_display)
        fname = os.path.basename(fpath)
        existing_count = len(boxes)
        title = f"[{i+1}/{len(files_with_paths)}] {fname} ({w}x{h}) 已有{existing_count}框"
        print(f"  {title}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            lower_key = ord(chr(key).lower()) if key < 256 else key

            if lower_key in KEY_TO_CLASS:
                current_class = KEY_TO_CLASS[lower_key]
                print(f"    类别 → {DISPLAY_NAMES.get(current_class)} ({CLASS_NAMES.get(current_class)})")
                draw_overlay()
                cv2.imshow("Label Tool", img_display)
            elif key == ord("n") or key == ord("N"):
                current_class = (current_class + 1) % len(CLASS_NAMES)
                print(f"    类别 → {DISPLAY_NAMES.get(current_class)} ({CLASS_NAMES.get(current_class)})")
                draw_overlay()
                cv2.imshow("Label Tool", img_display)
            elif key == ord("m") or key == ord("M"):
                current_class = (current_class - 1) % len(CLASS_NAMES)
                print(f"    类别 → {DISPLAY_NAMES.get(current_class)} ({CLASS_NAMES.get(current_class)})")
                draw_overlay()
                cv2.imshow("Label Tool", img_display)
            elif key == ord("z") or key == ord("Z"):
                if boxes:
                    removed = boxes.pop()
                    print(f"    撤销: {DISPLAY_NAMES.get(removed[0], '?')}")
                    draw_overlay()
                    cv2.imshow("Label Tool", img_display)
            elif key == ord("x") or key == ord("X"):
                boxes.clear()
                print("    已清空当前图片全部标注")
                draw_overlay()
                cv2.imshow("Label Tool", img_display)
            elif key == ord("h") or key == ord("H"):
                print_help()
            elif key == ord("s") or key == ord("S") or key == 13:
                if not boxes:
                    print("    [跳过] 没有标注框")
                    break
                save_func(fpath, existing_lbl)
                print(f"    [保存] {len(boxes)} 个框")
                labeled += 1
                break
            elif key == ord("d") or key == ord("D"):
                print("    [跳过] 此图不修改")
                break
            elif key == ord("q") or key == ord("Q") or key == 27:
                print(f"\n[退出] 共{mode_name} {labeled} 张")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"\n[完成] 共{mode_name} {labeled} 张图片")


def main():
    global current_class, boxes, img_orig, img_display

    parser = argparse.ArgumentParser(description="YOLO 标注工具")
    parser.add_argument("--split", type=float, default=0.2,
                        help="验证集比例 (默认 0.2)")
    parser.add_argument("--relabel", action="store_true",
                        help="重新标注已有 train/val 图片 (补标新类别)")
    args = parser.parse_args()

    os.makedirs(TRAIN_IMG, exist_ok=True)
    os.makedirs(TRAIN_LBL, exist_ok=True)
    os.makedirs(VAL_IMG, exist_ok=True)
    os.makedirs(VAL_LBL, exist_ok=True)

    if args.relabel:
        _relabel_mode()
        return

    files = sorted([
        f for f in os.listdir(UNLABELED)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    if not files:
        print(f"[提示] {UNLABELED} 中没有未标注的图片")
        print("  先运行: python -m yolo.collect")
        return

    print(f"[✓] 找到 {len(files)} 张未标注图片")
    print(f"[设置] 验证集比例: {args.split:.0%}")
    print()

    def save_new(fpath, _existing_lbl):
        is_val = random.random() < args.split
        d_img = VAL_IMG if is_val else TRAIN_IMG
        d_lbl = VAL_LBL if is_val else TRAIN_LBL
        save_annotation(fpath, d_img, d_lbl)
        split_name = "val" if is_val else "train"
        print(f"      → {split_name}/")

    file_pairs = [
        (os.path.join(UNLABELED, f), None) for f in files
    ]
    _label_loop(file_pairs, save_new, mode_name="标注")
    print(f"\n下一步: python -m yolo.train")


def _relabel_mode():
    """重新标注模式: 遍历 train/ 和 val/ 中已有图片，加载标注后让用户补标"""
    print("=" * 50)
    print("  补标模式 (relabel)")
    print("  加载已有标注，按 [F]/[1-9]/[B]/[T]/[P] 选择类别后画框")
    print("  按 [S] 保存  |  [D] 跳过  |  [Q] 退出")
    print("=" * 50)
    print()

    file_pairs = []
    for img_dir, lbl_dir in [(TRAIN_IMG, TRAIN_LBL), (VAL_IMG, VAL_LBL)]:
        if not os.path.isdir(img_dir):
            continue
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img_path = os.path.join(img_dir, f)
            name = os.path.splitext(f)[0]
            lbl_path = os.path.join(lbl_dir, name + ".txt")
            file_pairs.append((img_path, lbl_path))

    if not file_pairs:
        print("[提示] train/ 和 val/ 中没有已标注的图片")
        return

    print(f"[✓] 找到 {len(file_pairs)} 张已标注图片")
    print()

    def save_relabel(_fpath, existing_lbl):
        save_annotation_inplace(existing_lbl)

    _label_loop(file_pairs, save_relabel, mode_name="补标")


if __name__ == "__main__":
    main()
