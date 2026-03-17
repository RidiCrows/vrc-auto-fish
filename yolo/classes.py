"""
YOLO 多颜色鱼类别定义
=====================
统一使用 fish_trainer 的类别顺序，但保留 yolo 模块名称不变。
"""

CLASS_NAMES = {
    0: "fish_black",
    1: "fish_white",
    2: "fish_relic",
    3: "fish_green",
    4: "fish_blue",
    5: "fish_purple",
    6: "fish_golden",
    7: "fish_red",
    8: "fish_pink",
    9: "fish_rainbow",
    10: "bar",
    11: "track",
    12: "progress",
    13: "prog_hook",
    14: "fish_clover",
    15: "fish_question",
}

CLASS_COLORS = {
    0: (80, 80, 80),
    1: (255, 255, 255),
    2: (60, 140, 200),
    3: (0, 220, 0),
    4: (255, 140, 0),
    5: (220, 80, 220),
    6: (0, 215, 255),
    7: (0, 0, 255),
    8: (200, 120, 255),
    9: (0, 255, 255),
    10: (255, 255, 255),
    11: (255, 100, 0),
    12: (0, 200, 255),
    13: (180, 180, 180),
    14: (200, 220, 0),
    15: (80, 255, 255),
}

DISPLAY_NAMES = {
    0: "黑鱼",
    1: "白鱼",
    2: "遗物",
    3: "绿鱼",
    4: "蓝鱼",
    5: "紫鱼",
    6: "金鱼",
    7: "红鱼",
    8: "粉鱼",
    9: "彩鱼",
    10: "白条",
    11: "轨道",
    12: "进度条",
    13: "进度钩",
    14: "四叶草",
    15: "问号鱼",
}

OVERLAY_NAMES = {
    0: "fish_black",
    1: "fish_white",
    2: "fish_relic",
    3: "fish_green",
    4: "fish_blue",
    5: "fish_purple",
    6: "fish_golden",
    7: "fish_red",
    8: "fish_pink",
    9: "fish_rainbow",
    10: "bar",
    11: "track",
    12: "progress",
    13: "prog_hook",
    14: "fish_clover",
    15: "fish_question",
}

KEY_TO_CLASS = {
    ord("f"): 0,
    ord("1"): 1,
    ord("2"): 2,
    ord("3"): 3,
    ord("4"): 4,
    ord("5"): 5,
    ord("6"): 6,
    ord("7"): 7,
    ord("8"): 8,
    ord("9"): 9,
    ord("b"): 10,
    ord("t"): 11,
    ord("p"): 12,
    ord("k"): 13,
    ord("0"): 14,
    ord("/"): 15,
}

CLASS_SHORTCUTS = {
    0: "F",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "B",
    11: "T",
    12: "P",
    13: "K",
    14: "0",
    15: "?",
}


def class_items():
    return [(cls_id, CLASS_NAMES[cls_id]) for cls_id in sorted(CLASS_NAMES)]
