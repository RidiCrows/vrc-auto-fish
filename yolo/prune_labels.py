"""
YOLO 已标注数据清理入口
======================
保留 yolo 命令入口，但底层清理逻辑共享给 trainer_common。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from trainer_common.prune_labels import run_prune
from trainer_common.profiles import get_profile
from yolo.classes import CLASS_NAMES
from yolo.console import safe_print


def main(argv=None):
    run_prune(
        get_profile("runtime_yolo"),
        CLASS_NAMES,
        safe_print,
        argv=argv,
        aliases=config.LEGACY_FISH_KEY_ALIASES,
    )


if __name__ == "__main__":
    main()
