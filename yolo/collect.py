"""
YOLO 多颜色鱼训练数据采集
==========================
保留 yolo 命令入口，但采集实现共享给 trainer_common。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer_common.collect import run_collect
from trainer_common.profiles import get_profile
from yolo.console import safe_print


def build_parser():
    from trainer_common.collect import build_parser

    return build_parser(get_profile("runtime_yolo"))


def main(argv=None):
    run_collect(get_profile("runtime_yolo"), safe_print, argv=argv)


if __name__ == "__main__":
    main()
