"""
多颜色鱼训练数据采集
====================
独立入口保留在 fish_trainer，但采集实现共享给 trainer_common。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fish_trainer.console import safe_print
from trainer_common.collect import run_collect
from trainer_common.profiles import get_profile


def build_parser():
    from trainer_common.collect import build_parser

    return build_parser(get_profile("multicolor"))


def main(argv=None):
    run_collect(get_profile("multicolor"), safe_print, argv=argv)


if __name__ == "__main__":
    main()
