"""
多颜色鱼模型训练脚本
====================
独立入口保留在 fish_trainer，但训练实现共享给 trainer_common。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fish_trainer.console import safe_print
from trainer_common.profiles import get_profile
from trainer_common.train import run_train


def count_images(path):
    from trainer_common.dataset import count_images as shared_count_images

    return shared_count_images(path)


def main(argv=None):
    run_train(get_profile("multicolor"), safe_print, argv=argv)


if __name__ == "__main__":
    main()
