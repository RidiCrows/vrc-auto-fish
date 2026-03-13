"""
YOLO 多颜色鱼模型训练脚本
==========================
保留 yolo 命令入口，但训练实现共享给 trainer_common。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer_common.profiles import get_profile
from trainer_common.train import run_train
from yolo.console import safe_print


def count_images(path):
    from trainer_common.dataset import count_images as shared_count_images

    return shared_count_images(path)


def main(argv=None):
    run_train(get_profile("runtime_yolo"), safe_print, argv=argv)


if __name__ == "__main__":
    main()
