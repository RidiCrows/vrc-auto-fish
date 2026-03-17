"""
fish_trainer 已标注数据清理入口
==============================
保留 fish_trainer 命令入口，但底层清理逻辑共享给 trainer_common。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from fish_trainer.classes import CLASS_NAMES
from fish_trainer.console import safe_print
from trainer_common.prune_labels import run_prune
from trainer_common.profiles import get_profile


def main(argv=None):
    run_prune(
        get_profile("multicolor"),
        CLASS_NAMES,
        safe_print,
        argv=argv,
        aliases=config.LEGACY_FISH_KEY_ALIASES,
    )


if __name__ == "__main__":
    main()
