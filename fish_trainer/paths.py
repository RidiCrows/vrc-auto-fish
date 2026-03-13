"""
训练工具路径
============
保留 fish_trainer 目录结构，但底层路径计算共享给 trainer_common。
"""

from trainer_common.dataset import build_dataset_paths, ensure_dataset_dirs as ensure_dirs
from trainer_common.profiles import get_profile


_PROFILE = get_profile("multicolor")
_PATHS = build_dataset_paths(_PROFILE)

APP_ROOT = _PROFILE.app_root
BASE = _PATHS["BASE"]
UNLABELED = _PATHS["UNLABELED"]
TRAIN_IMG = _PATHS["TRAIN_IMG"]
TRAIN_LBL = _PATHS["TRAIN_LBL"]
VAL_IMG = _PATHS["VAL_IMG"]
VAL_LBL = _PATHS["VAL_LBL"]
DATA_YAML = _PATHS["DATA_YAML"]
RUNS_DIR = _PATHS["RUNS_DIR"]


def ensure_dataset_dirs():
    ensure_dirs(_PROFILE)
