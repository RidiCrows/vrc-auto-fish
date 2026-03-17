"""
YOLO 多颜色鱼训练数据采集
==========================
保留 yolo 命令入口，但采集实现共享给 trainer_common。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer_common.collect import run_collect
from trainer_common.profiles import get_profile, TrainerProfile
from yolo.console import safe_print


class CustomPathProfile:
    """包装 TrainerProfile，支持自定义数据目录"""
    def __init__(self, base_profile, custom_dataset_root=None):
        self._base_profile = base_profile
        self._custom_dataset_root = custom_dataset_root
    
    def __getattr__(self, name):
        """代理所有属性访问到 base_profile"""
        return getattr(self._base_profile, name)
    
    @property
    def dataset_root(self) -> str:
        """返回自定义的数据目录或默认值"""
        if self._custom_dataset_root:
            return self._custom_dataset_root
        return self._base_profile.dataset_root


def build_parser():
    from trainer_common.collect import build_parser
    parser = build_parser(get_profile("runtime_yolo"))
    # 添加 --base-dir 参数支持
    parser.add_argument("--base-dir", type=str, default="", help="数据目录（可选，默认使用配置路径）")
    return parser


def main(argv=None):
    import argparse
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    
    profile = get_profile("runtime_yolo")
    
    # 如果提供了自定义数据目录，创建自定义 profile
    if args.base_dir:
        base_dir = os.path.abspath(args.base_dir)
        profile = CustomPathProfile(profile, custom_dataset_root=base_dir)
    
    # 构建过滤后的 argv（移除 --base-dir 参数），用于 trainer_common.collect
    filtered_argv = [arg for arg in (argv or []) if not arg.startswith("--base-dir")]
    run_collect(profile, safe_print, argv=filtered_argv)


if __name__ == "__main__":
    main()
