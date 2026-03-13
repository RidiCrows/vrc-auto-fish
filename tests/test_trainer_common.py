import json
import os
import tempfile
import unittest

import config
from fish_trainer.train import count_images as fish_trainer_count_images
from trainer_common.dataset import get_dataset_stats, load_saved_roi
from trainer_common.profiles import get_profile
from yolo.train import count_images as yolo_count_images


class TrainerCommonTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_base_dir = config.BASE_DIR
        self.old_settings_file = config.SETTINGS_FILE
        config.BASE_DIR = self.tmpdir.name
        config.SETTINGS_FILE = os.path.join(self.tmpdir.name, "settings.json")

    def tearDown(self):
        config.BASE_DIR = self.old_base_dir
        config.SETTINGS_FILE = self.old_settings_file
        self.tmpdir.cleanup()

    def _touch(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"test")

    def test_count_images_is_shared_and_counts_files(self):
        img_dir = os.path.join(self.tmpdir.name, "images")
        self._touch(os.path.join(img_dir, "a.png"))
        self._touch(os.path.join(img_dir, "b.jpg"))
        self._touch(os.path.join(img_dir, "ignore.txt"))

        self.assertEqual(yolo_count_images(img_dir), 2)
        self.assertEqual(fish_trainer_count_images(img_dir), 2)

    def test_load_saved_roi_supports_new_settings_layout(self):
        with open(config.SETTINGS_FILE, "w", encoding="utf-8") as handle:
            json.dump({"current": {"DETECT_ROI": [1, 2, 3, 4]}}, handle)

        self.assertEqual(load_saved_roi(), [1, 2, 3, 4])

    def test_get_dataset_stats_uses_profile_layout(self):
        profile = get_profile("multicolor")
        dataset_root = profile.dataset_root
        self._touch(os.path.join(dataset_root, "images", "unlabeled", "u1.png"))
        self._touch(os.path.join(dataset_root, "images", "train", "t1.png"))
        self._touch(os.path.join(dataset_root, "labels", "train", "t1.txt"))
        self._touch(os.path.join(dataset_root, "images", "val", "v1.png"))
        self._touch(os.path.join(dataset_root, "labels", "val", "v1.txt"))

        stats = get_dataset_stats(profile)

        self.assertEqual(stats["unlabeled_images"], 1)
        self.assertEqual(stats["train_images"], 1)
        self.assertEqual(stats["train_labels"], 1)
        self.assertEqual(stats["val_images"], 1)
        self.assertEqual(stats["val_labels"], 1)
        self.assertEqual(stats["labeled_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
