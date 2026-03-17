import json
import os
import tempfile
import unittest
from io import StringIO
from contextlib import redirect_stdout

import config
from fish_trainer.classes import CLASS_NAMES as FISH_TRAINER_CLASS_NAMES
from fish_trainer.train import count_images as fish_trainer_count_images
from trainer_common.dataset import get_dataset_stats, load_saved_roi
from trainer_common.prune_labels import run_prune
from trainer_common.profiles import get_profile
from yolo.classes import CLASS_NAMES as YOLO_CLASS_NAMES
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

    def test_new_classes_keep_existing_indexes_stable(self):
        self.assertEqual(YOLO_CLASS_NAMES[10], "bar")
        self.assertEqual(YOLO_CLASS_NAMES[11], "track")
        self.assertEqual(YOLO_CLASS_NAMES[12], "progress")
        self.assertEqual(YOLO_CLASS_NAMES[13], "prog_hook")
        self.assertEqual(YOLO_CLASS_NAMES[14], "fish_clover")
        self.assertEqual(YOLO_CLASS_NAMES[15], "fish_question")
        self.assertEqual(YOLO_CLASS_NAMES[0], "fish_black")
        self.assertEqual(YOLO_CLASS_NAMES[2], "fish_relic")
        self.assertEqual(YOLO_CLASS_NAMES[3], "fish_green")

        self.assertEqual(FISH_TRAINER_CLASS_NAMES[10], "bar")
        self.assertEqual(FISH_TRAINER_CLASS_NAMES[11], "track")
        self.assertEqual(FISH_TRAINER_CLASS_NAMES[12], "progress")
        self.assertEqual(FISH_TRAINER_CLASS_NAMES[13], "fish_clover")
        self.assertEqual(FISH_TRAINER_CLASS_NAMES[14], "fish_question")
        self.assertEqual(FISH_TRAINER_CLASS_NAMES[0], "fish_black")
        self.assertEqual(FISH_TRAINER_CLASS_NAMES[2], "fish_relic")
        self.assertEqual(FISH_TRAINER_CLASS_NAMES[3], "fish_green")

    def test_prune_labels_removes_target_boxes_and_moves_empty_images(self):
        profile = get_profile("multicolor")
        dataset_root = profile.dataset_root
        train_img = os.path.join(dataset_root, "images", "train", "fish1.png")
        train_lbl = os.path.join(dataset_root, "labels", "train", "fish1.txt")
        val_img = os.path.join(dataset_root, "images", "val", "fish2.png")
        val_lbl = os.path.join(dataset_root, "labels", "val", "fish2.txt")

        self._touch(train_img)
        os.makedirs(os.path.dirname(train_lbl), exist_ok=True)
        with open(train_lbl, "w", encoding="utf-8") as handle:
            handle.write("2 0.5 0.5 0.2 0.2\n")
            handle.write("10 0.5 0.5 0.4 0.4\n")

        self._touch(val_img)
        os.makedirs(os.path.dirname(val_lbl), exist_ok=True)
        with open(val_lbl, "w", encoding="utf-8") as handle:
            handle.write("13 0.5 0.5 0.2 0.2\n")

        out = StringIO()
        with redirect_stdout(out):
            run_prune(
                profile,
                FISH_TRAINER_CLASS_NAMES,
                print,
                argv=["--classes", "fish_relic", "fish_clover"],
                aliases=config.LEGACY_FISH_KEY_ALIASES,
            )

        with open(train_lbl, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read().strip(), "10 0.5 0.5 0.4 0.4")

        self.assertFalse(os.path.exists(val_lbl))
        self.assertFalse(os.path.exists(val_img))
        self.assertTrue(
            os.path.exists(os.path.join(dataset_root, "images", "unlabeled", "fish2.png"))
        )
        self.assertIn("删除 1 个框", out.getvalue())


if __name__ == "__main__":
    unittest.main()
