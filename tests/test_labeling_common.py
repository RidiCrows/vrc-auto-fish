import os
import tempfile
import unittest

from trainer_common.labeling import (
    build_label_parser,
    list_relabel_entries,
    list_unlabeled_entries,
    load_existing_labels,
    save_new_labeled_entry,
    write_yolo_labels,
)


class LabelingCommonTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def _touch(self, path: str, content: bytes = b"data"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)

    def test_build_label_parser_supports_shared_args(self):
        parser = build_label_parser("demo")
        args = parser.parse_args(["--split", "0.3", "--relabel"])
        self.assertAlmostEqual(args.split, 0.3)
        self.assertTrue(args.relabel)

    def test_load_and_write_yolo_labels_round_trip(self):
        lbl_path = os.path.join(self.root, "labels", "demo.txt")
        boxes = [(1, 10, 20, 50, 60), (2, 0, 0, 100, 80)]
        write_yolo_labels(lbl_path, (100, 200, 3), boxes)

        loaded = load_existing_labels(lbl_path, 200, 100)

        self.assertEqual(len(loaded), len(boxes))
        for loaded_box, expected_box in zip(loaded, boxes):
            self.assertEqual(loaded_box[0], expected_box[0])
            for actual, expected in zip(loaded_box[1:], expected_box[1:]):
                self.assertLessEqual(abs(actual - expected), 1)

    def test_list_relabel_entries_collects_train_and_val(self):
        train_img = os.path.join(self.root, "images", "train")
        train_lbl = os.path.join(self.root, "labels", "train")
        val_img = os.path.join(self.root, "images", "val")
        val_lbl = os.path.join(self.root, "labels", "val")
        self._touch(os.path.join(train_img, "a.png"))
        self._touch(os.path.join(val_img, "b.jpg"))

        entries = list_relabel_entries(train_img, train_lbl, val_img, val_lbl)

        self.assertEqual(len(entries), 2)
        self.assertTrue(entries[0]["lbl_path"].endswith(".txt"))

    def test_list_unlabeled_entries_filters_non_images(self):
        unlabeled = os.path.join(self.root, "images", "unlabeled")
        self._touch(os.path.join(unlabeled, "a.png"))
        self._touch(os.path.join(unlabeled, "b.txt"))

        entries = list_unlabeled_entries(unlabeled)

        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["img_path"].endswith("a.png"))

    def test_save_new_labeled_entry_moves_image_and_writes_label(self):
        entry = {"img_path": os.path.join(self.root, "unlabeled", "a.png"), "lbl_path": None}
        self._touch(entry["img_path"])
        train_img = os.path.join(self.root, "images", "train")
        train_lbl = os.path.join(self.root, "labels", "train")
        val_img = os.path.join(self.root, "images", "val")
        val_lbl = os.path.join(self.root, "labels", "val")
        for path in (train_img, train_lbl, val_img, val_lbl):
            os.makedirs(path, exist_ok=True)
        logs = []

        dst_img_path, dst_lbl_path = save_new_labeled_entry(
            entry,
            (100, 200, 3),
            [(1, 10, 20, 50, 60)],
            0.0,
            train_img,
            train_lbl,
            val_img,
            val_lbl,
            logs.append,
        )

        self.assertTrue(os.path.exists(dst_img_path))
        self.assertTrue(os.path.exists(dst_lbl_path))
        self.assertFalse(os.path.exists(entry["img_path"]))
        self.assertIn("train", logs[0])


if __name__ == "__main__":
    unittest.main()
