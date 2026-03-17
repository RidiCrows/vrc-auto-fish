import unittest

from core.yolo_detector import YoloDetector


class YoloDetectorTests(unittest.TestCase):
    def test_normalize_fish_class_name_supports_legacy_aliases(self):
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish"), "fish_black")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_generic"), "fish_black")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_green"), "fish_green")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_copper"), "fish_relic")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_teal"), "fish_clover")
        self.assertEqual(YoloDetector._normalize_fish_class_name("fish_question"), "fish_question")

    def test_select_runtime_device_normalizes_legacy_gpu_name(self):
        self.assertEqual(
            YoloDetector.select_runtime_device("gpu", cuda_available=True),
            (0, "cuda"),
        )
        self.assertEqual(
            YoloDetector.select_runtime_device("auto", cuda_available=False),
            ("cpu", "cpu"),
        )
        self.assertEqual(
            YoloDetector.select_runtime_device("cpu", cuda_available=True),
            ("cpu", "cpu"),
        )

    def test_select_runtime_device_rejects_forced_cuda_without_cuda(self):
        with self.assertRaises(RuntimeError):
            YoloDetector.select_runtime_device("cuda", cuda_available=False)


if __name__ == "__main__":
    unittest.main()
