import os
import tempfile
import unittest

import config
from utils.i18n import (
    fish_name,
    get_language,
    normalize_language,
    read_persisted_language,
    set_language,
    t,
    write_persisted_language,
)


class I18nTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_settings_file = config.SETTINGS_FILE
        self.old_language = config.LANGUAGE
        config.SETTINGS_FILE = os.path.join(self.tmpdir.name, "settings.json")
        set_language("zh-CN")

    def tearDown(self):
        config.SETTINGS_FILE = self.old_settings_file
        set_language(self.old_language)
        self.tmpdir.cleanup()

    def test_set_language_translates_basic_keys(self):
        set_language("en-US")
        self.assertEqual(get_language(), "en-US")
        self.assertEqual(t("status.ready"), "Ready")

        set_language("ja-JP")
        self.assertEqual(get_language(), "ja-JP")
        self.assertEqual(t("status.ready"), "準備完了")

        set_language("zh-CN")
        self.assertEqual(t("status.ready"), "就绪")

    def test_write_and_read_persisted_language(self):
        write_persisted_language("en-US")
        self.assertEqual(read_persisted_language(), "en-US")

        write_persisted_language("ja")
        self.assertEqual(read_persisted_language(), "ja-JP")

    def test_normalize_language_supports_japanese_aliases(self):
        self.assertEqual(normalize_language("ja"), "ja-JP")
        self.assertEqual(normalize_language("ja-jp"), "ja-JP")
        self.assertEqual(normalize_language("jp"), "ja-JP")

    def test_fish_teal_name_is_available_in_all_languages(self):
        set_language("zh-CN")
        self.assertEqual(fish_name("fish_teal"), "青绿色鱼")

        set_language("en-US")
        self.assertEqual(fish_name("fish_teal"), "Teal Fish")

        set_language("ja-JP")
        self.assertEqual(fish_name("fish_teal"), "青緑魚")


if __name__ == "__main__":
    unittest.main()
