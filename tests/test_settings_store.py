import os
import tempfile
import unittest
from types import SimpleNamespace
import json

import config
from gui.settings_store import AppSettingsStore
from utils.i18n import set_language


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeEntry:
    def __init__(self):
        self.states = []

    def state(self, state_value):
        self.states.append(tuple(state_value))


class SettingsStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.settings_file = os.path.join(self.tmpdir.name, "settings.json")
        self.old_settings_file = config.SETTINGS_FILE
        self.old_language = config.LANGUAGE
        self.old_full_rate_wait_hook = config.FULL_RATE_WAIT_HOOK
        config.SETTINGS_FILE = self.settings_file
        config.LANGUAGE = "zh-CN"
        config.FULL_RATE_WAIT_HOOK = False
        self.log_messages = []
        self.rebuild_calls = []
        self.app = SimpleNamespace(
            _param_vars={
                "HOLD_MIN_S": (FakeVar("30"), "ms"),
                "SUCCESS_PROGRESS": (FakeVar("60"), "pct"),
            },
            _param_entries={"SUCCESS_PROGRESS": FakeEntry()},
            PARAM_DEFAULTS={"HOLD_MIN_S": 0.025, "SUCCESS_PROGRESS": 0.55},
            SETTINGS_DEFAULTS={
                "SKIP_SUCCESS_CHECK": False,
                "SYNC_PD_MODE": True,
                "FULL_RATE_WAIT_HOOK": False,
            },
            PERSISTED_CONFIG_ATTRS=(
                "LANGUAGE",
                "SKIP_SUCCESS_CHECK",
                "SYNC_PD_MODE",
                "FULL_RATE_WAIT_HOOK",
            ),
            var_grouped_params=FakeVar(True),
            var_preset_name=FakeVar(""),
            var_language=FakeVar("简体中文"),
            var_skip_success=FakeVar(False),
            var_sync_pd_mode=FakeVar(True),
            var_anti_mode=FakeVar("jump"),
            var_shake_time=FakeVar("0.020"),
            var_full_rate_wait_hook=FakeVar(False),
            _log_msg=self.log_messages.append,
            _log_t=lambda key, **kwargs: self.log_messages.append((key, kwargs)),
            _update_success_threshold_state=lambda: None,
            _render_params_panel=lambda: None,
            _update_window_title=lambda: None,
            _refresh_language_choices=lambda: None,
            _rebuild_ui_for_language=lambda: self.rebuild_calls.append(True),
        )
        self.store = AppSettingsStore(self.app)
        self.old_hold_min = config.HOLD_MIN_S
        self.old_success_progress = config.SUCCESS_PROGRESS

    def tearDown(self):
        config.SETTINGS_FILE = self.old_settings_file
        config.LANGUAGE = self.old_language
        config.FULL_RATE_WAIT_HOOK = self.old_full_rate_wait_hook
        set_language(self.old_language)
        config.HOLD_MIN_S = self.old_hold_min
        config.SUCCESS_PROGRESS = self.old_success_progress
        self.tmpdir.cleanup()

    def test_apply_params_updates_config_and_saves_file(self):
        self.store.apply_params()
        self.assertAlmostEqual(config.HOLD_MIN_S, 0.03)
        self.assertAlmostEqual(config.SUCCESS_PROGRESS, 0.6)
        self.assertTrue(os.path.exists(self.settings_file))
        with open(self.settings_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("current", data)

    def test_reset_params_restores_defaults(self):
        config.HOLD_MIN_S = 0.08
        config.SUCCESS_PROGRESS = 0.9
        self.store.save()

        self.store.reset_params()

        self.assertAlmostEqual(config.HOLD_MIN_S, 0.025)
        self.assertAlmostEqual(config.SUCCESS_PROGRESS, 0.55)
        self.assertTrue(os.path.exists(self.settings_file))

    def test_apply_loaded_setting_updates_widget_display(self):
        self.store.apply_loaded_setting("HOLD_MIN_S", 0.04)
        var, _ = self.app._param_vars["HOLD_MIN_S"]
        self.assertEqual(var.get(), "40")

    def test_save_and_load_preset(self):
        self.app.var_preset_name.set("测试预设")
        self.store.apply_params()
        self.store.save_preset("测试预设")

        self.app._param_vars["HOLD_MIN_S"][0].set("9")
        self.store.apply_params()
        self.assertAlmostEqual(config.HOLD_MIN_S, 0.009)

        loaded = self.store.load_preset("测试预设")
        self.assertTrue(loaded)
        self.assertAlmostEqual(config.HOLD_MIN_S, 0.03)
        self.assertEqual(self.store.get_active_preset_name(), "测试预设")

    def test_delete_preset(self):
        self.store.save_preset("预设A")
        self.assertIn("预设A", self.store.get_preset_names())
        deleted = self.store.delete_preset("预设A")
        self.assertTrue(deleted)
        self.assertNotIn("预设A", self.store.get_preset_names())

    def test_apply_loaded_setting_updates_language_and_rebuilds(self):
        handled = self.store.apply_loaded_setting("LANGUAGE", "en-US")
        self.assertTrue(handled)
        self.assertEqual(config.LANGUAGE, "en-US")
        self.assertEqual(len(self.rebuild_calls), 1)

    def test_collect_settings_includes_language(self):
        config.LANGUAGE = "en-US"
        data = self.store.collect_settings_data()
        self.assertEqual(data["LANGUAGE"], "en-US")

    def test_apply_loaded_whitelist_migrates_legacy_fish_keys(self):
        handled = self.store.apply_loaded_setting(
            "FISH_WHITELIST",
            {"fish_teal": False, "fish_copper": True, "fish_generic": False},
        )
        self.assertTrue(handled)
        self.assertFalse(config.FISH_WHITELIST["fish_clover"])
        self.assertTrue(config.FISH_WHITELIST["fish_relic"])
        self.assertFalse(config.FISH_WHITELIST["fish_black"])

    def test_apply_loaded_setting_normalizes_legacy_gpu_device_name(self):
        handled = self.store.apply_loaded_setting("YOLO_DEVICE", "gpu")
        self.assertTrue(handled)
        self.assertEqual(config.YOLO_DEVICE, "cuda")

    def test_apply_loaded_setting_updates_full_rate_wait_hook(self):
        handled = self.store.apply_loaded_setting("FULL_RATE_WAIT_HOOK", True)
        self.assertTrue(handled)
        self.assertTrue(config.FULL_RATE_WAIT_HOOK)
        self.assertTrue(self.app.var_full_rate_wait_hook.get())


if __name__ == "__main__":
    unittest.main()
