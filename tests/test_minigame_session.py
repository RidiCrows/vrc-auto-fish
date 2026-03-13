import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import config
from core.minigame_runtime import MinigameRuntime
from core.minigame_session import MinigameSession


class MinigameSessionTests(unittest.TestCase):
    def make_session(self):
        bot = SimpleNamespace(
            FISH_COLORS={"fish_blue": (255, 0, 0)},
            _current_fish_name="",
            input=SimpleNamespace(mouse_down=MagicMock(), mouse_up=MagicMock()),
        )
        return MinigameSession(bot), bot

    def test_build_detection_context_uses_runtime_thresholds(self):
        session, _bot = self.make_session()
        old_region_x = config.REGION_X
        try:
            config.REGION_X = 60
            ctx = session.build_detection_context(use_yolo=True, skip_success_check=False)
        finally:
            config.REGION_X = old_region_x

        self.assertEqual(ctx.bar_x_half, 60)
        self.assertEqual(ctx.fish_x_half, 120)
        self.assertTrue(ctx.use_yolo)

    def test_stabilize_fish_name_waits_for_required_frames(self):
        session, bot = self.make_session()
        runtime = MinigameRuntime()
        old_frames = config.YOLO_FISH_STABLE_FRAMES
        try:
            config.YOLO_FISH_STABLE_FRAMES = 2
            self.assertEqual(session.stabilize_fish_name("fish_blue", runtime), "")
            self.assertEqual(session.stabilize_fish_name("fish_blue", runtime), "fish_blue")
        finally:
            config.YOLO_FISH_STABLE_FRAMES = old_frames

        bot._current_fish_name = "fish_blue"
        self.assertEqual(session.stabilize_fish_name("fish_blue", runtime), "fish_blue")

    def test_maybe_activate_sets_game_active_and_taps_once(self):
        session, bot = self.make_session()
        runtime = MinigameRuntime()
        ctx = session.build_detection_context(use_yolo=True, skip_success_check=False)
        old_initial_press = config.INITIAL_PRESS_TIME
        old_il_record = config.IL_RECORD
        try:
            config.INITIAL_PRESS_TIME = 0
            config.IL_RECORD = False
            result = session.maybe_activate(
                fish=(1, 2, 3, 4, 0.9),
                bar=(1, 2, 3, 4, 0.9),
                yolo_progress=None,
                runtime=runtime,
                ctx=ctx,
            )
        finally:
            config.INITIAL_PRESS_TIME = old_initial_press
            config.IL_RECORD = old_il_record

        self.assertEqual(result, "ok")
        self.assertTrue(runtime.game_active)
        self.assertTrue(runtime.had_good_detection)
        bot.input.mouse_down.assert_called_once()
        bot.input.mouse_up.assert_called_once()


if __name__ == "__main__":
    unittest.main()
