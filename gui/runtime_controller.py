"""
GUI 运行时动作
==============
集中处理开始/停止/截图/ROI/轮询等运行时交互。
"""

import os
import threading
import tkinter as tk
from tkinter import ttk

import cv2

import config
from utils.i18n import fish_name, t
from utils.logger import log
from yolo.paths import TRAIN_IMG as YOLO_TRAIN_IMG, UNLABELED as YOLO_UNLABELED


class AppRuntimeController:
    """封装 FishingApp 的运行时动作。"""

    FISH_KEYS = [
        "fish_generic",
        "fish_black",
        "fish_white",
        "fish_copper",
        "fish_green",
        "fish_teal",
        "fish_blue",
        "fish_purple",
        "fish_golden",
        "fish_pink",
        "fish_red",
        "fish_rainbow",
    ]

    def __init__(self, app):
        self.app = app
        self._stats_win = None       # 統計ウィンドウ (Toplevel | None)
        self._stats_canvas = None    # 棒グラフ Canvas
        self._stats_body = None      # テーブル Frame
        self._stats_last_count = -1  # 前回更新時の fish_count

    def tr(self, key: str, default: str | None = None, **kwargs):
        return t(key, default=default, **kwargs)

    def _fish_pairs(self):
        return [(key, fish_name(key)) for key in self.FISH_KEYS]

    @staticmethod
    def has_non_ascii(path: str) -> bool:
        try:
            path.encode("ascii")
            return False
        except UnicodeEncodeError:
            return True

    def on_start(self):
        if self.app.bot.running:
            return
        if self.has_non_ascii(config.BASE_DIR):
            self.app._log_t("runtime.pathNonAscii")
            self.app._log_t("runtime.currentPath", path=config.BASE_DIR)
            self.app._log_t("runtime.moveToAsciiPath")
            return
        if not self.app.bot.window.is_valid():
            if not self.app.bot.window.find():
                self.app._log_t("runtime.vrchatWindowMissing")
                return

        self.app.var_window.set(
            f"{self.app.bot.window.title} (HWND={self.app.bot.window.hwnd})"
        )
        self.app._apply_params()
        self.app.bot.running = True
        self.app.bot.state = "status.running"
        if self.app.bot_thread is None or not self.app.bot_thread.is_alive():
            self.app.bot_thread = threading.Thread(
                target=self.app.bot.run, daemon=True
            )
            self.app.bot_thread.start()

        self.app.btn_start.config(state="disabled")
        self.app.btn_stop.config(state="normal")
        self.app.btn_roi.config(state="disabled")
        self.app.btn_clear_roi.config(state="disabled")
        self.app._log_t("runtime.startFishing")

    def on_stop(self):
        self.app.bot.running = False
        self.app.bot._force_minigame = False
        self.app.bot.input.safe_release()
        self.app.bot.shutdown_debug_overlay()
        self.app.btn_start.config(state="normal")
        self.app.btn_stop.config(state="disabled")
        self.app.btn_roi.config(state="normal")
        self.app.btn_clear_roi.config(state="normal")
        self.app._log_t("runtime.stopFishing")
        self.save_log_async()

    def on_toggle_debug(self):
        self.app.bot.debug_mode = not self.app.bot.debug_mode
        tag = self.tr("status.on") if self.app.bot.debug_mode else self.tr("status.off")
        self.app.var_debug.set(tag)
        self.app._log_t("runtime.debugModeChanged", state=tag)
        if self.app.bot.debug_mode:
            self.app._log_t("runtime.debugHint")

    def on_connect(self):
        if self.app.bot.window.find():
            self.app.var_window.set(
                f"{self.app.bot.window.title} (HWND={self.app.bot.window.hwnd})"
            )
            self.app.bot.screen.reset_capture_method()
            self.app._log_t("runtime.connected", title=self.app.bot.window.title)
            return
        self.app.var_window.set(self.tr("status.notFound"))
        self.app._log_t("runtime.windowNotFound")

    def screen_capture_safe(self):
        try:
            return self.app.bot.screen.grab_window(self.app.bot.window)
        except Exception as e:
            self.app._log_t("runtime.screenshotException", error=e)
            return None, None

    def on_screenshot(self):
        if not self.app.bot.window.is_valid():
            if not self.app.bot.window.find():
                self.app._log_t("runtime.screenshotWindowMissing")
                return
        img, region = self.screen_capture_safe()
        if img is None:
            self.app._log_t("runtime.screenshotFailed")
            return
        self.app.bot.screen.save_debug(img, "manual_screenshot")
        h, w = img.shape[:2]
        self.app._log_t("runtime.screenshotSaved", width=w, height=h)
        if region:
            self.app._log_t(
                "runtime.windowRegion",
                x=region[0], y=region[1], w=region[2], h=region[3]
            )

    def on_clear_log(self):
        self.app.txt_log.config(state="normal")
        self.app.txt_log.delete("1.0", "end")
        self.app.txt_log.config(state="disabled")

    def on_whitelist(self):
        win = tk.Toplevel(self.app.root)
        win.title(self.tr("runtime.whitelistTitle"))
        win.resizable(False, False)
        win.transient(self.app.root)
        win.grab_set()
        ttk.Label(win, text=self.tr("runtime.whitelistPrompt")).pack(pady=(10, 5))

        wl = config.FISH_WHITELIST
        chk_vars = {}
        body = ttk.Frame(win)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        for col in range(2):
            body.columnconfigure(col, weight=1)

        for i, (key, name) in enumerate(self._fish_pairs()):
            var = tk.BooleanVar(value=wl.get(key, True))
            chk_vars[key] = var
            row = i // 2
            col = i % 2
            ttk.Checkbutton(body, text=name, variable=var).grid(
                row=row, column=col, sticky="w", padx=12, pady=4
            )

        def apply_changes():
            for key, var in chk_vars.items():
                config.FISH_WHITELIST[key] = var.get()
            self.app._save_settings()
            enabled = [n for (k, n) in self._fish_pairs() if chk_vars[k].get()]
            self.app._log_t("runtime.whitelistUpdated", names=", ".join(enabled))
            win.destroy()

        ttk.Button(win, text=self.tr("runtime.confirm"), command=apply_changes).pack(pady=10)
        win.update_idletasks()
        req_w = max(win.winfo_reqwidth() + 20, 260)
        req_h = max(win.winfo_reqheight() + 10, 240)
        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        final_w = min(req_w, screen_w - 80)
        final_h = min(req_h, screen_h - 80)
        x = max((screen_w - final_w) // 2, 0)
        y = max((screen_h - final_h) // 2, 0)
        win.geometry(f"{final_w}x{final_h}+{x}+{y}")

    BAR_COLORS = {
        "fish_generic": "#a0a0a0",
        "fish_black":   "#3a3a3a",
        "fish_white":   "#e0e0e0",
        "fish_copper":  "#b87333",
        "fish_green":   "#4caf50",
        "fish_teal":    "#009688",
        "fish_blue":    "#2196f3",
        "fish_purple":  "#9c27b0",
        "fish_golden":  "#ffc107",
        "fish_pink":    "#e91e8f",
        "fish_red":     "#f44336",
        "fish_rainbow": "#ff9800",
    }

    def _draw_stats_chart(self, canvas, pairs, stats):
        """Canvas に積み上げ棒グラフを描画する。"""
        canvas.delete("all")
        bar_w = 34
        gap = 16
        chart_h = 160
        top_margin = 26
        bottom_margin = 55
        left_margin = 10

        n = len(pairs)
        canvas_w = left_margin + n * (bar_w + gap) + gap
        canvas_h = top_margin + chart_h + bottom_margin
        canvas.config(width=canvas_w, height=canvas_h)

        # 最大値を求めてスケール計算
        max_val = 0
        for key, _ in pairs:
            entry = stats.get(key, {})
            total = entry.get("success", 0) + entry.get("fail", 0)
            if total > max_val:
                max_val = total
        if max_val == 0:
            max_val = 1

        baseline = top_margin + chart_h
        # ベースライン
        canvas.create_line(
            left_margin, baseline,
            left_margin + n * (bar_w + gap) + gap, baseline,
            fill="#888888",
        )

        success_label = self.tr("runtime.statsSuccess")
        fail_label = self.tr("runtime.statsFail")

        for i, (key, display_name) in enumerate(pairs):
            entry = stats.get(key, {})
            s = entry.get("success", 0)
            f = entry.get("fail", 0)
            total = s + f

            x0 = left_margin + gap + i * (bar_w + gap)
            x1 = x0 + bar_w
            bar_color = self.BAR_COLORS.get(key, "#a0a0a0")

            # 成功部分（下）
            s_h = int(chart_h * s / max_val) if s else 0
            f_h = int(chart_h * f / max_val) if f else 0

            y_success_top = baseline - s_h
            if s > 0:
                canvas.create_rectangle(
                    x0, y_success_top, x1, baseline,
                    fill=bar_color, outline=bar_color,
                )
            # 失敗部分（上に積む）
            y_fail_top = y_success_top - f_h
            if f > 0:
                canvas.create_rectangle(
                    x0, y_fail_top, x1, y_success_top,
                    fill=bar_color, outline=bar_color, stipple="gray50",
                )

            # バー上に合計数
            if total > 0:
                canvas.create_text(
                    (x0 + x1) // 2, y_fail_top - 10,
                    text=str(total), font=("", 9), fill="#333333",
                )

            # 魚名ラベル（縦書き風に1文字ずつ改行）
            short = display_name.lstrip("🐟 ").strip()
            label = "\n".join(short[:4])
            canvas.create_text(
                (x0 + x1) // 2, baseline + 5,
                text=label, font=("", 8), anchor="n", fill="#333333",
            )

        # 凡例
        lx = left_margin + gap
        ly = top_margin - 4
        canvas.create_rectangle(lx, ly - 10, lx + 12, ly, fill="#4caf50", outline="#4caf50")
        canvas.create_text(lx + 16, ly - 5, text=success_label, font=("", 9), anchor="w")
        lx2 = lx + 70
        canvas.create_rectangle(lx2, ly - 10, lx2 + 12, ly, fill="#4caf50", outline="#4caf50", stipple="gray50")
        canvas.create_text(lx2 + 16, ly - 5, text=fail_label, font=("", 9), anchor="w")

    def _populate_stats_table(self, body):
        """テーブルフレームの中身を再描画する。"""
        for child in body.winfo_children():
            child.destroy()

        body.columnconfigure(0, weight=1)
        for c in range(1, 4):
            body.columnconfigure(c, weight=0)

        stats = self.app.bot.fish_stats
        pairs = self._fish_pairs()

        hdr_pad = {"padx": 8, "pady": 2}
        hdr_font = ("", 10, "bold")
        cell_font = ("", 10)
        total_font = ("", 10, "bold")
        row = 0

        ttk.Label(body, text="", anchor="w").grid(row=row, column=0, sticky="w", **hdr_pad)
        ttk.Label(body, text=self.tr("runtime.statsSuccess"), anchor="e",
                  font=hdr_font).grid(row=row, column=1, sticky="e", **hdr_pad)
        ttk.Label(body, text=self.tr("runtime.statsFail"), anchor="e",
                  font=hdr_font).grid(row=row, column=2, sticky="e", **hdr_pad)
        ttk.Label(body, text=self.tr("runtime.statsCount"), anchor="e",
                  font=hdr_font).grid(row=row, column=3, sticky="e", **hdr_pad)
        row += 1

        total_success = 0
        total_fail = 0
        for key, display_name in pairs:
            entry = stats.get(key, {})
            s = entry.get("success", 0)
            f = entry.get("fail", 0)
            total_success += s
            total_fail += f
            ttk.Label(body, text=display_name, anchor="w", font=cell_font).grid(
                row=row, column=0, sticky="w", **hdr_pad
            )
            ttk.Label(body, text=str(s), anchor="e", font=cell_font).grid(
                row=row, column=1, sticky="e", **hdr_pad
            )
            ttk.Label(body, text=str(f), anchor="e", font=cell_font).grid(
                row=row, column=2, sticky="e", **hdr_pad
            )
            ttk.Label(body, text=str(s + f), anchor="e", font=cell_font).grid(
                row=row, column=3, sticky="e", **hdr_pad
            )
            row += 1

        sep = ttk.Separator(body, orient="horizontal")
        sep.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=4)
        row += 1

        ttk.Label(body, text=self.tr("runtime.statsTotal"), anchor="w",
                  font=total_font).grid(row=row, column=0, sticky="w", **hdr_pad)
        ttk.Label(body, text=str(total_success), anchor="e",
                  font=total_font).grid(row=row, column=1, sticky="e", **hdr_pad)
        ttk.Label(body, text=str(total_fail), anchor="e",
                  font=total_font).grid(row=row, column=2, sticky="e", **hdr_pad)
        ttk.Label(body, text=str(total_success + total_fail), anchor="e",
                  font=total_font).grid(row=row, column=3, sticky="e", **hdr_pad)

        if config.SKIP_SUCCESS_CHECK:
            row += 1
            tk.Label(body, text=self.tr("runtime.statsSkipNote"),
                     foreground="gray", font=("", 9)).grid(
                row=row, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 0))

    def _refresh_stats_dialog(self):
        """統計ウィンドウが開いていれば、グラフとテーブルを再描画する。"""
        if self._stats_win is None:
            return
        try:
            if not self._stats_win.winfo_exists():
                self._stats_win = None
                return
        except tk.TclError:
            self._stats_win = None
            return
        stats = self.app.bot.fish_stats
        pairs = self._fish_pairs()
        self._draw_stats_chart(self._stats_canvas, pairs, stats)
        self._populate_stats_table(self._stats_body)

    def on_stats(self):
        # 既に開いていればフォーカスだけ当てる
        if self._stats_win is not None:
            try:
                if self._stats_win.winfo_exists():
                    self._stats_win.lift()
                    self._stats_win.focus_force()
                    return
            except tk.TclError:
                pass
            self._stats_win = None

        win = tk.Toplevel(self.app.root)
        win.title(self.tr("runtime.statsTitle"))
        win.resizable(False, False)
        win.transient(self.app.root)

        def _on_close_stats():
            self._stats_win = None
            self._stats_canvas = None
            self._stats_body = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close_stats)

        stats = self.app.bot.fish_stats
        pairs = self._fish_pairs()

        # ── 棒グラフ ──
        chart = tk.Canvas(win, bg="#f5f5f5", highlightthickness=0)
        chart.pack(padx=12, pady=(10, 4))
        self._draw_stats_chart(chart, pairs, stats)

        # ── テーブル ──
        body = ttk.Frame(win)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self._populate_stats_table(body)

        ttk.Button(win, text=self.tr("runtime.confirm"), command=_on_close_stats).pack(pady=10)

        # 参照を保持
        self._stats_win = win
        self._stats_canvas = chart
        self._stats_body = body
        self._stats_last_count = self.app.bot.fish_count

        win.update_idletasks()
        req_w = max(win.winfo_reqwidth() + 20, 600)
        req_h = max(win.winfo_reqheight() + 10, 420)
        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        final_w = min(req_w, screen_w - 80)
        final_h = min(req_h, screen_h - 80)
        x = max((screen_w - final_w) // 2, 0)
        y = max((screen_h - final_h) // 2, 0)
        win.geometry(f"{final_w}x{final_h}+{x}+{y}")

    def on_topmost(self):
        topmost = self.app.var_topmost.get()
        self.app.root.wm_attributes("-topmost", 1 if topmost else 0)
        if not topmost:
            self.app.root.lift()
            self.app.root.focus_force()

    def preload_yolo(self):
        def load():
            try:
                from core.bot import _get_yolo_detector
                self.app.bot.yolo = _get_yolo_detector()
            except Exception as e:
                self.app._log_t("runtime.yoloPreloadFailed", error=e)

        threading.Thread(target=load, daemon=True).start()

    def update_yolo_status(self):
        """更新 YOLO 状态显示。"""
        model_ok = os.path.exists(config.YOLO_MODEL)
        unlabeled = YOLO_UNLABELED
        train = YOLO_TRAIN_IMG
        n_unlabeled = len([
            f for f in os.listdir(unlabeled)
            if f.endswith((".png", ".jpg"))
        ]) if os.path.isdir(unlabeled) else 0
        n_train = len([
            f for f in os.listdir(train)
            if f.endswith((".png", ".jpg"))
        ]) if os.path.isdir(train) else 0

        parts = [
            self.tr("yolo.modelOk") if model_ok else self.tr("yolo.modelMissing"),
            self.tr("yolo.trainCount", count=n_train),
            self.tr("yolo.unlabeledCount", count=n_unlabeled),
        ]
        self.app.var_yolo_status.set(" | ".join(parts))

    def on_select_roi(self):
        if not self.app.bot.window.is_valid():
            if not self.app.bot.window.find():
                self.app._log_t("runtime.connectFirst")
                return
        img, _ = self.screen_capture_safe()
        if img is None:
            self.app._log_t("runtime.roiCaptureFailed")
            return

        self.app._log_t("runtime.roiPrompt")

        def select_worker(snap):
            win_name = self.tr("runtime.roiSelectWindow")
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            h, w = snap.shape[:2]
            dw = min(w, 1280)
            dh = int(h * dw / w)
            cv2.resizeWindow(win_name, dw, dh)
            roi = cv2.selectROI(win_name, snap, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            x, y, w_r, h_r = [int(v) for v in roi]
            if w_r > 10 and h_r > 10:
                roi_value = [x, y, w_r, h_r]
                self.app.root.after(0, lambda: self.app.settings_store.apply_detect_roi(roi_value))
                self.app.root.after(0, self.app._save_settings)
                self.app._log_t("runtime.roiSet", x=x, y=y, w=w_r, h=h_r)
            else:
                self.app._log_t("runtime.roiCancelled")

        threading.Thread(target=select_worker, args=(img,), daemon=True, name="ROISelect").start()

    def on_clear_roi(self):
        self.app.settings_store.apply_detect_roi(None)
        self.app._save_settings()
        self.app._log_t("runtime.roiCleared")

    def poll(self):
        try:
            for _ in range(20):
                msg = log.log_queue.get_nowait()
                self.app._append_log(msg)
        except Exception:
            pass

        self.app.var_state.set(self.app._translate_bot_state(self.app.bot.state))
        current_count = self.app.bot.fish_count
        self.app.var_count.set(str(current_count))
        if current_count != self._stats_last_count:
            self._stats_last_count = current_count
            self._refresh_stats_dialog()
        self.app.var_debug.set(
            self.tr("status.on") if self.app.bot.debug_mode else self.tr("status.off")
        )
        self.app.lbl_state.config(
            foreground="green" if self.app.bot.running else "gray"
        )

        if self.app.bot_thread and not self.app.bot_thread.is_alive() and self.app.bot.running:
            self.app.bot.running = False
            self.app.btn_start.config(state="normal")
            self.app.btn_stop.config(state="disabled")
            self.app.btn_roi.config(state="normal")
            self.app.btn_clear_roi.config(state="normal")

        self.app.root.after(100, self.poll)

    def on_close(self):
        self.app.bot.running = False
        self.app.bot._force_minigame = False
        self.app.bot.input.safe_release()
        self.app.bot.shutdown_debug_overlay()
        self.app._save_settings()
        self.save_log()
        self.app.root.destroy()

    def save_log(self):
        path = os.path.join(config.DEBUG_DIR, "last_run.log")
        log.save(path)
        self.app._log_t("runtime.logSaved", path=path)

    def save_log_async(self):
        path = os.path.join(config.DEBUG_DIR, "last_run.log")

        def worker():
            log.save(path)
            try:
                self.app.root.after(0, lambda: self.app._log_t("runtime.logSaved", path=path))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()
