"""Map stable gestures to actions (media keys, slide keys, built-in commands).

All gesture -> action bindings come from config.yaml (see gesture/config.py); nothing
about the mapping is hardcoded here. Keys are sent with pynput. With dry_run=True (or if
pynput is unavailable) actions are only logged and shown on screen, which is safe for demos.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser

from .config import Binding, bindings_for, default_config, setting


def _legacy_table(cfg: dict) -> dict[str, tuple[str, str]]:
    mode = cfg.get("default_mode", next(iter(cfg["modes"])))
    return {g: (b.action, b.label) for g, b in bindings_for(cfg, mode).items()}


# Kept for backward compatibility: {gesture: (action, label)} for the default mode.
GESTURE_ACTIONS = _legacy_table(default_config())


class ActionDispatcher:
    def __init__(self, dry_run: bool = False, cooldown: float | None = None,
                 repeat_interval: float | None = None, config: dict | None = None,
                 mode: str | None = None):
        self.config = cfg = config if config is not None else default_config()
        self.cooldown = cooldown if cooldown is not None else setting(cfg, "cooldowns", "action", 1.0)
        self.repeat_interval = (repeat_interval if repeat_interval is not None
                                else setting(cfg, "cooldowns", "repeat_interval", 0.35))
        self.modes = list(cfg["modes"])
        self._bindings = {m: bindings_for(cfg, m) for m in self.modes}
        self.mode = mode or cfg.get("default_mode", self.modes[0])
        if self.mode not in self._bindings:
            raise ValueError(f"unknown mode {self.mode!r}; choose from {self.modes}")
        self.last_gesture = "unknown"
        self.last_fire = float("-inf")
        self.log: list[tuple[float, str]] = []
        self._keyboard = None
        self._Key = None
        self.dry_run = True
        self.set_dry_run(dry_run)

    # ------------------------------------------------------------ state
    @property
    def bindings(self) -> dict[str, Binding]:
        """Active gesture -> Binding for the current mode (globals included)."""
        return self._bindings[self.mode]

    def cycle_mode(self) -> str:
        self.mode = self.modes[(self.modes.index(self.mode) + 1) % len(self.modes)]
        return self.mode

    def set_dry_run(self, dry_run: bool) -> bool:
        """Switch dry-run on/off. Going live loads pynput lazily; stays dry if unavailable."""
        if not dry_run and self._keyboard is None:
            try:
                from pynput.keyboard import Controller, Key
                self._keyboard, self._Key = Controller(), Key
            except Exception as e:  # no display / unsupported OS
                print(f"[actions] pynput unavailable ({e}); staying in dry-run.")
                dry_run = True
        self.dry_run = dry_run
        return self.dry_run

    # ------------------------------------------------------------ dispatch
    def update(self, gesture: str, now: float | None = None) -> str | None:
        """Feed the current stable gesture; returns the action label if one fired."""
        now = time.monotonic() if now is None else now
        fired = None
        binding = self.bindings.get(gesture)
        if binding is not None:
            new_gesture = gesture != self.last_gesture
            if new_gesture and now - self.last_fire >= self.cooldown:
                fired = self._fire(binding, now)
            elif not new_gesture and binding.repeat and now - self.last_fire >= self.repeat_interval:
                fired = self._fire(binding, now)
        self.last_gesture = gesture
        return fired

    def trigger(self, gesture: str, now: float | None = None) -> str | None:
        """Fire a one-shot event (e.g. a swipe) if the current mode binds it.

        Unlike update(), this ignores the held-gesture logic; the caller (the swipe
        detector) owns its own cooldown.
        """
        binding = self.bindings.get(gesture)
        if binding is None:
            return None
        return self._fire(binding, time.monotonic() if now is None else now)

    def _fire(self, binding: Binding, now: float) -> str:
        self.last_fire = now
        self.log.append((now, binding.label))
        if binding.command == "toggle_dry_run":  # meta command: always runs
            self.set_dry_run(not self.dry_run)
            return f"{binding.label}: {'DRY-RUN' if self.dry_run else 'LIVE'}"
        if self.dry_run:
            return binding.label
        if binding.command == "screenshot":
            self._screenshot()
        elif binding.command == "open_spotify":
            self._open_spotify()
        elif binding.keys:
            self._press(binding.keys)
        return binding.label

    def _resolve_key(self, name: str):
        key = getattr(self._Key, name, None)
        if key is None and len(name) == 1:
            key = name
        if key is None:
            print(f"[actions] unknown key {name!r} in config.yaml; ignored.")
        return key

    def _press(self, names: tuple[str, ...]):
        keys = [k for k in (self._resolve_key(n) for n in names) if k is not None]
        for k in keys:
            self._keyboard.press(k)
        for k in reversed(keys):
            self._keyboard.release(k)

    # ------------------------------------------------------------ commands
    def _screenshot(self):
        opts = setting(self.config, "commands", "screenshot", {}) or {}
        folder = os.path.expanduser(opts.get("dir", "~/Desktop"))
        path = os.path.join(folder, f"{opts.get('prefix', 'gesture_')}{time.strftime('%Y%m%d_%H%M%S')}.png")

        def run():
            try:
                if sys.platform == "darwin":
                    subprocess.run(["screencapture", "-x", path], check=True, timeout=10)
                else:
                    import pyautogui
                    pyautogui.screenshot(path)
                print(f"[actions] screenshot saved to {path}")
            except Exception as e:
                print(f"[actions] screenshot failed: {e}")
        threading.Thread(target=run, daemon=True).start()  # don't stall the video loop

    def _open_spotify(self):
        opts = setting(self.config, "commands", "spotify", {}) or {}
        app = opts.get("app", "Spotify")
        url = opts.get("fallback_url", "https://music.youtube.com")

        def run():
            if sys.platform == "darwin":
                try:
                    subprocess.run(["open", "-a", app], check=True, timeout=10,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except Exception:
                    print(f"[actions] couldn't open {app}; opening {url} instead.")
            webbrowser.open(url)
        threading.Thread(target=run, daemon=True).start()
