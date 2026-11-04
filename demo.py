"""Webcam demo: real-time hand gesture recognition mapped to media controls.

Usage:
    python demo.py                 # webcam 0, sends real media keys
    python demo.py --dry-run       # only show/log actions, no key presses
    python demo.py --camera 1      # another camera
    python demo.py --video clip.mp4 --record out.mp4
    python demo.py --classifier ml # trained model (train.py) instead of the rules

Keys: q / Esc = quit, d = toggle dry-run, m = cycle mode (media / slides / mouse),
      h = legend, g = classifier debug overlay, p = save the current hand + its expected
      label to tests/fixtures/real_hands.json. Mappings and tuning live in config.yaml.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import Counter, OrderedDict, deque

import cv2
import numpy as np

from gesture import ActionDispatcher, GestureSmoother, load_config
from gesture import motion
from gesture.classifier import HoldThrough, PoseTracker
from gesture.config import setting
from gesture.debug import DEFAULT_FIXTURES, append_fixture, debug_lines, state_string
from gesture.features import KEY_LABELS
from gesture import display as D
from gesture.hand_tracker import HandTracker, draw_hand, hand_box, lerp_color, palm_center, rounded_rect

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Pillow missing -> Hershey fallback
    Image = ImageDraw = ImageFont = None

# ------------------------------------------------------------------ theme (BGR)
PINK = (216, 79, 255)       # #FF4FD8
PURPLE = (246, 92, 139)     # #8B5CF6
GLASS = (34, 14, 26)        # deep plum, blended semi-transparent
GLASS_ALPHA = 0.62
TRACK = (80, 50, 70)
WHITE = (255, 255, 255)
MUTED = (235, 205, 225)     # soft lavender text
AMBER = (32, 176, 255)      # #FFB020
GREEN = (153, 211, 52)      # #34D399
HIGHLIGHT = (150, 60, 200)  # pink-tinted glass for the active legend row

ICONS = {
    "open_palm": "✋", "fist": "✊", "thumbs_up": "👍", "thumbs_down": "👎",
    "peace": "✌️", "pointing": "☝️", "rock": "🤟", "ok": "👌", "call_me": "🤙",
    "swipe_right": "👉", "swipe_left": "👈", "pinch": "🤏", "unknown": "🤔", "no hand": "🫥",
}
LEGEND_NAMES = {"rock": "🤟/🤘 Rock (thumb out or in)"}  # legend wording where it differs
MODE_STYLE = {"media": ("🎧", PINK), "slides": ("🖥️", PURPLE), "mouse": ("🖱️", (212, 182, 45))}
CLASSIFIER_STYLE = {"RULES": ("📐", MUTED), "ML": ("🧠", (255, 150, 120))}  # lavender / periwinkle
TOAST_SECONDS = 1.0
PULSE_SECONDS = 0.6


def pretty(g: str) -> str:
    return g.replace("_", " ").title()


class ConfidenceSmoother(GestureSmoother):
    """GestureSmoother + a read-only `confidence` (agreement ratio of the top label).

    Subclassed here so gesture/classifier.py and its voting behaviour stay untouched.
    The ratio uses the full window size, so the bar fills up while the window warms up.
    """

    @property
    def confidence(self) -> float:
        if not self.history:
            return 0.0
        _, count = Counter(self.history).most_common(1)[0]
        return count / self.history.maxlen


# ------------------------------------------------------------------ text rendering
FONT_CANDIDATES = [
    ("/System/Library/Fonts/SFNS.ttf", 0, 0),           # (path, regular index, bold index)
    ("/System/Library/Fonts/Helvetica.ttc", 0, 1),
]
EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"


def _is_emoji(ch: str) -> bool:
    o = ord(ch)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x231A <= o <= 0x23FF or o in (0xFE0F, 0x200D)


class TextRenderer:
    """Anti-aliased text via Pillow, cached as BGRA sprites and alpha-blended into frames.

    Fonts are loaded once (``preload``); falls back to cv2 Hershey if no font is found.
    """

    def __init__(self, cache_size: int = 512):
        self.path = None
        self.emoji_font = None
        self._fonts: dict[tuple[int, bool], object] = {}
        self._emoji_cache: dict[tuple[str, int], object] = {}
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        if ImageFont is not None:
            for path, reg, bold in FONT_CANDIDATES:
                if os.path.exists(path):
                    try:
                        ImageFont.truetype(path, 12, index=reg)
                        self.path, self._idx = path, (reg, bold)
                        break
                    except OSError:
                        pass
            if self.path and os.path.exists(EMOJI_FONT):
                try:
                    self.emoji_font = ImageFont.truetype(EMOJI_FONT, 64)
                except OSError:
                    try:
                        self.emoji_font = ImageFont.truetype(EMOJI_FONT, 160)
                    except OSError:
                        self.emoji_font = None

    @property
    def using_pil(self) -> bool:
        return self.path is not None

    def preload(self, sizes):
        if not self.using_pil:
            return
        for size in sizes:
            for bold in (False, True):
                self._font(size, bold)

    def _font(self, size: int, bold: bool):
        key = (size, bold)
        if key not in self._fonts:
            f = ImageFont.truetype(self.path, size, index=self._idx[1 if bold else 0])
            if self.path.endswith("SFNS.ttf"):  # variable font: pick a weight
                try:
                    f.set_variation_by_name(b"Semibold" if bold else b"Regular")
                except Exception:
                    pass
            self._fonts[key] = f
        return self._fonts[key]

    def _emoji(self, cluster: str, height: int):
        key = (cluster, height)
        if key not in self._emoji_cache:
            img = None
            if self.emoji_font is not None:
                try:
                    canvas = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
                    ImageDraw.Draw(canvas).text((10, 10), cluster, font=self.emoji_font,
                                                embedded_color=True)
                    bbox = canvas.getbbox()
                    if bbox:
                        img = canvas.crop(bbox)
                        img = img.resize((max(1, round(img.width * height / img.height)), height),
                                         Image.LANCZOS)
                except Exception:
                    img = None
            self._emoji_cache[key] = img
        return self._emoji_cache[key]

    @staticmethod
    def _runs(text: str):
        runs, buf, emo = [], "", None
        for ch in text:
            e = _is_emoji(ch) or bool(emo and ch in "️‍")
            if emo is None or e == emo:
                buf += ch
            else:
                runs.append((buf, emo))
                buf = ch
            emo = e
        if buf:
            runs.append((buf, emo))
        return runs

    def sprite(self, text: str, size: int, color, bold: bool = False):
        key = (text, size, tuple(color), bold)
        spr = self._cache.get(key)
        if spr is not None:
            self._cache.move_to_end(key)
            return spr
        spr = self._render_pil(text, size, color, bold) if self.using_pil else \
            self._render_hershey(text, size, color, bold)
        self._cache[key] = spr
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return spr

    def _render_pil(self, text, size, color, bold):
        font = self._font(size, bold)
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        emoji_h = int(size * 1.05)
        pieces, width = [], 0
        for run, emo in self._runs(text):
            if emo:
                img = self._emoji(run, emoji_h)
                if img is None:
                    continue  # no emoji font -> just drop the glyph
                pieces.append(("img", img, width))
                width += img.width
            else:
                pieces.append(("txt", run, width))
                width += int(math.ceil(font.getlength(run)))
        width = max(width, 1)
        canvas = Image.new("RGBA", (width + 2, line_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        rgb = (color[2], color[1], color[0], 255)
        for kind, obj, x in pieces:
            if kind == "txt":
                draw.text((x, 0), obj, font=font, fill=rgb)
            else:
                y = max(0, (line_h - obj.height) // 2 + int(size * 0.04))
                canvas.alpha_composite(obj, (x, y))
        arr = np.asarray(canvas)
        return np.ascontiguousarray(arr[..., [2, 1, 0, 3]])  # RGBA -> BGRA

    def _render_hershey(self, text, size, color, bold):
        for a, b in (("→", "->"), ("…", "..."), ("·", "|")):
            text = text.replace(a, b)
        text = "".join(ch for ch in text if ord(ch) < 128).strip() or " "
        fs, th = size / 30, max(1, round(size / (12 if bold else 16)))
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        mask = np.zeros((tht + base + 4, tw + 4), np.uint8)
        cv2.putText(mask, text, (2, tht + 2), cv2.FONT_HERSHEY_SIMPLEX, fs, 255, th, cv2.LINE_AA)
        spr = np.empty(mask.shape + (4,), np.uint8)
        spr[..., :3] = color
        spr[..., 3] = mask
        return spr

    def measure(self, text: str, size: int, bold: bool = False):
        spr = self.sprite(text, size, WHITE, bold)
        return spr.shape[1], spr.shape[0]

    def draw(self, img, text, org, size, color=WHITE, bold=False, alpha=1.0, anchor="lt"):
        """Blend text into img. anchor: horizontal l/m/r + vertical t/m (e.g. 'mm')."""
        spr = self.sprite(text, size, color, bold)
        sh, sw = spr.shape[:2]
        x, y = int(org[0]), int(org[1])
        if anchor[0] == "m":
            x -= sw // 2
        elif anchor[0] == "r":
            x -= sw
        if anchor[1] == "m":
            y -= sh // 2
        blit(img, spr, x, y, alpha)
        return sw, sh


def blit(img, spr, x, y, alpha=1.0):
    """Alpha-blend a BGRA sprite into img (cv2.blendLinear: ~6x faster than numpy floats)."""
    h, w = img.shape[:2]
    sh, sw = spr.shape[:2]
    x, y = int(x), int(y)
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + sw, w), min(y + sh, h)
    if x2 <= x1 or y2 <= y1 or alpha <= 0:
        return
    s = spr[y1 - y:y2 - y, x1 - x:x2 - x]
    roi = img[y1:y2, x1:x2]
    a = s[..., 3].astype(np.float32) * (alpha / 255.0)
    roi[:] = cv2.blendLinear(np.ascontiguousarray(s[..., :3]), roi, a, 1.0 - a)


def fade_blend(frame, rect, draw_fn, alpha):
    """Draw with draw_fn(tmp, ox, oy) on a copy of a small ROI and blend it back with alpha."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (max(int(rect[0]), 0), max(int(rect[1]), 0),
                      min(int(rect[2]), w), min(int(rect[3]), h))
    if x2 <= x1 or y2 <= y1 or alpha <= 0:
        return
    roi = frame[y1:y2, x1:x2]
    tmp = roi.copy()
    draw_fn(tmp, x1, y1)
    cv2.addWeighted(tmp, alpha, roi, 1 - alpha, 0, dst=roi)


# ------------------------------------------------------------------ cached glass panels
class Card:
    """An offscreen glass panel: colour + opacity, built once per content, then composited
    onto every frame with one blend (no full-frame copy, no per-frame text rendering)."""

    def __init__(self, w, h, radius, fill=GLASS, alpha=GLASS_ALPHA, border=PINK, border_px=1):
        self.w, self.h = int(w), int(h)
        self.col = np.zeros((self.h, self.w, 3), np.uint8)
        self.opa = np.zeros((self.h, self.w), np.uint8)
        if alpha > 0:
            self.shape(lambda im, c: rounded_rect(im, (0, 0), (self.w - 1, self.h - 1), radius, c, -1), fill, alpha)
        if border is not None:
            self.shape(lambda im, c: rounded_rect(im, (0, 0), (self.w - 1, self.h - 1), radius, c, border_px), border)

    def shape(self, fn, color, opacity=1.0):
        """Draw with an OpenCV primitive fn(img, color) at the given opacity."""
        fn(self.col, color)
        fn(self.opa, int(255 * opacity))
        return self

    def text(self, T, s, org, size, color=WHITE, bold=False, anchor="lt"):
        spr = T.sprite(s, size, color, bold)
        sh, sw = spr.shape[:2]
        x, y = int(org[0]), int(org[1])
        x -= sw // 2 if anchor[0] == "m" else sw if anchor[0] == "r" else 0
        y -= sh // 2 if anchor[1] == "m" else 0
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + sw, self.w), min(y + sh, self.h)
        if x2 > x1 and y2 > y1:
            sp = spr[y1 - y:y2 - y, x1 - x:x2 - x]
            a = sp[..., 3:4].astype(np.float32) / 255.0
            c = self.col[y1:y2, x1:x2]
            c[:] = (c * (1 - a) + sp[..., :3] * a).astype(np.uint8)
            o = self.opa[y1:y2, x1:x2]
            np.maximum(o, sp[..., 3], out=o)
        return sw, sh

    def finalize(self):
        self._a = self.opa.astype(np.float32) / 255.0  # per-pixel weights for cv2.blendLinear
        self._inv = 1.0 - self._a
        return self

    def blit(self, frame, x, y, alpha=1.0):
        H, W = frame.shape[:2]
        x, y = int(x), int(y)
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + self.w, W), min(y + self.h, H)
        if x2 <= x1 or y2 <= y1 or alpha <= 0:
            return
        sl = (slice(y1 - y, y2 - y), slice(x1 - x, x2 - x))
        roi = frame[y1:y2, x1:x2]
        col = self.col if (x1 - x, y1 - y, x2 - x1, y2 - y1) == (0, 0, self.w, self.h) else \
            np.ascontiguousarray(self.col[sl])
        if alpha >= 0.999:
            a, inv = self._a[sl], self._inv[sl]
        else:
            a = self._a[sl] * alpha
            inv = 1.0 - a
        roi[:] = cv2.blendLinear(col, roi, np.ascontiguousarray(a), np.ascontiguousarray(inv))


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


DEBUG_COLORS = {"ext": GREEN, "curl": (150, 130, 140), "amb": AMBER}


# ------------------------------------------------------------------ UI
class UI:
    """All overlay drawing. Sizes derive from scale = frame_width / 1280. Every panel is a
    cached Card keyed by its content, so text is only rendered when something changes."""

    LEGEND_ALPHA = 0.42  # more see-through than the HUD

    def __init__(self, frame_w: int, frame_h: int, text: TextRenderer, min_ratio: float):
        self.w, self.h = frame_w, frame_h
        self.s = s = frame_w / 1280
        self.text = text
        self.min_ratio = min_ratio
        px = lambda v: max(1, round(v * s))  # noqa: E731
        self.px = px
        self.sz = {k: max(10, round(v * s)) for k, v in dict(
            title=24, body=15, small=12, pill=14, legend=14, toast=30, hint=22, tag=16, debug=13).items()}
        text.preload(self.sz.values())
        self.margin, self.pad, self.radius, self.border = px(18), px(14), px(16), px(1.5)
        self._grad_cache: dict[tuple[int, int], np.ndarray] = {}
        self._cards: OrderedDict = OrderedDict()

    # -- cache
    def card(self, key, build) -> Card:
        c = self._cards.get(key)
        if c is None:
            c = build().finalize()
            self._cards[key] = c
            if len(self._cards) > 96:
                self._cards.popitem(last=False)
        else:
            self._cards.move_to_end(key)
        return c

    def _gradient(self, w, h, vertical=False):
        key = (w, h, vertical)
        if key not in self._grad_cache:
            n = h if vertical else w
            t = np.linspace(1, 0, n, dtype=np.float32) if vertical else np.linspace(0, 1, n, dtype=np.float32)
            g = np.array(PINK, np.float32) * (1 - t[:, None]) + np.array(PURPLE, np.float32) * t[:, None]
            g = np.repeat(g[:, None, :], w, axis=1) if vertical else np.repeat(g[None, :, :], h, axis=0)
            self._grad_cache[key] = g.astype(np.uint8)
        return self._grad_cache[key]

    def bar(self, frame, x, y, w, h, value, tick=True):
        rounded_rect(frame, (x, y), (x + w, y + h), h // 2, TRACK, -1)
        fw = int(w * max(0.0, min(1.0, value)))
        if fw >= h:
            roi = frame[y:y + h + 1, x:x + w + 1]
            mask = np.zeros(roi.shape[:2], np.uint8)
            rounded_rect(mask, (0, 0), (fw, h), h // 2, 255, -1)
            a = mask[..., None].astype(np.float32) / 255.0
            roi[:] = (roi * (1 - a) + self._gradient(roi.shape[1], roi.shape[0]) * a).astype(np.uint8)
        if tick:
            tx = x + int(w * self.min_ratio)  # where the smoother accepts a gesture
            cv2.line(frame, (tx, y - self.px(3)), (tx, y + h + self.px(3)), WHITE, self.px(1.5), cv2.LINE_AA)

    # -- main entry
    def render(self, frame, *, raw, stable, conf, fps, dry_run, show_help, hand_present,
               toast, now, bindings, mode, extra_rows=(), volume=None, pinch_pts=None,
               mouse_ui=None, exit_progress=0.0, classifier="RULES", hand_tags=(),
               debug=None, prompt=None, hand_views=(), effects=None, recent_actions=(), hud_override=None):
        self.bindings = bindings
        self.recent_actions = list(recent_actions)[:3]
        self.extra_rows = list(extra_rows)
        if mouse_ui is not None:
            self._mouse_box(frame, mouse_ui["box"])  # under the panels
        if debug:
            self._debug_dots(frame, debug.get("hands", ()))

        occupied = [self._hud(frame, raw, stable, conf, hand_present, hud_override)]
        occupied += self._pills(frame, fps, dry_run, mode, classifier)
        if show_help:
            occupied.append(self._legend(frame, stable))
        if volume is not None:
            occupied.append(self._volume(frame, volume, pinch_pts))
        if debug:
            occupied.append(self._debug(frame, debug["lines"]))
        if prompt:
            occupied.append(self._prompt(frame, prompt))
        if not hand_present and not prompt:
            self._hint(frame, now)
        toast_rect = self._toast(frame, toast, now) if toast else None
        if toast_rect:
            occupied.append(toast_rect)
        self._tags(frame, hand_tags, occupied)
        if hand_views or effects:
            # hand-anchored overlays never cover a panel: back up the panel areas and
            # put them back afterwards (cheap: the panels are small)
            panels = [r for r in occupied if r[2] > r[0] and r[3] > r[1]]
            reach = self._hand_reach(hand_views, effects or {})  # where overlays can land
            saved = [(r, frame[max(r[1], 0):r[3], max(r[0], 0):r[2]].copy())
                     for r in panels if any(D.overlaps(r, z) for z in reach)]
            self._hand_layer(frame, hand_views, effects or {}, panels, now)
            for r, patch in saved:
                frame[max(r[1], 0):r[3], max(r[0], 0):r[2]] = patch

        if mouse_ui is not None:
            self._mouse(frame, mouse_ui)
        if exit_progress > 0:
            self._exit_ring(frame, exit_progress)
        if toast:
            self._pulse(frame, toast, now)

    # -- panels (each returns its rect)
    def _hud(self, frame, raw, stable, conf, hand_present, override=None):
        """Compact top-left card: gesture, its action, confidence."""
        T, sz, px, pad = self.text, self.sz, self.px, self.pad
        g = stable if hand_present else "no hand"
        title = f"{ICONS.get(g, '')} {pretty(g)}"
        if override:  # e.g. pinch mode: ("🤏 Pinch", "→  Volume 62%")
            title, action, acolor = override[0], override[1], WHITE
        elif g in self.bindings:
            action, acolor = f"→  {self.bindings[g].label}", WHITE
        elif hand_present:
            action, acolor = "Hold a gesture steady…", MUTED
        else:
            action, acolor = "Waiting for a hand", MUTED
        pct = f"{conf * 100:.0f}%" if hand_present else ""
        chips = self.recent_actions  # newest first (raw label lives in the 'g' debug card)
        tw, th = T.measure(title, sz["title"], True)
        aw, ah = T.measure(action, sz["body"])
        lw, sh = T.measure("CONFIDENCE", sz["small"], True)
        chip_dims = [T.measure(c, sz["small"], True) for c in chips]
        chip_h = (max(h for _, h in chip_dims) + px(6)) if chips else 0
        chips_w = sum(w + px(14) for w, _ in chip_dims) + px(6) * max(0, len(chips) - 1)
        bar_h = px(6)
        width = max(px(240), tw + 2 * pad, aw + 2 * pad, lw + px(50) + 2 * pad, chips_w + 2 * pad)
        width = min(width, int(self.w * 0.42))
        y_action = pad + th
        y_label = y_action + ah + px(8)
        y_bar = y_label + sh + px(5)
        y_chips = y_bar + bar_h + px(8)
        height = (y_chips + chip_h + pad) if chips else (y_bar + bar_h + pad)
        x1 = y1 = self.margin

        def build():
            c = Card(width, height, self.radius)
            c.text(T, title, (pad, pad - px(2)), sz["title"], WHITE, bold=True)
            c.text(T, action, (pad, y_action), sz["body"], acolor)
            c.text(T, "CONFIDENCE", (pad, y_label), sz["small"], MUTED, bold=True)
            if pct:
                c.text(T, pct, (width - pad, y_label), sz["small"], MUTED, anchor="rt")
            x = pad
            for i, (chip, (cw, _)) in enumerate(zip(chips, chip_dims)):
                w = cw + px(14)
                if x + w > width - pad:
                    break
                col = WHITE if i == 0 else MUTED
                c.shape(lambda im, cc, x=x, w=w: rounded_rect(im, (x, y_chips), (x + w, y_chips + chip_h), chip_h // 2,
                                                              cc, -1), (70, 40, 70), 0.85)
                c.shape(lambda im, cc, x=x, w=w: rounded_rect(im, (x, y_chips), (x + w, y_chips + chip_h), chip_h // 2,
                                                              cc, self.border), PINK if i == 0 else TRACK)
                c.text(T, chip, (x + px(7), y_chips + chip_h // 2), sz["small"], col, bold=i == 0, anchor="lm")
                x += w + px(6)
            return c
        self.card(("hud", title, action, acolor, pct, tuple(chips), width), build).blit(frame, x1, y1)
        self.bar(frame, x1 + pad, y1 + y_bar, width - 2 * pad, bar_h, conf if hand_present else 0)
        return (x1, y1, x1 + width, y1 + height)

    def _pills(self, frame, fps, dry_run, mode, classifier="RULES"):
        T, px = self.text, self.px
        run, rcol = ("DRY-RUN", AMBER) if dry_run else ("LIVE", GREEN)
        icon, mcol = MODE_STYLE.get(mode, ("", PINK))
        cicon, ccol = CLASSIFIER_STYLE.get(classifier, ("", MUTED))
        items = [(f"{fps:.0f} FPS", PINK, False), (f"{cicon} {classifier}", ccol, False),
                 (f"{icon} {mode.upper()}", mcol, False), (run, rcol, True)]
        dims = []
        for label, _, dot in items:
            tw, th = T.measure(label, self.sz["pill"], True)
            dims.append((tw + px(22) + (px(14) if dot else 0), th + px(10)))
        gap = px(8)
        total = sum(w for w, _ in dims) + gap * (len(items) - 1)
        ph = max(h for _, h in dims)

        def build():
            c = Card(total, ph, 0, alpha=0, border=None)
            x = 0
            for (label, col, dot), (w, _) in zip(items, dims):
                r = ph // 2
                c.shape(lambda im, cc, x=x, w=w: rounded_rect(im, (x, 0), (x + w - 1, ph - 1), r, cc, -1), GLASS, GLASS_ALPHA)
                c.shape(lambda im, cc, x=x, w=w: rounded_rect(im, (x, 0), (x + w - 1, ph - 1), r, cc, self.border), col)
                tx = x + px(11)
                if dot:
                    c.shape(lambda im, cc, tx=tx: cv2.circle(im, (tx + px(4), ph // 2), px(4), cc, -1, cv2.LINE_AA), col)
                    tx += px(14)
                c.text(T, label, (tx, ph // 2), self.sz["pill"], col, bold=True, anchor="lm")
                x += w + gap
            return c
        x1, y1 = self.w - self.margin - total, self.margin
        self.card(("pills", tuple(i[0] for i in items), ph), build).blit(frame, x1, y1)
        return [(x1, y1, x1 + total, y1 + ph)]

    def _legend(self, frame, stable):
        """Bottom-left, narrow and see-through; hidden until 'h'."""
        T, sz, px = self.text, self.sz, self.px
        pad = px(12)
        rows = [(g, LEGEND_NAMES.get(g, f"{ICONS.get(g, '•')} {pretty(g)}"), b.label)
                for g, b in self.bindings.items()]
        active = {k for k, _, _, on in self.extra_rows if on}
        rows += [(k, name, label) for k, name, label, _ in self.extra_rows]
        if not rows:
            return (0, 0, 0, 0)
        active |= {g for g, _, _ in rows if g == stable}
        _, rh = T.measure("Ag✋", sz["legend"])
        row_h = rh + px(6)
        col1 = max(T.measure(r[1], sz["legend"], True)[0] for r in rows)
        col2 = max(T.measure(r[2], sz["legend"])[0] for r in rows)
        _, hh = T.measure("GESTURES", sz["small"], True)
        width = min(col1 + col2 + px(20) + 2 * pad, int(self.w * 0.34))
        height = pad + hh + px(6) + row_h * len(rows) + pad
        x1, y2 = self.margin, self.h - self.margin
        y1 = y2 - height
        y_rows = pad + hh + px(6)

        def build():
            c = Card(width, height, self.radius, alpha=self.LEGEND_ALPHA)
            c.text(T, "GESTURES", (pad, pad), sz["small"], MUTED, bold=True)
            for i, (g, name, label) in enumerate(rows):
                on = g in active
                ry = y_rows + i * row_h
                if on:
                    c.shape(lambda im, cc, ry=ry: rounded_rect(im, (px(6), ry), (width - px(6), ry + row_h), px(8), cc, -1),
                            HIGHLIGHT, 0.7)
                    c.shape(lambda im, cc, ry=ry: rounded_rect(im, (px(6), ry), (width - px(6), ry + row_h), px(8), cc,
                                                               self.border), PINK)
                c.text(T, name, (pad, ry + row_h // 2), sz["legend"], WHITE, bold=on, anchor="lm")
                c.text(T, label, (width - pad, ry + row_h // 2), sz["legend"], WHITE if on else MUTED,
                       bold=on, anchor="rm")
            return c
        key = ("legend", tuple(rows), frozenset(active), width)
        self.card(key, build).blit(frame, x1, y1)
        return (x1, y1, x1 + width, y2)

    def _hint(self, frame, now):
        T, px = self.text, self.px
        msg = "Show your hand ✋"
        tw, th = T.measure(msg, self.sz["hint"], True)
        pw, ph = tw + px(48), th + px(22)

        def build():
            c = Card(pw, ph, ph // 2)
            c.text(T, msg, (pw // 2, ph // 2), self.sz["hint"], WHITE, bold=True, anchor="mm")
            return c
        breathe = 0.7 + 0.3 * (0.5 + 0.5 * math.sin(now * 3.0))
        self.card(("hint", pw), build).blit(frame, self.w // 2 - pw // 2, self.h // 2 - ph // 2, breathe)

    def _volume(self, frame, level, pinch_pts):
        """Vertical volume bar on the right edge + a line between the pinching tips."""
        T, px, pad = self.text, self.px, self.pad
        bw, bh = px(14), int(self.h * 0.34)
        lw, lh = T.measure("100%", self.sz["body"], True)
        _, sh = T.measure("VOL", self.sz["small"], True)
        pw = max(px(58), lw + 2 * px(10), T.measure("🔊 VOL", self.sz["small"], True)[0] + 2 * px(10))
        ph = pad + lh + px(6) + bh + px(6) + sh + pad
        x1, y1 = self.w - self.margin - pw, self.h // 2 - ph // 2
        by1 = pad + lh + px(6)

        def build():
            c = Card(pw, ph, self.radius)
            c.text(T, f"{level}%", (pw // 2, pad), self.sz["body"], WHITE, bold=True, anchor="mt")
            c.text(T, "🔊 VOL", (pw // 2, by1 + bh + px(6)), self.sz["small"], MUTED, bold=True, anchor="mt")
            c.shape(lambda im, cc: rounded_rect(im, (pw // 2 - bw // 2, by1), (pw // 2 + bw // 2, by1 + bh), bw // 2, cc, -1),
                    TRACK)
            return c
        self.card(("vol", level, pw), build).blit(frame, x1, y1)
        fill_h = int(bh * level / 100)
        cx, gy2 = x1 + pw // 2, y1 + by1 + bh
        if fill_h > bw // 2:
            roi = frame[gy2 - fill_h:gy2 + 1, cx - bw // 2:cx + bw // 2 + 1]
            mask = np.zeros(roi.shape[:2], np.uint8)
            rounded_rect(mask, (0, 0), (roi.shape[1] - 1, fill_h), bw // 2, 255, -1)
            a = mask[..., None].astype(np.float32) / 255.0
            grad = self._gradient(roi.shape[1], roi.shape[0], vertical=True)
            roi[:] = (roi * (1 - a) + grad * a).astype(np.uint8)
        if pinch_pts:
            (ax, ay), (bx, by) = pinch_pts
            cv2.line(frame, (ax, ay), (bx, by), WHITE, px(3), cv2.LINE_AA)
            mx, my = (ax + bx) // 2, (ay + by) // 2
            cv2.circle(frame, (mx, my), px(10), PINK, -1, cv2.LINE_AA)
            T.draw(frame, f"{level}%", (mx + px(16), my), self.sz["pill"], WHITE, bold=True, anchor="lm")
        return (x1, y1, x1 + pw, y1 + ph)

    def _debug(self, frame, lines):
        """'g' overlay: finger states, distances and the rule that rejected the pose."""
        T, px, pad = self.text, self.px, self.pad
        sizes = [T.measure(ln, self.sz["debug"], i == 0) for i, ln in enumerate(lines)]
        width = min(max(w for w, _ in sizes) + 2 * pad, self.w - 2 * self.margin)
        height = sum(h + px(3) for _, h in sizes) + 2 * pad - px(3)
        x1, y2 = self.w // 2 - width // 2, self.h - self.margin
        y1 = y2 - height

        def build():
            c = Card(width, height, self.radius, alpha=0.72, border=AMBER)
            y = pad
            for i, (ln, (_, h)) in enumerate(zip(lines, sizes)):
                c.text(T, ln, (pad, y), self.sz["debug"], WHITE if i == 0 else MUTED, bold=i == 0)
                y += h + px(3)
            return c
        self.card(("debug", tuple(lines), width), build).blit(frame, x1, y1)
        return (x1, y1, x1 + width, y2)

    def _debug_dots(self, frame, hands):
        """Ring each fingertip: green = extended, grey = curled, amber = ambiguous."""
        tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
        for pts, tri in hands:
            for f, i in tips.items():
                cv2.circle(frame, pts[i], self.px(15), DEBUG_COLORS[tri[f]], self.px(3), cv2.LINE_AA)

    def _prompt(self, frame, prompt):
        T, px, pad = self.text, self.px, self.pad
        lines = prompt
        sizes = [T.measure(ln, self.sz["body"], i == 0) for i, ln in enumerate(lines)]
        width = max(w for w, _ in sizes) + 2 * pad
        height = sum(h + px(4) for _, h in sizes) + 2 * pad
        x1, y1 = self.w // 2 - width // 2, self.h // 2 - height // 2

        def build():
            c = Card(width, height, self.radius, alpha=0.8, border=AMBER)
            y = pad
            for i, (ln, (_, h)) in enumerate(zip(lines, sizes)):
                c.text(T, ln, (width // 2, y), self.sz["body"], WHITE if i == 0 else MUTED, bold=i == 0, anchor="mt")
                y += h + px(4)
            return c
        self.card(("prompt", tuple(lines)), build).blit(frame, x1, y1)
        return (x1, y1, x1 + width, y1 + height)

    def _tags(self, frame, tags, occupied):
        """Gesture tag per hand, placed where it doesn't cover a panel (or skipped)."""
        T, px = self.text, self.px
        size = self.sz["tag"]
        for box, label in tags:
            if not label:
                continue
            tw, th = T.measure(label, size, True)
            tag_w, tag_h = tw + px(24), th + px(10)
            bx1, by1, bx2, by2 = box
            gap = px(6)
            candidates = [(bx1, by1 - tag_h - gap), (bx2 - tag_w, by1 - tag_h - gap),
                          (bx1, by2 + gap), (bx2 - tag_w, by2 + gap), (bx1 + gap, by1 + gap)]
            for tx, ty in candidates:
                rect = (tx, ty, tx + tag_w, ty + tag_h)
                if tx >= 0 and ty >= 0 and rect[2] <= self.w and rect[3] <= self.h and \
                        not any(_overlap(rect, o) for o in occupied):
                    def build():
                        c = Card(tag_w, tag_h, tag_h // 2, fill=PINK, alpha=1.0, border=None)
                        c.text(T, label, (px(12), tag_h // 2), size, WHITE, bold=True, anchor="lm")
                        return c
                    self.card(("tag", label), build).blit(frame, tx, ty)
                    occupied.append(rect)
                    break

    # -- hand-anchored overlays (box, tag + hold ring, emoji pop, action pill, volume meter)
    def _dashed_box(self, frame, box, color, th):
        x1, y1, x2, y2 = box
        dash = self.px(12)
        segs = []
        for x in range(x1, x2, dash * 2):
            segs += [((x, y1), (min(x + dash, x2), y1)), ((x, y2), (min(x + dash, x2), y2))]
        for y in range(y1, y2, dash * 2):
            segs += [((x1, y), (x1, min(y + dash, y2))), ((x2, y), (x2, min(y + dash, y2)))]
        cv2.polylines(frame, [np.array(sg, np.int32) for sg in segs], False, color, th, cv2.LINE_AA)

    def _hand_reach(self, views, fx):
        """Rects that hand overlays (box, tag, pop, pill, meter) could touch this frame."""
        m = self.px(170)  # tag above/below + emoji pop + floating pill + meter
        boxes = [v["box"] for v in views] + [e["box"] for k in ("pops", "floats") for e in fx.get(k, ())]
        if fx.get("meter"):
            boxes.append(fx["meter"]["box"])
        return [(b[0] - m, b[1] - m, b[2] + m, b[3] + m) for b in boxes]

    def _tag_card(self, text, color, unknown, emoji_slot):
        """Tag pill (cached per text/colour). Leaves a round slot for the emoji + ring."""
        T, px = self.text, self.px
        size = self.sz["tag"]
        tw, th = T.measure(text, size, True)
        tag_h = th + px(12)
        slot = tag_h if emoji_slot else 0
        tag_w = slot + tw + px(22)

        def build():
            fill, alpha = ((60, 50, 60), 0.75) if unknown else (color, 0.95)
            c = Card(tag_w, tag_h, tag_h // 2, fill=fill, alpha=alpha,
                     border=(110, 100, 110) if unknown else None, border_px=self.border)
            if slot:  # dark disc behind the emoji so the hold ring reads on any accent colour
                c.shape(lambda im, cc: cv2.circle(im, (tag_h // 2, tag_h // 2), tag_h // 2 - px(2), cc, -1, cv2.LINE_AA),
                        (45, 28, 45), 0.9)
            c.text(T, text, (slot + px(11), tag_h // 2), size, MUTED if unknown else WHITE, bold=True, anchor="lm")
            return c
        return self.card(("tag2", text, color, unknown, emoji_slot), build)

    def _emoji_sprite(self, emoji, size, scale_q=None):
        """Emoji sprite at a size; scaled copies are cached per 5% step (no per-frame resize)."""
        key = ("emoji", emoji, size, scale_q)
        spr = self._cards.get(key)
        if spr is None:
            base = self.text.sprite(emoji, size, WHITE)
            if scale_q is not None and scale_q != 20:
                k = scale_q / 20
                base = cv2.resize(base, (max(1, int(base.shape[1] * k)), max(1, int(base.shape[0] * k))),
                                  interpolation=cv2.INTER_LINEAR)
            spr = base
            self._cards[key] = spr
            if len(self._cards) > 160:
                self._cards.popitem(last=False)
        return spr

    def _hand_layer(self, frame, views, fx, avoid, now):
        T, px = self.text, self.px
        th2 = max(1, round(2 * self.s))
        for v in views:
            box, col = v["box"], v["color"]
            unknown = v["label"] in ("unknown", "no hand")
            # ---- box: solid accent, or dashed grey when unknown
            if unknown:
                self._dashed_box(frame, box, col, th2)
            else:
                rounded_rect(frame, box[:2], box[2:], int(18 * self.s), col, th2)
            # ---- tag: "…" when unknown, else emoji slot + name; follows the box smoothly
            name = "…" if unknown else pretty(v["label"]) + v.get("suffix", "")
            card = self._tag_card(name, col, unknown, emoji_slot=not unknown)
            target = D.place_tag(box, card.w, card.h, self.w, self.h, avoid, gap=px(6))
            if target is None:
                continue
            tx, ty = v["tag_smoother"].update(target)
            tx, ty, _, _ = D.clamp_rect((tx, ty, tx + card.w, ty + card.h), self.w, self.h)
            card.blit(frame, tx, ty)
            if unknown:
                continue
            # emoji inside the round slot, with the hold-to-fire ring around it
            r = card.h // 2
            cx, cy = tx + r, ty + r
            spr = self._emoji_sprite(ICONS.get(v["label"], "✨"), max(10, int(card.h * 0.6)))
            blit(frame, spr, cx - spr.shape[1] // 2, cy - spr.shape[0] // 2)
            prog = v.get("progress")
            if prog is not None:
                flash = D.flash_curve(now - v.get("flash_t", -1e9))
                k, white = flash if flash else (1.0, 0.0)
                rr = int((r - px(1)) * k)
                ring_w = max(2, round(3 * self.s * (1 + white)))
                cv2.circle(frame, (cx, cy), rr, D.lerp_color(col, (40, 30, 40), 0.55), ring_w, cv2.LINE_AA)
                if prog > 0:
                    ring_col = D.lerp_color(WHITE, col, 0.0 if v.get("confirmed") else 0.35)
                    cv2.ellipse(frame, (cx, cy), (rr, rr), -90, 0, 360 * prog, ring_col, ring_w, cv2.LINE_AA)
                if flash:  # bright pop ring when the action fires
                    cv2.circle(frame, (cx, cy), int(rr * (1 + 0.5 * (1 - white))), D.lerp_color(col, WHITE, white),
                               max(1, round(2 * self.s)), cv2.LINE_AA)

        # ---- big emoji pop when a new gesture is confirmed
        for p in fx.get("pops", ()):
            curve = D.pop_curve(now - p["t0"])
            if not curve:
                continue
            scale, alpha = curve
            spr = self._emoji_sprite(p["emoji"], max(20, round(120 * self.s)), round(scale * 20))
            bx1, by1, bx2, by2 = p["box"]  # above the box's top-right corner (meter sits at the side)
            x = min(max(bx2 - spr.shape[1] // 2, 0), self.w - spr.shape[1])
            y = max(by1 - spr.shape[0] - px(4), 0)
            blit(frame, spr, x, y, alpha)

        # ---- action pill floating up from the hand
        for fl in fx.get("floats", ()):
            curve = D.float_curve(now - fl["t0"])
            if not curve:
                continue
            rise, alpha = curve
            label = fl["text"]

            def build(label=label, color=fl["color"]):
                tw, th = T.measure(label, self.sz["pill"], True)
                c = Card(tw + px(22), th + px(10), (th + px(10)) // 2, alpha=0.85, border=color, border_px=th2)
                c.text(T, label, (c.w // 2, c.h // 2), self.sz["pill"], WHITE, bold=True, anchor="mm")
                return c
            card = self.card(("float", label, fl["color"]), build)
            bx1, by1, bx2, _ = fl["box"]
            x = (bx1 + bx2) // 2 - card.w // 2
            y = by1 - card.h - px(10) - int(px(70) * rise)
            x, y, _, _ = D.clamp_rect((x, y, x + card.w, y + card.h), self.w, self.h)
            card.blit(frame, x, y, alpha)

        # ---- volume meter next to the hand while thumbs up/down repeats
        m = fx.get("meter")
        if m:
            bx1, by1, bx2, by2 = m["box"]
            mw, mh = px(12), max(px(60), min(px(120), by2 - by1))
            x = bx2 + px(10)
            if x + mw + px(40) > self.w:
                x = bx1 - px(10) - mw
            y = (by1 + by2) // 2 - mh // 2
            segs = 16
            lit = int(round(m["level"] / 100 * segs))
            bump = now - m.get("last_tick", -1e9) < 0.2
            gap = max(1, mh // (segs * 5))
            seg_h = (mh - gap * (segs - 1)) // segs
            for i in range(segs):
                sy2 = y + mh - i * (seg_h + gap)
                on = i < lit
                c = D.lerp_color(m["color"], WHITE, 0.5) if (on and bump and i == lit - 1) else \
                    (m["color"] if on else (70, 55, 70))
                cv2.rectangle(frame, (x, sy2 - seg_h), (x + mw + (px(3) if bump and i == lit - 1 else 0), sy2),
                              c, -1)
            arrow = "▲" if m["dir"] > 0 else "▼"
            T.draw(frame, f"{arrow} {round(m['level'])}", (x + mw // 2, y - px(4) - T.measure("0", self.sz["small"], True)[1]),
                   self.sz["small"], WHITE, bold=True, anchor="mt")

    def _mouse_box(self, frame, b):
        """Dashed active-area box: this part of the camera frame maps to the whole screen."""
        px = self.px
        x1, y1 = int(self.w * (1 - b) / 2), int(self.h * (1 - b) / 2)
        x2, y2 = self.w - x1, self.h - y1
        dash = px(14)
        for x in range(x1, x2, dash * 2):  # dashed rectangle
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), PURPLE, px(2), cv2.LINE_AA)
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), PURPLE, px(2), cv2.LINE_AA)
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), PURPLE, px(2), cv2.LINE_AA)
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), PURPLE, px(2), cv2.LINE_AA)
        self.text.draw(frame, "screen area", (x2 - px(8), y2 - px(6) - self.text.measure(
            "screen area", self.sz["small"], True)[1]), self.sz["small"], PURPLE, bold=True, anchor="rt")

    def _mouse(self, frame, m):
        """Fingertip cursor marker (green + filled while the button is held)."""
        px = self.px
        tip = m.get("tip")
        if tip:
            state = "SCROLL" if m["scrolling"] else "DRAG" if m["pressed"] else ""
            col = GREEN if m["pressed"] else PINK
            cv2.circle(frame, tip, px(16), col, px(3), cv2.LINE_AA)
            if m["pressed"]:
                cv2.circle(frame, tip, px(8), col, -1, cv2.LINE_AA)
            cv2.line(frame, (tip[0] - px(24), tip[1]), (tip[0] - px(18), tip[1]), col, px(2), cv2.LINE_AA)
            cv2.line(frame, (tip[0] + px(18), tip[1]), (tip[0] + px(24), tip[1]), col, px(2), cv2.LINE_AA)
            if state:
                self.text.draw(frame, state, (tip[0] + px(28), tip[1]), self.sz["pill"], col,
                               bold=True, anchor="lm")

    def _exit_ring(self, frame, progress):
        """Countdown ring shown while both palms are open."""
        T, px = self.text, self.px
        cx, cy, r = self.w // 2, self.h // 2, px(70)

        def disc(img, ox, oy):
            cv2.circle(img, (cx - ox, cy - oy), r + px(18), GLASS, -1, cv2.LINE_AA)
        fade_blend(frame, (cx - r - px(20), cy - r - px(20), cx + r + px(21), cy + r + px(21)), disc, 0.7)
        cv2.circle(frame, (cx, cy), r, TRACK, px(8), cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, 360 * progress,
                    lerp_color(PINK, PURPLE, progress), px(8), cv2.LINE_AA)
        T.draw(frame, "✋✋", (cx, cy - px(14)), self.sz["body"], WHITE, anchor="mm")
        T.draw(frame, "EXIT", (cx, cy + px(16)), self.sz["pill"], WHITE, bold=True, anchor="mm")

    def _toast(self, frame, toast, now):
        T, sz, px, pad = self.text, self.sz, self.px, self.pad
        age = now - toast["t0"]
        if age > TOAST_SECONDS:
            return None
        # hold briefly, then fade out; small slide-in at the start
        fade = 1.0 if age < 0.25 else max(0.0, 1 - (age - 0.25) / (TOAST_SECONDS - 0.25))
        rise = int(px(14) * min(1.0, age / 0.15))
        msg = f"{ICONS.get(toast['gesture'], '')}  {toast['label']}"
        tw, th = T.measure(msg, sz["toast"], True)
        _, ch = T.measure("ACTION", sz["small"], True)
        pw, ph = tw + 2 * px(28), ch + th + px(6) + 2 * pad
        cap = toast.get("caption", "ACTION")

        def build():
            c = Card(pw, ph, self.radius, alpha=0.78, border_px=px(2))
            c.text(T, cap, (pw // 2, pad), sz["small"], PINK, bold=True, anchor="mt")
            c.text(T, msg, (pw // 2, pad + ch + px(6)), sz["toast"], WHITE, bold=True, anchor="mt")
            return c
        x1, y1 = self.w // 2 - pw // 2, self.margin + rise  # top-centre, away from the hand
        self.card(("toast", msg, cap), build).blit(frame, x1, y1, fade)
        return (x1, y1, x1 + pw, y1 + ph)

    def _pulse(self, frame, toast, now):
        age = now - toast["t0"]
        if age > PULSE_SECONDS or toast.get("center") is None:
            return
        t = age / PULSE_SECONDS
        ease = 1 - (1 - t) ** 3
        cx, cy = toast["center"]
        rad = int(self.px(30) + self.px(110) * ease)
        th = max(1, int(self.px(8) * (1 - t)) + 1)
        col = lerp_color(PINK, PURPLE, t)

        def ring(img, ox, oy):
            cv2.circle(img, (cx - ox, cy - oy), rad, col, th, cv2.LINE_AA)
        m = rad + th + 2
        fade_blend(frame, (cx - m, cy - m, cx + m, cy + m), ring, 1 - t)


# ------------------------------------------------------------------ main loop
class HandFx:
    """Per-hand display state: smoothed box + tag position, colour fade, ring flash."""

    def __init__(self):
        self.box = D.RectSmoother(alpha=0.4)
        self.tag = D.PointSmoother(alpha=0.35)
        self.fade = D.ColorFade(duration=0.15)
        self.flash_t = float("-inf")

    def reset(self):
        self.box.reset()
        self.tag.reset()


VOLUME_KEYS = {"media_volume_up": +1, "media_volume_down": -1}


def run(args):
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera/video. Try --camera 1 or check permissions.")

    cfg = load_config(args.config)
    tracker = HandTracker()
    smoother = ConfidenceSmoother(window=int(setting(cfg, "smoothing", "window", 7)),
                                  min_ratio=float(setting(cfg, "smoothing", "min_ratio", 0.6)))
    ok_ratio = float(setting(cfg, "pinch", "ok_ratio", 0.25))
    ml = None
    if args.classifier == "ml":
        from gesture.ml import DEFAULT_MODEL, MLGestureClassifier
        ml = MLGestureClassifier.load(args.model or setting(cfg, "ml", "model", DEFAULT_MODEL),
                                      min_confidence=float(setting(cfg, "ml", "min_confidence", 0.6)))
        print(f"[ml] using {ml.name} ({len(ml.labels)} labels), min confidence {ml.min_confidence}")
    pose_trackers: dict[str, PoseTracker] = {}  # per hand: finger-state hysteresis
    dispatcher = ActionDispatcher(dry_run=args.dry_run, config=cfg, mode=args.mode)

    sw_cfg, vol_cfg, m_cfg = (cfg.get(k) or {} for k in ("swipe", "volume", "mouse"))
    swipe = motion.SwipeDetector(
        window=sw_cfg.get("window", 0.4), min_travel=sw_cfg.get("min_travel", 0.25),
        max_vertical_ratio=sw_cfg.get("max_vertical_ratio", 0.5),
        cooldown=sw_cfg.get("cooldown", 1.0), still_hold=sw_cfg.get("still_hold", 0.3),
        still_eps=sw_cfg.get("still_eps", 0.04))
    slider = motion.VolumeSlider(min_ratio=vol_cfg.get("min_ratio", 0.25),
                                 max_ratio=vol_cfg.get("max_ratio", 1.1),
                                 step=vol_cfg.get("step", 3), alpha=vol_cfg.get("alpha", 0.4))
    volume_modes = set(vol_cfg.get("modes", ["media"]))
    pinch_ctl = motion.PinchController(slider)
    hold = HoldThrough(grace=0.3)  # thumbs up/down survive 0.3 s of unknown frames (volume repeat)
    screen_w, screen_h = motion.screen_size()
    mouse_box = float(m_cfg.get("box", 0.6))
    mouse_logic = motion.MouseLogic(
        screen_w, screen_h, alpha=m_cfg.get("alpha", 0.35), box=mouse_box,
        click_ratio=setting(cfg, "pinch", "click_ratio", 0.30),
        release_factor=setting(cfg, "pinch", "release_factor", 1.4),
        debounce=m_cfg.get("click_debounce", 0.3), click_freeze=m_cfg.get("click_freeze", 0.15),
        scroll_gain=m_cfg.get("scroll_gain", 25))
    vmouse = motion.VirtualMouse()
    exit_timer = motion.HoldTimer(setting(cfg, "exit", "hold", 1.0))
    text = TextRenderer()  # fonts are loaded here / in UI(), never inside the loop
    if not text.using_pil:
        print("[ui] Pillow or system font not found - using OpenCV Hershey text.")
    ui = None
    writer = None
    toast = None
    show_help = False   # legend collapsed by default: 'h' shows it
    show_debug = False  # 'g'
    pending = None      # 'p': landmarks waiting for the expected label
    prev = time.monotonic()
    fps = 0.0
    hand_fx = [HandFx(), HandFx()]   # live gesture display, per hand slot
    pops, floats = [], []             # big-emoji pops and floating action pills
    recent_actions = deque(maxlen=3)  # chips under the HUD's confidence bar
    meter, meter_dir = D.VolumeMeter(), +1
    prev_stable = "unknown"

    print("Running... press q to quit (h legend, g debug, p save a test fixture).")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.video and not args.no_mirror:
                frame = cv2.flip(frame, 1)  # mirror view feels natural on a webcam
            fh, fw = frame.shape[:2]
            if ui is None or (ui.w, ui.h) != (fw, fh):
                ui = UI(fw, fh, text, smoother.min_ratio)

            hands = tracker.process(frame)[:2]
            now = time.monotonic()

            # ---- static poses: one hysteresis tracker per hand (keyed by handedness)
            infos, labels, seen = [], [], []
            for i, hand in enumerate(hands):
                key = hand.handedness if hand.handedness not in seen else f"{hand.handedness}{i}"
                seen.append(key)
                info = pose_trackers.setdefault(key, PoseTracker(ok_ratio)).update(hand.iso)
                infos.append(info)
                labels.append(ml.predict(hand.iso, hand.handedness) if ml else info.label)
            for key in list(pose_trackers):
                if key not in seen:
                    del pose_trackers[key]

            primary = hands[0] if hands else None  # first detected hand drives the controls
            raw, stable = "no hand", "unknown"
            mouse_events: list[tuple] = []
            pinch = motion.PinchState()
            if primary:
                lm, iso = primary.landmarks, primary.iso
                raw = labels[0]  # straight from the classifier: swipe/pinch never change it
                pinch = pinch_ctl.update(now, iso, infos[0].states, enabled=dispatcher.mode in volume_modes)
                if pinch.ended:  # the L-shape voted "pointing" all along: don't let it fire on exit
                    smoother.reset()
                    hold.reset()
                stable = hold.update(now, smoother.update(raw))
            else:
                smoother.reset()
                hold.reset()
                swipe.reset()
                pinch_ctl.reset()
                mouse_events += mouse_logic.reset()

            # ---- two hands, both open palm for `exit.hold` s -> quit
            both_open = len(labels) == 2 and all(p == "open_palm" for p in labels)
            exit_progress = exit_timer.update(now, both_open)
            if exit_timer.done(now):
                print("[exit] both palms held open - bye!")
                break

            # ---- motion + continuous controls (they only gate what gets *dispatched*)
            fired, fired_gesture = None, None
            dispatch_gesture = stable
            volume_level = pinch_pts = mouse_ui = None
            pinching = pinch.active
            if pinch.active:
                volume_level = pinch.level
                pinch_pts = [(int(lm[i][0] * fw), int(lm[i][1] * fh)) for i in (4, 8)]
                if pinch.send is not None and not dispatcher.dry_run:  # DRY-RUN: only the bar moves
                    motion.set_system_volume(pinch.send)
            if both_open:
                dispatch_gesture = "unknown"
                swipe.reset()
            elif primary:
                states = infos[0].states
                if pinch.gating:  # pinch mode owns the hand: no static action, no swipe
                    swipe.reset()
                    dispatch_gesture = "unknown"
                else:
                    sw = swipe.update(now, motion.palm_center(lm), raw)
                    if sw:
                        fired, fired_gesture = dispatcher.trigger(sw, now), sw
                    dispatch_gesture = swipe.gate(stable, now)  # no ✋ action while swiping

                if dispatcher.mode == "mouse":
                    mf = mouse_logic.update(now, lm, iso=iso, states=states)
                    mouse_events += mf.events
                    mouse_ui = {"box": mouse_box, "pressed": mf.pressed, "scrolling": mf.scrolling,
                                "tip": (int(lm[8][0] * fw), int(lm[8][1] * fh))}
                    if mf.pressed or mf.scrolling:
                        dispatch_gesture = "unknown"
            if dispatcher.mode == "mouse" and mouse_ui is None:
                mouse_ui = {"box": mouse_box, "pressed": False, "scrolling": False, "tip": None}
            if dispatcher.dry_run:
                vmouse.release_all()  # never leave the real button held down
            else:
                vmouse.apply(mouse_events)

            static_fired = dispatcher.update(dispatch_gesture, now)
            if static_fired:
                fired, fired_gesture = static_fired, dispatch_gesture
            if fired:
                print(f"[action] {fired_gesture} -> {fired}")
                toast = {"label": fired, "gesture": fired_gesture, "t0": now,
                         "center": palm_center(frame.shape, primary) if primary else None}
            if toast and now - toast["t0"] > max(TOAST_SECONDS, PULSE_SECONDS):
                toast = None

            # ---- live gesture display (display only: classification + actions are above)
            fired_binding = dispatcher.bindings.get(fired_gesture) if fired else None
            if fired:
                hand_fx[0].flash_t = now  # ring flashes + pops when the action fires
            views, debug_hands = [], []
            shown = "pinch" if pinch.active else stable  # what the HUD / tag / legend display
            new_confirm = shown not in ("unknown", "pinch") and shown != prev_stable and not pinch.gating
            prev_stable = shown
            for i, hand in enumerate(hands):
                fx = hand_fx[i]
                if i == 0 and pinch.active:  # pinch mode: the ring shows the volume
                    disp, prog, confirmed = "pinch", (pinch.level or 0) / 100, True
                elif i == 0:  # hold-to-fire ring reads the smoother's votes (no logic change)
                    disp, prog, confirmed = D.hold_progress(smoother.history, smoother.history.maxlen,
                                                            smoother.min_ratio, stable)
                else:
                    disp, prog, confirmed = labels[i], None, labels[i] != "unknown"
                fx.fade.set(D.gesture_color(disp), now)
                col = fx.fade.value(now)
                pts = draw_hand(frame, hand, PINK, PURPLE, scale=ui.s, glow_color=col,
                                tips=infos[i].states, draw_box=False)
                box = fx.box.update(hand_box(pts, fw, fh, ui.s))
                views.append({"box": box, "color": col, "label": disp, "progress": prog,
                              "confirmed": confirmed, "tag_smoother": fx.tag, "flash_t": fx.flash_t,
                              "suffix": (f" · {pinch.level}%" if disp == "pinch" else "") if i == 0 else "  (2nd)"})
                debug_hands.append((pts, infos[i].tri))
            for i in range(len(hands), len(hand_fx)):
                hand_fx[i].reset()
            if new_confirm and views:
                pops.append({"emoji": ICONS.get(stable, "✨"), "t0": now, "box": views[0]["box"]})
            if fired:
                chip = f"{D.action_icon(fired_binding)} {fired}"
                recent_actions.appendleft(chip)
                if views:
                    floats.append({"text": chip, "color": D.gesture_color(fired_gesture), "t0": now,
                                   "box": views[0]["box"]})
                key0 = fired_binding.keys[0] if fired_binding and fired_binding.keys else None
                if key0 in VOLUME_KEYS:
                    meter_dir = VOLUME_KEYS[key0]
                    meter.tick(meter_dir, now)
            if volume_level is not None:
                meter.set_level(volume_level)  # the pinch slider knows the real level
            pops = [p for p in pops if now - p["t0"] < 0.8]
            floats = [f for f in floats if now - f["t0"] < 1.0]
            held = dispatcher.bindings.get(stable)
            holding_volume = bool(held and held.repeat and held.keys and held.keys[0] in VOLUME_KEYS)
            effects = {"pops": pops, "floats": floats}
            if views and meter.visible(now, holding_volume):
                effects["meter"] = {"box": views[0]["box"], "level": meter.level, "dir": meter_dir,
                                    "color": D.gesture_color("thumbs_up" if meter_dir > 0 else "thumbs_down"),
                                    "last_tick": meter.recent[-1] if meter.recent else float("-inf")}

            extra = [("both_palms", "✋✋ Both palms 1 s", "Quit", both_open)]
            if dispatcher.mode in volume_modes:
                extra.insert(0, ("pinch_vol", "🤏 Pinch (thumb+index)", "Volume slider", pinching))
            if dispatcher.mode == "mouse":
                pressed = bool(mouse_ui and mouse_ui["pressed"])
                scrolling = bool(mouse_ui and mouse_ui["scrolling"])
                extra[:0] = [("m_move", "☝️ Index tip", "Move cursor", bool(mouse_ui and mouse_ui["tip"])
                              and not pressed and not scrolling),
                             ("m_click", "🤏 Pinch / hold", "Click / drag", pressed),
                             ("m_scroll", "✌️ Peace, move up/down", "Scroll", scrolling)]

            debug = None
            if show_debug:
                if infos:
                    lines = debug_lines(infos[0])  # raw label lives here, not in the HUD
                    lines[0] += f"   ·   smoothed: {stable} ({smoother.confidence * 100:.0f}%)"
                    if ml:
                        lines[0] += f"   (ML: {labels[0]}, p={ml.last_confidence:.2f})"
                    if len(infos) > 1:
                        lines.append(f"2nd hand: {state_string(infos[1])}  ->  {labels[1]}")
                else:
                    lines = ["no hand"]
                lines.append("g hide · p save this hand as a test fixture")
                debug = {"lines": lines, "hands": debug_hands}
            prompt = None
            if pending is not None:
                prompt = ["Save this hand as a test fixture: which pose is it?",
                          "  ".join(f"{k} {ICONS.get(v, '')}" for k, v in KEY_LABELS.items()),
                          f"rules said: {pending['rule']}   ·   Esc cancels"]

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            hud_override = ("🤏 Pinch", f"→  Volume {pinch.level}%") if pinch.active else None
            ui.render(frame, raw=raw, stable=shown, conf=1.0 if pinch.active else smoother.confidence, fps=fps,
                      dry_run=dispatcher.dry_run, show_help=show_help,
                      hand_present=bool(hands), toast=toast, now=now,
                      bindings=dispatcher.bindings, mode=dispatcher.mode, extra_rows=extra,
                      volume=volume_level, pinch_pts=pinch_pts, mouse_ui=mouse_ui,
                      exit_progress=exit_progress, classifier=args.classifier.upper(),
                      debug=debug, prompt=prompt, hand_views=views, effects=effects,
                      recent_actions=list(recent_actions), hud_override=hud_override)

            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (fw, fh))
                writer.write(frame)

            cv2.imshow("Hand Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            ch = chr(key) if key < 128 else ""
            if pending is not None:  # waiting for the expected label of a saved hand
                if key == 27:
                    pending = None
                elif ch in KEY_LABELS:
                    label = KEY_LABELS[ch]
                    s = append_fixture(pending["landmarks"], pending["aspect"], label,
                                       pending["handedness"], path=args.fixtures)
                    print(f"[fixture] saved {label} (rules said {s['rule_output']}) -> {args.fixtures}")
                    toast = {"label": f"Saved: {pretty(label)}", "gesture": label, "t0": now,
                             "center": None, "caption": "TEST FIXTURE"}
                    pending = None
                continue
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                dispatcher.set_dry_run(not dispatcher.dry_run)
            if key == ord("m"):
                vmouse.apply(mouse_logic.reset())  # release a held drag before leaving mouse mode
                swipe.reset()
                if pinch_ctl.reset():
                    smoother.reset()
                hold.reset()
                print(f"[mode] {dispatcher.cycle_mode()}")
            if key == ord("h"):
                show_help = not show_help
            if key == ord("g"):
                show_debug = not show_debug
            if key == ord("p"):
                if primary:
                    pending = {"landmarks": list(primary.landmarks), "aspect": primary.aspect,
                               "handedness": primary.handedness, "rule": infos[0].label}
                else:
                    toast = {"label": "No hand to save", "gesture": "no hand", "t0": now,
                             "center": None, "caption": "TEST FIXTURE"}
    finally:
        vmouse.release_all()
        cap.release()
        tracker.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Actions fired this session: {len(dispatcher.log)}")


def main():
    ap = argparse.ArgumentParser(description="Hand gesture recognition demo")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="use a video file instead of the webcam")
    ap.add_argument("--dry-run", action="store_true", help="don't send real media keys")
    ap.add_argument("--record", help="save annotated output to this .mp4 path")
    ap.add_argument("--no-mirror", action="store_true", help="don't flip the webcam image")
    ap.add_argument("--config", help="path to config.yaml (default: ./config.yaml)")
    ap.add_argument("--classifier", choices=["rules", "ml"], default="rules",
                    help="static-pose classifier: hand-written rules or the trained model")
    ap.add_argument("--model", help="model file for --classifier ml (default: models/gesture_model.joblib)")
    ap.add_argument("--mode", help="start in this mode, e.g. media / slides / mouse "
                                   "(default: default_mode in config.yaml)")
    ap.add_argument("--fixtures", default=DEFAULT_FIXTURES,
                    help="where 'p' saves labelled hands (default: tests/fixtures/real_hands.json)")
    args = ap.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted - bye!")


if __name__ == "__main__":
    main()
