"""Real-time hand landmark detection using the MediaPipe Tasks API (HandLandmarker).

MediaPipe >= 0.10.30 removed the legacy `mp.solutions.hands` module, so this uses the
current Tasks API. The model file (~7.5 MB) is downloaded once on first run.
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


@dataclass
class Hand:
    landmarks: list[tuple[float, float, float]]  # 21 normalized (x, y, z) points
    handedness: str  # "Left" / "Right" (as seen by the model)
    score: float
    aspect: float = 1.0  # frame width / height

    @property
    def iso(self) -> list[tuple[float, float, float]]:
        """Landmarks with x and z scaled by width/height, so distances and angles are true
        to the image (MediaPipe normalises x by width but y by height). Use these for
        classification; use `landmarks` for drawing and screen mapping."""
        k = self.aspect
        return [(x * k, y, z * k) for x, y, z in self.landmarks]


def ensure_model(path: str = DEFAULT_MODEL_PATH) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[setup] Downloading hand landmark model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, max_hands: int = 2,
                 min_detection: float = 0.6, min_tracking: float = 0.5):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def process(self, frame_bgr) -> list[Hand]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs strictly increasing timestamps (ms)
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        h, w = frame_bgr.shape[:2]

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            hands.append(Hand(
                landmarks=[(p.x, p.y, p.z) for p in lms],
                handedness=handed[0].category_name,
                score=handed[0].score,
                aspect=w / h,
            ))
        return hands

    def close(self):
        self._landmarker.close()


# ---------------------------------------------------------------- drawing helpers
FINGERTIPS = (4, 8, 12, 16, 20)
# position of each landmark along its finger: 0 = wrist, 1 = fingertip (drives the gradient)
_FINGER_T = [0.0] + [k / 4 for _ in range(5) for k in range(1, 5)]


def rounded_rect(img, p1, p2, radius, color, thickness=-1, line_type=cv2.LINE_AA):
    """Filled (thickness < 0) or outlined rounded rectangle."""
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    if r == 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, line_type)
        return
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x2 - r, y2 - r), (x1 + r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, line_type)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, line_type)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, line_type)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, line_type)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, line_type)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, line_type)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, line_type)


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def palm_center(frame_shape, hand: Hand) -> tuple[int, int]:
    h, w = frame_shape[:2]
    idx = (0, 5, 9, 13, 17)
    return (int(sum(hand.landmarks[i][0] for i in idx) / 5 * w),
            int(sum(hand.landmarks[i][1] for i in idx) / 5 * h))


def hand_box(pts, frame_w: int, frame_h: int, scale: float):
    """Rounded box around the hand in pixels: (x1, y1, x2, y2)."""
    m = int(22 * scale)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (max(min(xs) - m, 0), max(min(ys) - m, 0),
            min(max(xs) + m, frame_w - 1), min(max(ys) + m, frame_h - 1))


GLOW_DOWNSCALE = 4  # the glow is blurred at 1/4 resolution: ~16x fewer pixels to blur
SEGMENTS = 2  # gradient pieces per bone (2 looks as smooth as 3 at bone width, 1/3 fewer lines)
_SEG_COLORS: dict = {}  # (color, color2) -> per-bone segment colours (constant, so cached)


def _joint_color(color, color2, i):
    key = (tuple(color), tuple(color2), i)
    c = _SEG_COLORS.get(key)
    if c is None:
        c = _SEG_COLORS[key] = lerp_color(color, color2, _FINGER_T[i])
    return c


def _segment_colors(color, color2):
    key = (tuple(color), tuple(color2))
    cols = _SEG_COLORS.get(key)
    if cols is None:
        cols = []
        for a, b in HAND_CONNECTIONS:
            ta, tb = _FINGER_T[a], _FINGER_T[b]
            cols.append([lerp_color(color, color2, ta + (tb - ta) * (k + 0.5) / SEGMENTS) for k in range(SEGMENTS)])
        _SEG_COLORS[key] = cols
    return cols


TIP_FINGER = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}


def draw_hand(frame, hand: Hand, color=(216, 79, 255), color2=(246, 92, 139),
              label: str | None = None, scale: float | None = None, text=None,
              glow_color=None, tips: dict | None = None, draw_box: bool = True):
    """Draw a glowing pink->purple hand skeleton and a rounded box around it.

    `label` draws a gesture tag on the box (demo.py passes None and places the tag itself,
    so it never covers a panel). `glow_color` tints the soft glow (default: the bone
    gradient). `tips` = {finger: extended} makes extended fingertips bigger and brighter
    (with a halo) and curled ones smaller and dimmer. `draw_box=False` skips the box. `text` is an optional renderer with
    ``measure(text, size, bold)`` / ``draw(img, text, (x, y), size, color, bold=...)``;
    without it the tag uses OpenCV Hershey text. Returns the landmark pixel coordinates.
    """
    h, w = frame.shape[:2]
    s = scale if scale is not None else w / 1280
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
    bone = max(2, round(4 * s))

    seg_cols = _segment_colors(color, color2)

    def bone_segments():
        # split each bone into SEGMENTS pieces so the gradient looks continuous
        for (a, b), cols in zip(HAND_CONNECTIONS, seg_cols):
            (xa, ya), (xb, yb) = pts[a], pts[b]
            for k in range(SEGMENTS):
                u0, u1 = k / SEGMENTS, (k + 1) / SEGMENTS
                p = (int(xa + (xb - xa) * u0), int(ya + (yb - ya) * u0))
                q = (int(xa + (xb - xa) * u1), int(ya + (yb - ya) * u1))
                yield p, q, cols[k]

    # ---- soft glow: drawn + blurred on a downscaled copy of the hand's ROI only
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = int(40 * s)
    gx1, gy1 = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
    gx2, gy2 = min(max(xs) + pad, w), min(max(ys) + pad, h)
    f = GLOW_DOWNSCALE
    sw, sh = (gx2 - gx1) // f, (gy2 - gy1) // f
    if sw > 2 and sh > 2:
        small = np.zeros((sh, sw, 3), np.uint8)
        thick = max(1, round(bone * 4 / f))
        if glow_color is not None:  # one colour: whole bones in a single polylines call
            segs = [np.array([((pts[a][0] - gx1) // f, (pts[a][1] - gy1) // f),
                              ((pts[b][0] - gx1) // f, (pts[b][1] - gy1) // f)], np.int32)
                    for a, b in HAND_CONNECTIONS]
            cv2.polylines(small, segs, False, glow_color, thick, cv2.LINE_AA)
        else:
            for p, q, c in bone_segments():
                cv2.line(small, ((p[0] - gx1) // f, (p[1] - gy1) // f), ((q[0] - gx1) // f, (q[1] - gy1) // f),
                         c, thick, cv2.LINE_AA)
        if tips:  # halo around the fingertips the classifier sees as extended
            for i, finger in TIP_FINGER.items():
                if tips.get(finger):
                    cv2.circle(small, ((pts[i][0] - gx1) // f, (pts[i][1] - gy1) // f),
                               max(2, round(16 * s / f)), (255, 255, 255), -1, cv2.LINE_AA)
        k = max(3, int(21 * s / f)) | 1
        small = cv2.GaussianBlur(small, (k, k), 0)
        roi = frame[gy1:gy1 + sh * f, gx1:gx1 + sw * f]
        glow = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(roi, 1.0, glow, 0.75, 0, dst=roi)

    # ---- crisp gradient bones
    for p, q, c in bone_segments():
        cv2.line(frame, p, q, c, bone, cv2.LINE_AA)

    # ---- joints: small dots; fingertips bigger with a white outline
    for i, p in enumerate(pts):
        c = seg_cols[0][0] if i == 0 else _joint_color(color, color2, i)
        if i in FINGERTIPS:
            extended = True if tips is None else bool(tips.get(TIP_FINGER[i]))
            if extended:
                r = max(4, round((11 if tips else 9) * s))
                cv2.circle(frame, p, r + max(1, round((2.5 if tips else 2) * s)), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, p, r, lerp_color(c, (255, 255, 255), 0.25) if tips else c, -1, cv2.LINE_AA)
            else:  # curled: small and dim
                r = max(3, round(5 * s))
                cv2.circle(frame, p, r + 1, (60, 45, 60), -1, cv2.LINE_AA)
                cv2.circle(frame, p, r, lerp_color(c, (110, 95, 110), 0.6), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, p, max(2, round(4 * s)), c, -1, cv2.LINE_AA)

    # ---- rounded bounding box (+ optional tag)
    bx1, by1, bx2, by2 = hand_box(pts, w, h, s)
    if draw_box:
        rounded_rect(frame, (bx1, by1), (bx2, by2), int(18 * s), color, max(1, round(2 * s)))
    if label and draw_box:
        size = max(12, round(18 * s))
        if text is not None:
            tw, th = text.measure(label, size, True)
        else:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size / 30, max(1, round(2 * s)))
        px, py = int(12 * s), int(6 * s)
        tag_h = th + 2 * py
        tx1, ty1 = bx1, by1 - tag_h - int(6 * s)
        if ty1 < 0:  # no room above the box -> tuck the tag inside it
            tx1, ty1 = bx1 + int(6 * s), by1 + int(6 * s)
        rounded_rect(frame, (tx1, ty1), (tx1 + tw + 2 * px, ty1 + tag_h), tag_h // 2, color, -1)
        if text is not None:
            text.draw(frame, label, (tx1 + px, ty1 + py), size, (255, 255, 255), bold=True)
        else:
            cv2.putText(frame, label, (tx1 + px, ty1 + py + th), cv2.FONT_HERSHEY_SIMPLEX,
                        size / 30, (255, 255, 255), max(1, round(2 * s)), cv2.LINE_AA)
    return pts
