"""tools/make_gif.py on a tiny synthetic video (Pillow backend: no ffmpeg needed)."""
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.make_gif import make_gif  # noqa: E402


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("gif") / "clip.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    for i in range(60):  # 2 s @ 30 fps: a pink dot moving across a gradient
        f = np.zeros((180, 320, 3), np.uint8)
        f[:] = np.linspace(20, 120, 320, dtype=np.uint8)[None, :, None]
        cv2.circle(f, (20 + i * 5, 90), 18, (216, 79, 255), -1)
        w.write(f)
    w.release()
    return path


def test_gif_size_fps_and_width(video, tmp_path):
    out = str(tmp_path / "docs" / "demo.gif")
    res = make_gif(video, out, fps=12, width=160, backend="pillow", verbose=False)
    assert res["fits"] and os.path.exists(out)
    with Image.open(out) as im:
        assert im.size[0] == 160 and im.n_frames in (24, 25)
        assert im.info.get("loop") == 0
        total = 0
        for k in range(im.n_frames):
            im.seek(k)
            total += im.info["duration"]
        assert total / im.n_frames == pytest.approx(1000 / 12, abs=1.0)  # 12 fps on average


def test_frame_delays_average_to_target():
    from tools.make_gif import frame_delays
    d = frame_delays(120, 12)
    assert set(d) <= {80, 90} and sum(d) == 10000  # 120 frames at 12 fps = exactly 10 s


def test_trim(video, tmp_path):
    res = make_gif(video, str(tmp_path / "t.gif"), fps=12, width=120, start=0.5, duration=1.0,
                   backend="pillow", verbose=False)
    with Image.open(res["path"]) as im:
        assert 11 <= im.n_frames <= 13


def test_shrinks_until_it_fits_or_reports(video, tmp_path):
    big = make_gif(video, str(tmp_path / "a.gif"), fps=12, width=320, backend="pillow", verbose=False)
    limit = big["mb"] * 0.6
    res = make_gif(video, str(tmp_path / "b.gif"), fps=12, width=320, max_mb=limit,
                   backend="pillow", verbose=False)
    assert res["fits"] and res["mb"] <= limit
    assert (res["width"], res["fps"], res["colors"]) != (320, 12, 128)  # it had to back off
    impossible = make_gif(video, str(tmp_path / "c.gif"), fps=12, width=320, max_mb=1e-6,
                          backend="pillow", verbose=False)
    assert not impossible["fits"] and os.path.exists(impossible["path"])


def test_missing_video_message(tmp_path):
    with pytest.raises(SystemExit, match="Record one"):
        make_gif(str(tmp_path / "nope.mp4"), str(tmp_path / "x.gif"), verbose=False)
