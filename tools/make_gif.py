"""Convert a recorded demo video into a README-sized GIF.

    python demo.py --dry-run --record demo.mp4      # record first (q to stop)
    python tools/make_gif.py                        # demo.mp4 -> docs/demo.gif
    python tools/make_gif.py clip.mp4 -o docs/demo.gif --fps 12 --width 720 --max-mb 10 \
        --start 2 --duration 12

Uses ffmpeg when it's on PATH (two-pass palette: best quality per MB), otherwise
OpenCV + Pillow, which this project already depends on. If the GIF comes out over
--max-mb it retries with fewer colours, then a lower frame rate, then a smaller width,
until it fits.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def _attempts(fps: float, width: int, colors: int):
    """Settings to try, best first: fewer colours, then lower fps, then narrower."""
    seen = set()
    for w in (width, int(width * 0.83), int(width * 0.67), int(width * 0.5)):
        for f in (fps, max(6.0, fps * 0.75), max(5.0, fps * 0.5)):
            for c in (colors, max(32, colors // 2)):
                key = (round(f, 2), w - w % 2, c)
                if key not in seen:
                    seen.add(key)
                    yield key


def _ffmpeg_gif(src, dst, fps, width, colors, start, duration):
    trim = (["-ss", str(start)] if start else []) + (["-t", str(duration)] if duration else [])
    scale = f"fps={fps},scale={width}:-2:flags=lanczos"
    with tempfile.TemporaryDirectory() as tmp:
        palette = os.path.join(tmp, "palette.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", *trim, "-i", src, "-vf",
                        f"{scale},palettegen=max_colors={colors}:stats_mode=diff", palette], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *trim, "-i", src, "-i", palette,
                        "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", dst], check=True)


def _read_frames(src, fps, width, start, duration):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Can't open video: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t, next_t, frames = 0.0, start, []
    end = start + duration if duration else float("inf")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / src_fps
        i += 1
        if t < start - 1e-9:
            continue
        if t > end:
            break
        if t + 1e-9 >= next_t:  # sample at the target frame rate
            h, w = frame.shape[:2]
            nh = max(2, round(h * width / w)) // 2 * 2
            frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            next_t += 1.0 / fps
    cap.release()
    if not frames:
        raise SystemExit("No frames read (check --start / --duration).")
    return frames


def _pillow_gif(frames, dst, fps, colors):
    from PIL import Image

    # one shared palette from a sample of frames: stable colours, smaller deltas between frames
    sample = frames[:: max(1, len(frames) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, fr in enumerate(sample):
        strip.paste(fr, (0, k * fr.height))
    palette = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    out = [fr.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for fr in frames]
    out[0].save(dst, save_all=True, append_images=out[1:], duration=frame_delays(len(out), fps),
                loop=0, optimize=True, disposal=1)


def frame_delays(n: int, fps: float) -> list[int]:
    """Per-frame delays in ms. GIF stores delays in 1/100 s, so 1000/12 = 83.3 ms can't be
    stored exactly; alternate 80/90 ms so the average (and the clip length) is right."""
    ticks = [round((k + 1) * 100 / fps) for k in range(n)]  # cumulative, in 1/100 s
    return [(b - a) * 10 for a, b in zip([0] + ticks[:-1], ticks)]


def make_gif(src, dst, fps=12.0, width=720, max_mb=10.0, colors=128, start=0.0, duration=None,
             backend="auto", verbose=True) -> dict:
    """Write the GIF; returns {"path", "mb", "fps", "width", "colors", "backend", "fits"}."""
    if not os.path.exists(src):
        raise SystemExit(f"No such video: {src}. Record one with: python demo.py --record {src}")
    if backend == "auto":
        backend = "ffmpeg" if shutil.which("ffmpeg") else "pillow"
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    cache: dict = {}
    result = None
    for f, w, c in _attempts(fps, width, colors):
        if backend == "ffmpeg":
            _ffmpeg_gif(src, dst, f, w, c, start, duration)
        else:
            key = (f, w)
            if key not in cache:
                cache.clear()  # keep memory to one frame set
                cache[key] = _read_frames(src, f, w, start, duration)
            _pillow_gif(cache[key], dst, f, c)
        mb = os.path.getsize(dst) / 1e6
        result = {"path": dst, "mb": mb, "fps": f, "width": w, "colors": c, "backend": backend,
                  "fits": mb <= max_mb}
        if verbose:
            print(f"[gif] {backend}: {w}px @ {f:g} fps, {c} colours -> {mb:.2f} MB")
        if result["fits"]:
            break
    if verbose and not result["fits"]:
        print(f"[gif] still {result['mb']:.1f} MB > {max_mb} MB at the smallest settings; "
              "try --duration to shorten the clip.", file=sys.stderr)
    return result


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Convert demo.mp4 into docs/demo.gif")
    ap.add_argument("video", nargs="?", default="demo.mp4")
    ap.add_argument("-o", "--out", default=os.path.join("docs", "demo.gif"))
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--max-mb", type=float, default=10.0)
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start")
    ap.add_argument("--duration", type=float, help="seconds to keep (default: all)")
    ap.add_argument("--backend", choices=["auto", "ffmpeg", "pillow"], default="auto")
    args = ap.parse_args(argv)
    res = make_gif(args.video, args.out, args.fps, args.width, args.max_mb, args.colors,
                   args.start, args.duration, args.backend)
    print(f"-> {res['path']} ({res['mb']:.2f} MB)")
    if not res["fits"]:
        raise SystemExit(1)
    return res


if __name__ == "__main__":
    main()
