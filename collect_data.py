"""Record labelled hand-landmark samples for the ML classifier.

    python collect_data.py                    # appends to data/landmarks.csv
    python collect_data.py --mirror-augment   # also save an x-flipped copy of every sample
    python collect_data.py --guided           # walk through every gesture in one session

Guided mode (--guided [--per-gesture 200]): shows each gesture with a how-to hint,
counts down 3-2-1, then records automatically while your hand is visible (max ~10
samples/s, near-duplicate frames skipped) until it has N samples, then moves on.
  SPACE pause / resume   s skip gesture   r redo gesture (drops its rows)   q / Esc save + quit

Manual mode keys:
  1-9      choose the gesture label (1 open_palm ... 9 call_me), 0 = "unknown" (any other pose)
  SPACE    hold to record (one hold = one "take")
  r        toggle recording on/off without holding SPACE
  q / Esc  quit

Features are normalised by gesture/features.py (wrist at the origin, scaled by palm size),
exactly as at training and inference time. Raw landmarks are saved too, so evaluate.py
can run the rule-based classifier on the same samples.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter

import cv2

from demo import AMBER, GLASS, GREEN, ICONS, MUTED, PINK, PURPLE, TRACK, WHITE, Card, TextRenderer, pretty
from gesture.features import CSV_COLUMNS, KEY_LABELS, extract_features, rows_for_sample
from gesture.guided import COUNTDOWN, GUIDED_ORDER, HINTS, VARY_TIP, GuidedSession
from gesture.hand_tracker import HandTracker, draw_hand, rounded_rect

REC_RED = (80, 80, 255)
FIRST_HOLD = 0.65   # s: covers the keyboard's initial auto-repeat delay after pressing SPACE
REPEAT_HOLD = 0.25  # s: gap between auto-repeat events while SPACE stays down
KEY_ORDER = list("1234567890")


def existing_counts(path: str) -> tuple[Counter, bool]:
    """Label counts already in the CSV; also checks the header matches this version."""
    counts: Counter = Counter()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return counts, False
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise SystemExit(f"{path} has a different column layout. Move it aside or use --out.")
        for row in reader:
            counts[row["label"]] += 1
    return counts, True


class HoldDetector:
    """'SPACE is held' from OpenCV key events (which have no key-up): true while
    auto-repeat events keep arriving."""

    def __init__(self):
        self.first = self.last = float("-inf")

    def press(self, now: float) -> bool:
        """Register a SPACE event; returns True if it starts a new hold."""
        new = now - self.last > FIRST_HOLD
        if new:
            self.first = now
        self.last = now
        return new

    def held(self, now: float) -> bool:
        window = FIRST_HOLD if self.last - self.first < FIRST_HOLD else REPEAT_HOLD
        return now - self.last < window


def draw_panel(frame, s, text, x1, y1, x2, y2, border=PINK):
    roi = frame[max(y1, 0):y2, max(x1, 0):x2]
    over = roi.copy()
    rounded_rect(over, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), int(18 * s), GLASS, -1)
    cv2.addWeighted(over, 0.62, roi, 0.38, 0, dst=roi)
    rounded_rect(frame, (x1, y1), (x2, y2), int(18 * s), border, max(1, round(1.5 * s)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record labelled landmarks for train.py")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("data", "landmarks.csv"))
    ap.add_argument("--mirror-augment", action="store_true",
                    help="also save an x-flipped copy (other hand) of every sample")
    ap.add_argument("--no-mirror", action="store_true", help="don't flip the webcam image")
    ap.add_argument("--rate", type=float, default=None,
                    help="max samples per second while recording (default: 15 manual, 10 guided)")
    ap.add_argument("--guided", action="store_true", help="record every gesture in one guided session")
    ap.add_argument("--per-gesture", type=int, default=200, help="guided: samples per gesture")
    args = ap.parse_args(argv)
    if args.guided:
        return run_guided(args)
    args.rate = args.rate or 15.0

    counts, has_header = existing_counts(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Try --camera 1 or check permissions.")
    tracker = HandTracker(max_hands=1)
    text = TextRenderer()

    session = time.strftime("%Y%m%d-%H%M%S")
    key = "1"
    hold, toggled = HoldDetector(), False
    take_n, sample_n, last_sample = 0, 0, float("-inf")
    session_counts: Counter = Counter()
    f = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    if not has_header:
        writer.writeheader()

    print(f"Recording to {args.out}  (mirror augment: {'on' if args.mirror_augment else 'off'})")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)  # same view as demo.py, so handedness matches
            h, w = frame.shape[:2]
            s = w / 1280
            now = time.monotonic()
            label = KEY_LABELS[key]

            hands = tracker.process(frame)
            recording = toggled or hold.held(now)
            if hands:
                draw_hand(frame, hands[0], PINK, PURPLE, label=f"{ICONS.get(label, '')} {pretty(label)}",
                          scale=s, text=text)
                if recording and now - last_sample >= 1.0 / args.rate:
                    # isotropic coords (x, z scaled by width/height), like the classifier sees them
                    for row in rows_for_sample(label, hands[0].iso, hands[0].handedness,
                                               f"{session}-{sample_n}", f"{session}-t{take_n}",
                                               time.time(), args.mirror_augment):
                        writer.writerow(row)
                        counts[label] += 1
                        session_counts[label] += 1
                    sample_n += 1
                    last_sample = now
                    if sample_n % 15 == 0:
                        f.flush()

            # ---- HUD: current label + recording state
            px = lambda v: max(1, round(v * s))  # noqa: E731
            big, body, small = round(34 * s), round(19 * s), round(15 * s)
            title = f"[{key}]  {ICONS.get(label, '')} {pretty(label)}"
            tw, th = text.measure(title, big, True)
            x1, y1 = px(22), px(22)
            x2, y2 = x1 + max(tw, px(330)) + px(36), y1 + th + px(80)
            draw_panel(frame, s, text, x1, y1, x2, y2, REC_RED if recording else PINK)
            text.draw(frame, "LABEL", (x1 + px(18), y1 + px(12)), small, MUTED, bold=True)
            text.draw(frame, title, (x1 + px(18), y1 + px(30)), big, WHITE, bold=True)
            if recording:
                blink = int(now * 3) % 2 == 0
                cv2.circle(frame, (x1 + px(26), y2 - px(24)), px(7), REC_RED if blink else (40, 40, 140),
                           -1, cv2.LINE_AA)
                state = f"REC  take {take_n}" + ("" if hands else "  (no hand)")
                text.draw(frame, state, (x1 + px(42), y2 - px(24)), body, REC_RED, bold=True, anchor="lm")
            else:
                text.draw(frame, "Hold SPACE to record  ·  r toggle", (x1 + px(18), y2 - px(24)),
                          body, MUTED, anchor="lm")

            # ---- live sample counts per label
            row_h = round(30 * s)
            cw = px(330)
            cx2, cy1 = w - px(22), px(22)
            cx1 = cx2 - cw
            cy2 = cy1 + px(48) + row_h * len(KEY_ORDER) + px(30)
            draw_panel(frame, s, text, cx1, cy1, cx2, cy2)
            text.draw(frame, "SAMPLES  (session +new)", (cx1 + px(18), cy1 + px(14)), small, MUTED, bold=True)
            top = max(counts.values(), default=1) or 1
            for i, k in enumerate(KEY_ORDER):
                lab = KEY_LABELS[k]
                ry = cy1 + px(44) + i * row_h
                active = k == key
                if active:
                    rounded_rect(frame, (cx1 + px(8), ry), (cx2 - px(8), ry + row_h), px(10), PINK,
                                 max(1, round(1.5 * s)))
                bar_w = int((cw - px(36)) * counts[lab] / top)
                if bar_w > 0:
                    cv2.rectangle(frame, (cx1 + px(18), ry + row_h - px(6)),
                                  (cx1 + px(18) + bar_w, ry + row_h - px(4)), PURPLE, -1)
                text.draw(frame, f"{k}  {ICONS.get(lab, '')} {pretty(lab)}", (cx1 + px(18), ry + row_h // 2 - px(2)),
                          body, WHITE, bold=active, anchor="lm")
                new = f" +{session_counts[lab]}" if session_counts[lab] else ""
                text.draw(frame, f"{counts[lab]}{new}", (cx2 - px(18), ry + row_h // 2 - px(2)),
                          body, WHITE if active else MUTED, bold=active, anchor="rm")
            aug = "mirror augment ON" if args.mirror_augment else "mirror augment off"
            text.draw(frame, aug, (cx1 + px(18), cy2 - px(22)), small, MUTED)

            cv2.imshow("Collect gesture data", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            ch = chr(k) if k < 128 else ""
            if ch in KEY_LABELS:
                key = ch
            elif k == ord(" "):
                if hold.press(now) and not toggled:
                    take_n += 1
            elif ch == "r":
                toggled = not toggled
                if toggled:
                    take_n += 1
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        total = sum(session_counts.values())
        print(f"Saved {total} rows this session -> {args.out}")
        for lab, n in sorted(session_counts.items()):
            print(f"  {lab:12s} +{n}  (total {counts[lab]})")




# ------------------------------------------------------------------ guided mode
def _guided_card(text, s, w, session, now, hand):
    """Big top-centre card for the current gesture (cached per content)."""
    g = session.gesture
    n, total = session.position()
    px = lambda v: max(1, round(v * s))  # noqa: E731
    big, body, small = round(42 * s), round(18 * s), round(13 * s)
    if session.paused:
        status, scol = "⏸  Paused · SPACE to resume", AMBER
    elif session.phase == COUNTDOWN:
        status, scol = f"Get ready…  {session.countdown_left(now)}", WHITE
    elif not hand:
        status, scol = "Show your hand ✋", AMBER
    else:
        status, scol = f"●  REC   {session.count} / {session.per_gesture}", REC_RED
    tip = VARY_TIP if session.show_vary_tip else ""
    lines = [(f"GESTURE {n}/{total}", small, MUTED, True),
             (f"{ICONS.get(g, '')}  {pretty(g)}", big, WHITE, True),
             (HINTS.get(g, ""), body, MUTED, False),
             (status, body, scol, True),
             (tip, body, GREEN, True),
             ("SPACE pause · s skip · r redo · q save & quit", small, MUTED, False)]
    sizes = [text.measure(t or " ", sz, b) for t, sz, _, b in lines]
    pad, gap = px(18), px(6)
    cw = min(max(max(sw for sw, _ in sizes) + 2 * pad, px(560)), w - px(32))
    bar_gap = px(14)
    ch = pad * 2 + sum(sh for _, sh in sizes) + gap * (len(lines) - 1) + bar_gap
    key = ("guided", tuple(t for t, *_ in lines), cw)

    def build():
        c = Card(cw, ch, px(20), border=REC_RED if status.startswith("●") else PINK, border_px=px(2))
        y = pad
        bar_y = 0
        for i, ((t, sz, col, b), (_, sh)) in enumerate(zip(lines, sizes)):
            if t:
                c.text(text, t, (cw // 2, y), sz, col, bold=b, anchor="mt")
            y += sh + gap
            if i == 3:  # progress bar goes under the status line
                bar_y = y
                y += bar_gap
        c.bar_y = bar_y
        return c
    return key, build, cw, ch


def run_guided(args):
    rate = args.rate or 10.0
    counts, has_header = existing_counts(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Try --camera 1 or check permissions.")
    tracker = HandTracker(max_hands=1)
    text = TextRenderer()
    sid = time.strftime("%Y%m%d-%H%M%S")

    f = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    if not has_header:
        writer.writeheader()

    def write_rows(rows):  # called when a gesture is finished / skipped / on quit
        writer.writerows(rows)
        f.flush()
        for r in rows:
            counts[r["label"]] += 1

    session = GuidedSession(write_rows, list(GUIDED_ORDER), per_gesture=args.per_gesture,
                            max_rate=rate, session=f"{sid}-guided")
    cards: dict = {}
    sample_n = 0
    done_at = None
    print(f"Guided session -> {args.out}  ({args.per_gesture} samples per gesture, "
          f"max {rate:g}/s, mirror augment {'on' if args.mirror_augment else 'off'})")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            s = w / 1280
            now = time.monotonic()
            hands = tracker.process(frame)
            session.update(now)

            if session.done:
                done_at = done_at or now
            elif hands:
                hand = hands[0]
                label = session.gesture
                feats = extract_features(hand.iso, hand.handedness)[:63]

                def make_rows(take, hand=hand, label=label):
                    return rows_for_sample(label, hand.iso, hand.handedness, f"{sid}-g{sample_n}", take,
                                           time.time(), args.mirror_augment)
                if session.offer(now, feats, make_rows):
                    sample_n += 1
                draw_hand(frame, hand, PINK, PURPLE, scale=s)

            # ---- UI
            if session.done:
                msg = "✅  All gestures recorded — saving…"
                tw, th = text.measure(msg, round(30 * s), True)
                key = ("done", msg)
                if key not in cards:
                    c = Card(tw + round(60 * s), th + round(40 * s), round(20 * s), border=GREEN, border_px=2)
                    c.text(text, msg, (c.w // 2, c.h // 2), round(30 * s), WHITE, bold=True, anchor="mm")
                    cards[key] = c.finalize()
                cards[key].blit(frame, w // 2 - cards[key].w // 2, round(24 * s))
            else:
                key, build, cw, ch = _guided_card(text, s, w, session, now, bool(hands))
                if key not in cards:
                    if len(cards) > 64:
                        cards.clear()
                    cards[key] = build().finalize()
                card = cards[key]
                x1, y1 = w // 2 - cw // 2, round(24 * s)
                card.blit(frame, x1, y1)
                bx, bw, bh = x1 + round(24 * s), cw - round(48 * s), max(4, round(8 * s))
                by = y1 + card.bar_y
                rounded_rect(frame, (bx, by), (bx + bw, by + bh), bh // 2, TRACK, -1)
                fill = int(bw * session.progress)
                if fill > bh:
                    rounded_rect(frame, (bx, by), (bx + fill, by + bh), bh // 2, PINK, -1)
                if session.phase == COUNTDOWN and not session.paused:  # huge 3-2-1 in the middle
                    n = str(session.countdown_left(now))
                    spr_size = round(160 * s)
                    text.draw(frame, n, (w // 2, h // 2), spr_size, WHITE, bold=True, anchor="mm", alpha=0.85)

            cv2.imshow("Collect gesture data (guided)", frame)
            k = cv2.waitKey(1) & 0xFF
            if done_at is not None and now - done_at > 1.2:
                break
            if k in (ord("q"), 27):
                session.quit(now)
                break
            if k == ord(" "):
                session.toggle_pause(now)
            elif k == ord("s"):
                print(f"[guided] skipped {session.gesture} ({session.count} samples kept)")
                session.skip(now)
            elif k == ord("r"):
                print(f"[guided] redo {session.gesture}")
                session.redo(now)
    except KeyboardInterrupt:
        pass
    finally:
        if not session.done:
            session.quit(time.monotonic())  # save whatever the current gesture has
        f.close()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("\n" + session.summary_table())
        if args.mirror_augment:
            print("(rows include the mirrored copies)")
        print(f"Saved to {args.out}")
    return session


if __name__ == "__main__":
    main()
