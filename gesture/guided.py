"""Guided data collection: walk through every gesture in one session.

Pure logic (no camera, no drawing) so it's unit-tested; collect_data.py --guided drives
it with webcam frames and draws its state.

Per gesture: COUNTDOWN (3-2-1) -> RECORDING (auto, while a hand is visible, until
`per_gesture` samples) -> next gesture. SPACE pauses/resumes, skip keeps what was recorded
and moves on, redo throws away this gesture's rows and starts it again, quit saves and stops.

Rows are buffered per gesture and only handed to `write_rows` when the gesture is
finished, skipped, or the session is quit, so "redo" can discard them without editing
the CSV. Samples are rate-limited (`max_rate`) and near-duplicate frames are skipped, so
the dataset is varied. Each gesture's recording is split into `takes_per_gesture`
takes (plus a new take after every pause), so train.py can split by take.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

GUIDED_ORDER = ["open_palm", "fist", "thumbs_up", "thumbs_down", "peace", "pointing",
                "rock", "ok", "call_me", "unknown"]

HINTS = {
    "open_palm": "Spread all five fingers, palm towards the camera",
    "fist": "Curl all four fingers, thumb folded over them",
    "thumbs_up": "Make a fist and point your thumb up",
    "thumbs_down": "Make a fist and point your thumb down",
    "peace": "Index + middle finger up in a V, the rest curled",
    "pointing": "Only your index finger up",
    "rock": "Index + pinky up, middle + ring down (thumb out 🤟 or tucked 🤘)",
    "ok": "Touch thumb and index tips, other three fingers up",
    "call_me": "Thumb and pinky out, the other fingers curled",
    "unknown": "Anything else: relaxed, half-curled, random in-between poses",
}
VARY_TIP = "Now vary it: move closer / further, tilt your hand, switch hands"

COUNTDOWN, RECORDING, DONE = "countdown", "recording", "done"


def near_duplicate(a, b, eps: float) -> bool:
    """Mean absolute difference of two normalised feature vectors below eps (palm units)."""
    return a is not None and b is not None and float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))) < eps


@dataclass
class GestureResult:
    gesture: str
    samples: int = 0
    rows: int = 0
    status: str = "pending"  # pending | done | skipped | partial (quit mid-way)


@dataclass
class GuidedSession:
    write_rows: Callable[[list[dict]], None]
    gestures: list[str] = field(default_factory=lambda: list(GUIDED_ORDER))
    per_gesture: int = 200
    countdown: float = 3.0
    max_rate: float = 10.0       # samples per second
    dedup_eps: float = 0.015     # mean |Δfeature| in palm units below this = same frame
    takes_per_gesture: int = 5
    session: str = "guided"

    def __post_init__(self):
        if not self.gestures:
            raise ValueError("need at least one gesture")
        self.index = 0
        self.results = [GestureResult(g) for g in self.gestures]
        self.paused = False
        self._buffer: list[dict] = []
        self._attempt = 0
        self._pause_take = 0
        self._phase_t0: float | None = None
        self._paused_at: float | None = None
        self._last_t = float("-inf")
        self._last_feats = None
        self.phase = COUNTDOWN

    # ---------------------------------------------------------- state
    @property
    def gesture(self) -> str | None:
        return None if self.phase == DONE else self.gestures[self.index]

    @property
    def count(self) -> int:
        return 0 if self.phase == DONE else self.results[self.index].samples

    @property
    def progress(self) -> float:
        return min(1.0, self.count / self.per_gesture) if self.per_gesture else 1.0

    @property
    def show_vary_tip(self) -> bool:
        return self.phase == RECORDING and self.progress >= 0.5

    @property
    def done(self) -> bool:
        return self.phase == DONE

    def position(self) -> tuple[int, int]:
        """(1-based gesture number, total) for 'Gesture 3/10'."""
        return min(self.index + 1, len(self.gestures)), len(self.gestures)

    def countdown_left(self, now: float) -> int:
        """3, 2, 1 during the countdown (0 once recording)."""
        if self.phase != COUNTDOWN or self._phase_t0 is None:
            return int(self.countdown) if self.phase == COUNTDOWN else 0
        ref = self._paused_at if self.paused else now
        return max(0, int(np.ceil(self.countdown - (ref - self._phase_t0))))

    def take_id(self) -> str:
        seg = min(self.takes_per_gesture - 1, self.count * self.takes_per_gesture // max(1, self.per_gesture))
        return f"{self.session}-{self.gesture}-a{self._attempt}-p{self._pause_take}-t{seg}"

    # ---------------------------------------------------------- time
    def update(self, now: float):
        """Advance the countdown. Call once per frame."""
        if self.phase == DONE or self.paused:
            return
        if self._phase_t0 is None:
            self._phase_t0 = now
        if self.phase == COUNTDOWN and now - self._phase_t0 >= self.countdown:
            self.phase = RECORDING
            self._phase_t0 = now

    def offer(self, now: float, features, make_rows: Callable[[str], list[dict]]) -> bool:
        """A frame with a hand. Records it if it's time and it isn't a near-duplicate.

        `make_rows(take_id)` builds the CSV row(s) for this sample (2 with mirror augment).
        Returns True if the sample was kept.
        """
        self.update(now)
        if self.phase != RECORDING or self.paused:
            return False
        if now - self._last_t < 1.0 / self.max_rate - 1e-9:
            return False
        if near_duplicate(self._last_feats, features, self.dedup_eps):
            return False
        rows = make_rows(self.take_id())
        self._buffer.extend(rows)
        res = self.results[self.index]
        res.samples += 1
        res.rows += len(rows)
        self._last_t, self._last_feats = now, np.asarray(features, dtype=float)
        if res.samples >= self.per_gesture:
            self._finish("done", now)
        return True

    # ---------------------------------------------------------- controls
    def toggle_pause(self, now: float) -> bool:
        if self.phase == DONE:
            return False
        if self.paused:
            if self._phase_t0 is not None and self._paused_at is not None:
                self._phase_t0 += now - self._paused_at  # freeze the countdown while paused
            self.paused, self._paused_at = False, None
            self._pause_take += 1  # a new take after every pause
        else:
            self.paused, self._paused_at = True, now
        return self.paused

    def skip(self, now: float):
        """Keep what this gesture recorded so far and move on."""
        if self.phase != DONE:
            self._finish("skipped", now)

    def redo(self, now: float):
        """Throw away this gesture's rows from this session and start it again."""
        if self.phase == DONE:
            return
        self._buffer.clear()
        res = self.results[self.index]
        res.samples = res.rows = 0
        res.status = "pending"
        self._attempt += 1
        self._pause_take = 0
        self._restart(now)

    def quit(self, now: float):
        """Save the current gesture's rows and stop."""
        if self.phase != DONE:
            res = self.results[self.index]
            self._flush()
            if res.status == "pending":
                res.status = "partial" if res.samples else "skipped"
            self.phase = DONE

    # ---------------------------------------------------------- internals
    def _flush(self):
        if self._buffer:
            self.write_rows(list(self._buffer))
            self._buffer.clear()

    def _restart(self, now: float):
        self.phase = COUNTDOWN
        self._phase_t0 = now
        self.paused, self._paused_at = False, None
        self._last_t, self._last_feats = float("-inf"), None

    def _finish(self, status: str, now: float):
        self._flush()
        self.results[self.index].status = status
        self.index += 1
        self._attempt = self._pause_take = 0
        if self.index >= len(self.gestures):
            self.phase = DONE
        else:
            self._restart(now)

    # ---------------------------------------------------------- report
    def summary_table(self) -> str:
        w = max(len(r.gesture) for r in self.results)
        lines = [f"{'gesture':<{w}}  samples   rows  status", f"{'-' * w}  -------  -----  -------"]
        for r in self.results:
            lines.append(f"{r.gesture:<{w}}  {r.samples:>7}  {r.rows:>5}  {r.status}")
        lines.append(f"{'-' * w}  -------  -----")
        lines.append(f"{'total':<{w}}  {sum(r.samples for r in self.results):>7}  "
                     f"{sum(r.rows for r in self.results):>5}")
        return "\n".join(lines)
