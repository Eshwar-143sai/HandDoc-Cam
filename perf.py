"""
Frame-rate profiling for the live loop, plus the on-screen overlay that shows it.

The loop calls ``FrameProfiler.tick()`` once per processed camera frame and ``mark("stage")`` after each stage
(hand inference, access gate, render, virtual-camera send, preview UI). From that the profiler derives:

* **fps**        frames per second over the last ~1 second (what a call participant actually receives)
* **1% low**     the fps equivalent of the slowest 1 % of recent frames (exposes stutter an average hides)
* **busy / headroom**  CPU time spent per frame versus the time available at the target fps, so you can see how
                 much slack there is *before* a frame is dropped - the number that tells you whether an
                 optimisation really helped even when the camera caps fps at 30
* **per-stage ms**  which stage eats the budget

Only the local preview shows the overlay; it is never drawn into the virtual-camera stream.
"""
from __future__ import annotations

import time
from array import array
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

STAGE_ORDER: tuple[str, ...] = ("infer", "gate", "render", "send", "ui")
STAGE_LABELS: dict[str, str] = {"infer": "hands", "gate": "gate", "render": "render", "send": "vcam", "ui": "preview"}
GREEN, AMBER, RED = (80, 220, 80), (0, 190, 255), (70, 70, 255)  # BGR


@dataclass(frozen=True)
class PerfSnapshot:
    """A point-in-time view of the profiler, cheap to compute and safe to pass around."""

    frames: int                       # frames processed so far
    fps: float                        # frames per second over the last ~1 s
    frame_ms: float                   # mean time between frames over that window
    low1_fps: float                   # fps equivalent of the slowest 1 % of recent frames
    busy_ms: float                    # smoothed processing time per frame (sum of the stages)
    budget_ms: float                  # time available per frame at the target fps
    headroom_pct: float               # share of the budget still unused (0 when over budget)
    dropped: int                      # recent frames slower than 1.5x the budget (visible stutters)
    stage_ms: dict[str, float]        # smoothed milliseconds per stage
    intervals_ms: tuple[float, ...]   # most recent frame intervals, for the frame-time graph


class FrameProfiler:
    """Measures frame rate and per-stage cost of a real-time loop.

    Args:
        target_fps: the rate the stream is meant to hold (sets the per-frame time budget).
        history: how many recent frames feed the 1 % low and stutter count.
        ema: smoothing factor (0..1) for the per-stage and busy-time averages.
        clock: a monotonic seconds clock; injectable so tests can drive time deterministically.
    """

    def __init__(self, target_fps: float = 30.0, history: int = 300, ema: float = 0.1,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        """Start with empty history; the arguments are described in the class docstring."""
        self.target_fps = max(1.0, float(target_fps))
        self._clock, self._ema = clock, float(ema)
        self._stamps: deque[float] = deque(maxlen=max(3, history))     # start time of each frame
        self._intervals: deque[float] = deque(maxlen=max(3, history))  # seconds between frames
        self._all = array("f")                                         # every interval, for the run summary
        self._cur: dict[str, float] = {}                               # stage seconds of the frame in progress
        self._mark_t: float | None = None
        self._stage_ema: dict[str, float] = {}
        self._stage_sum: dict[str, float] = {}
        self._busy_ema = 0.0
        self._folded = 0
        self._frames = 0
        self._t_first: float | None = None

    def tick(self) -> None:
        """Call once at the start of every processed frame (also closes the previous frame's stage timings)."""
        now = self._clock()
        if self._stamps:
            iv = now - self._stamps[-1]
            self._intervals.append(iv)
            self._all.append(iv)
        self._fold_current()
        self._stamps.append(now)
        self._mark_t = now
        self._frames += 1
        if self._t_first is None:
            self._t_first = now

    def mark(self, stage: str) -> None:
        """Attribute the time since the previous mark (or tick) to ``stage``."""
        now = self._clock()
        if self._mark_t is not None:
            self._cur[stage] = self._cur.get(stage, 0.0) + (now - self._mark_t)
        self._mark_t = now

    def _fold_current(self) -> None:
        """Merge the finished frame's stage times into the smoothed averages and run totals."""
        if not self._cur:
            return
        a, busy = self._ema, sum(self._cur.values())
        self._busy_ema = busy if self._folded == 0 else (1 - a) * self._busy_ema + a * busy
        for name, sec in self._cur.items():
            prev = self._stage_ema.get(name)
            self._stage_ema[name] = sec if prev is None else (1 - a) * prev + a * sec
            self._stage_sum[name] = self._stage_sum.get(name, 0.0) + sec
        self._folded += 1
        self._cur = {}

    def snapshot(self) -> PerfSnapshot:
        """Current statistics. Returns zeros until at least two frames have been seen."""
        budget = 1000.0 / self.target_fps
        stamps = self._stamps
        fps = 0.0
        if len(stamps) >= 2:
            end, i = stamps[-1], len(stamps) - 1
            while i > 0 and end - stamps[i - 1] <= 1.0:  # walk back over the last second
                i -= 1
            span = end - stamps[i]
            fps = (len(stamps) - 1 - i) / span if span > 0 else 0.0
        iv = np.fromiter(self._intervals, dtype=np.float64, count=len(self._intervals))
        low1 = fps
        dropped = 0
        if iv.size >= 20:
            low1 = 1.0 / float(np.percentile(iv, 99))
            dropped = int((iv > 1.5 / self.target_fps).sum())
        busy_ms = self._busy_ema * 1000.0
        return PerfSnapshot(
            frames=self._frames, fps=fps, frame_ms=1000.0 / fps if fps > 0 else 0.0, low1_fps=low1,
            busy_ms=busy_ms, budget_ms=budget,
            headroom_pct=float(np.clip(100.0 * (1.0 - busy_ms / budget), 0.0, 100.0)),
            dropped=dropped, stage_ms={k: v * 1000.0 for k, v in self._stage_ema.items()},
            intervals_ms=tuple(float(x) * 1000.0 for x in iv[-90:]))

    def summary(self) -> str:
        """One-paragraph report for the whole run (printed when the app exits)."""
        if self._frames < 2 or self._t_first is None or not self._stamps:
            return "Performance: not enough frames to report."
        span = self._stamps[-1] - self._t_first
        avg = (self._frames - 1) / span if span > 0 else 0.0
        iv = np.frombuffer(self._all, dtype=np.float32) if len(self._all) else np.zeros(1, np.float32)
        low1 = 1.0 / float(np.percentile(iv, 99)) if iv.size >= 20 else avg
        per = ", ".join(f"{STAGE_LABELS.get(k, k)} {1000.0 * self._stage_sum[k] / self._folded:.1f} ms"
                        for k in STAGE_ORDER if k in self._stage_sum and self._folded)
        return (f"Performance: {self._frames} frames in {span:.0f} s | avg {avg:.1f} fps | 1% low {low1:.1f} fps | "
                f"target {self.target_fps:.0f} fps | mean per frame: {per or 'n/a'}")


def fps_color(fps: float, good: float, warn: float) -> tuple[int, int, int]:
    """Green when ``fps`` >= ``good``, amber when >= ``warn``, red below that (BGR)."""
    return GREEN if fps >= good else AMBER if fps >= warn else RED


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color: tuple[int, int, int],
          thick: int = 1) -> None:
    """Anti-aliased text with a dark outline so it stays readable on any background."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _panel_size(mode: str, snap: PerfSnapshot, frame_w: int) -> tuple[int, int]:
    """Panel width and height in pixels for ``mode`` ('fps' or 'full'), shrunk to fit a narrow frame."""
    pw, ph = (290, 232 if snap.dropped else 214) if mode == "full" else (206, 62)
    return min(pw, max(8, frame_w - 4)), ph


def _paint_panel(canvas: np.ndarray, snap: PerfSnapshot, mode: str, good: float, warn: float) -> None:
    """Draw the panel's text, bars and graph onto ``canvas`` (origin top-left, background colour 20)."""
    ph, pw = canvas.shape[:2]
    full = mode == "full"
    col = fps_color(snap.fps, good, warn)
    _text(canvas, f"{snap.fps:5.1f} FPS", (10, 30), 0.9, col, 2)
    _text(canvas, f"/{1000.0 / snap.budget_ms:.0f}", (pw - 52, 30), 0.55, (200, 200, 200))
    over = snap.busy_ms > snap.budget_ms
    _text(canvas, f"{snap.frame_ms:4.1f} ms   busy {snap.busy_ms:4.1f} ms", (10, 52), 0.5, RED if over else (230, 230, 230))
    if not full:
        return
    _text(canvas, f"1% low {snap.low1_fps:4.1f} fps   {snap.headroom_pct:3.0f}% headroom", (10, 72), 0.5,
          fps_color(snap.low1_fps, good, warn))
    y = 92
    if snap.dropped:
        _text(canvas, f"stutters: {snap.dropped} slow frames recently", (10, y), 0.45, AMBER)
        y += 18
    bar_w = pw - 120
    for name in STAGE_ORDER:
        ms = snap.stage_ms.get(name)
        if ms is None:
            continue
        frac = float(np.clip(ms / snap.budget_ms, 0.0, 1.0))
        _text(canvas, f"{STAGE_LABELS[name]:<7}", (10, y + 11), 0.42, (210, 210, 210))
        cv2.rectangle(canvas, (70, y), (70 + bar_w, y + 12), (60, 60, 60), -1)
        cv2.rectangle(canvas, (70, y), (70 + int(bar_w * frac), y + 12), GREEN if frac < 0.5 else AMBER if frac < 0.8 else RED, -1)
        _text(canvas, f"{ms:4.1f}", (76 + bar_w, y + 11), 0.42, (210, 210, 210))
        y += 17
    gy, gh = y + 6, 38  # frame-time graph: one bar per recent frame, white line = the time budget
    cv2.rectangle(canvas, (10, gy), (pw - 10, gy + gh), (45, 45, 45), -1)
    for i, v in enumerate(snap.intervals_ms[-(pw - 20) // 3:]):
        bh = int(np.clip(v / (2.0 * snap.budget_ms), 0.0, 1.0) * gh)
        c = GREEN if v <= snap.budget_ms * 1.15 else RED
        cv2.rectangle(canvas, (10 + i * 3, gy + gh - bh), (12 + i * 3, gy + gh), c, -1)
    cv2.line(canvas, (10, gy + gh // 2), (pw - 10, gy + gh // 2), (255, 255, 255), 1)


class PerfOverlay:
    """Draws the performance panel in the top-right corner of the preview.

    The panel's text is rendered to a small cached bitmap and refreshed every ``refresh_s`` seconds (default 4 Hz):
    re-rendering ~30 anti-aliased text strings on every frame cost more than the rest of the preview, and digits that
    change 30 times a second are unreadable anyway. Each frame then only shades the panel region and copies the cached
    ink onto it with one masked copy.
    """

    def __init__(self, refresh_s: float = 0.25, clock: Callable[[], float] = time.perf_counter) -> None:
        """``refresh_s=0`` re-renders on every call (exact, slower)."""
        self.refresh_s, self._clock = refresh_s, clock
        self._key: tuple[str, int, int, bool] | None = None
        self._t = -1e9
        self._origin = (0, 0)
        self._crop: np.ndarray | None = None   # the part of the panel that is on screen
        self._mask: np.ndarray | None = None   # (h, w, 3) bool: pixels that carry text / bars
        self._dark: np.ndarray | None = None   # flat background used to shade the panel

    def _render(self, snap: PerfSnapshot, mode: str, good: float, warn: float, w: int, h: int) -> None:
        """Paint the panel bitmap and precompute everything the per-frame blit needs."""
        pw, ph = _panel_size(mode, snap, w)
        canvas = np.full((ph, pw, 3), 20, np.uint8)
        _paint_panel(canvas, snap, mode, good, warn)
        x0, y0 = max(2, w - pw - 10), 10
        ch, cw = max(0, min(h, y0 + ph) - y0), max(0, min(w, x0 + pw) - x0)
        self._origin = (x0, y0)
        if ch == 0 or cw == 0:
            self._crop = self._mask = self._dark = None
            return
        self._crop = np.ascontiguousarray(canvas[:ch, :cw])
        self._mask = np.repeat((self._crop != 20).any(axis=2)[:, :, None], 3, axis=2)
        self._dark = np.full_like(self._crop, 20)

    def draw(self, img: np.ndarray, snap: PerfSnapshot, mode: str = "fps", good: float = 27.0,
             warn: float = 20.0) -> None:
        """Draw onto ``img`` in place. ``mode``: "off" draws nothing, "fps" a compact readout, "full" adds budget use,
        per-stage bars and a frame-time graph. The fps number is green / amber / red against ``good`` / ``warn``."""
        if mode == "off" or snap.frames < 2:
            return
        h, w = img.shape[:2]
        key = (mode, w, h, bool(snap.dropped))
        now = self._clock()
        if key != self._key or now - self._t >= self.refresh_s:
            self._render(snap, mode, good, warn, w, h)
            self._key, self._t = key, now
        if self._crop is None or self._mask is None or self._dark is None:
            return
        x0, y0 = self._origin
        ch, cw = self._crop.shape[:2]
        roi = img[y0:y0 + ch, x0:x0 + cw]
        cv2.addWeighted(roi, 0.38, self._dark, 0.62, 0, dst=roi)   # shade just the panel
        np.copyto(roi, self._crop, where=self._mask)               # then the cached text / bars / graph


def draw_perf(img: np.ndarray, snap: PerfSnapshot, mode: str = "fps", good: float = 27.0,
              warn: float = 20.0) -> None:
    """One-shot, uncached version of :class:`PerfOverlay` (always shows the exact snapshot given)."""
    PerfOverlay(refresh_s=0.0).draw(img, snap, mode, good, warn)
