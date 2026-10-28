#!/usr/bin/env python3
"""Per-frame cost of everything this app does EXCEPT the camera and MediaPipe (hardware/model dependent), measured on
this machine, against the time budget of a 30 fps stream.   Run:  python tests/bench_pipeline.py [--size 1920x1080]"""
import sys, time, argparse
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import handdoc_cam as H, perf
from perf import FrameProfiler, PerfOverlay

ap = argparse.ArgumentParser(); ap.add_argument("--size", default="1280x720"); ap.add_argument("--fps", type=float, default=30.0)
a = ap.parse_args(); W, HH = map(int, a.size.lower().split("x")); budget = 1000.0 / a.fps
rng = np.random.default_rng(0)
frame = rng.integers(0, 255, (HH, W, 3), dtype=np.uint8)
cut = np.dstack([rng.integers(0, 255, (900, 600, 3), dtype=np.uint8), np.full((900, 600), 255, np.uint8)]); tex = H.Texture(cut)
def hand(cx, cy, L):
    lm = np.zeros((21, 2), np.float32); lm[0] = [cx, cy]
    for k, o in {5: -.45, 9: -.15, 13: .15, 17: .45}.items():
        b = lm[0] + np.array([o * L * .9, -L]); lm[k] = b
        for j in range(1, 4): lm[k + j] = b + np.array([0, -L * .3 * j])
    lm[1] = lm[0] + np.array([-.3 * L, -.2 * L])
    for j in range(2, 5): lm[j] = lm[j - 1] + np.array([-.15 * L, -.2 * L])
    return lm
L = HH / 7.2
ONE = [hand(W / 2, HH * .88, L)]; TWO = [hand(W * .25, HH * .88, L), hand(W * .75, HH * .88, L)]
class Item: texture = lambda self: tex; label = "phone"; size_cm = 13.0; size_src = "bench"; note = ""
import argparse as ap2
args = ap2.Namespace(max_ratio_one=None, max_ratio_two=None, max_reach_one=None, max_reach_two=None, max_one_hand_cm=None, max_two_hand_cm=None)
state = dict(palm_cm=10.0, span_cm=18.0, show=True, gate=True, scale=1.0)
item = Item(); quad = H.resolve_access(item, ONE, state, args)["target"]; assert quad is not None
quad2 = H.resolve_access(item, TWO, state, args)["target"]; assert quad2 is not None
roi2 = H.composite(frame.copy(), tex, quad2, 1.0); assert roi2 is not None   # the region the app actually touches
filt = H.OneEuro(); vec = quad.ravel().astype(np.float64)
prof = FrameProfiler(a.fps); ov = PerfOverlay(); t = [0.0]
for _ in range(120): prof.tick(); prof.mark("infer"); prof.mark("ui")
def ms(fn, n=200):
    fn(); t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1000
rows = [("frame.copy() (stream frame)", lambda: frame.copy()),
        ("access gate + placement (1 hand)", lambda: H.resolve_access(item, ONE, state, args)),
        ("access gate + placement (2 hands)", lambda: H.resolve_access(item, TWO, state, args)),
        ("smoothing filter (One-Euro)", lambda: filt(vec, time.perf_counter())),
        ("warp + shadow + blend (composite)", lambda: H.composite(frame.copy(), tex, quad, 1.0)),
        ("thumbs mask, whole frame (old way)", lambda: H.thumb_mask(TWO, frame.shape)),
        ("thumbs mask, object region only", lambda: H.thumb_mask(TWO, frame.shape, roi=roi2)),
        ("FPS overlay, cached, compact", lambda: ov.draw(frame, prof.snapshot(), "fps")),
        ("FPS overlay, cached, full", lambda: ov.draw(frame, prof.snapshot(), "full")),
        ("profiler: snapshot()", lambda: prof.snapshot()),
        ("profiler: tick() + 5 marks", lambda: (prof.tick(), [prof.mark(s) for s in perf.STAGE_ORDER]))]
print(f"{W}x{HH}, budget {budget:.1f} ms per frame at {a.fps:.0f} fps (camera + MediaPipe NOT included)\n")
res = {}
for name, fn in rows:
    res[name] = ms(fn); print(f"  {name:<38}{res[name]:7.3f} ms")
path = res["frame.copy() (stream frame)"] + res["access gate + placement (2 hands)"] + res["smoothing filter (One-Euro)"] + \
       res["warp + shadow + blend (composite)"] + res["thumbs mask, object region only"]
over = res["FPS overlay, cached, full"] + res["profiler: snapshot()"] + res["profiler: tick() + 5 marks"]
print(f"\n  render path per frame (2 hands):      {path:6.2f} ms  = {100 * path / budget:4.1f}% of the frame budget")
print(f"  profiler + full overlay per frame:    {over:6.2f} ms  = {100 * over / budget:4.1f}% of the frame budget")
print(f"  left for MediaPipe + camera + imshow: {budget - path - over:6.2f} ms")
