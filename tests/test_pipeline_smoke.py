#!/usr/bin/env python3
"""End-to-end smoke test of the REAL main() loop with the hardware replaced by fakes:
fake webcam (static frames), fake MediaPipe (one synthetic hand), fake virtual camera, no GUI window.
Checks the whole wiring: object photo -> cut-out -> gate -> render -> virtual camera, the FPS overlay on the
preview only (never in the stream), the 'p' key cycling modes, and the run summary printed on exit.
Run:  python tests/test_pipeline_smoke.py"""
import sys, io, time, tempfile, contextlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import handdoc_cam as H
import config

W, HH, FPS = 1280, 720, 40
tmp = Path(tempfile.mkdtemp())
fails = 0
def check(label, ok, extra=""):
    global fails
    fails += (not ok); print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  -> ' + str(extra)) if extra else ''}")

# ---- the fakes ------------------------------------------------------------------------------------------------
rng = np.random.default_rng(3)
BASE = np.clip(rng.normal(120, 25, (HH, W, 3)), 0, 255).astype(np.uint8)       # one static noisy "camera" frame

class FakeCap:
    n = 0
    def isOpened(self): return True
    def set(self, *a): return True
    def get(self, *a): return 0
    def read(self):
        time.sleep(1.0 / FPS); FakeCap.n += 1; return True, BASE.copy()
    def release(self): pass

def synthetic_hand(cx, cy, L=100.0):
    lm = np.zeros((21, 2), np.float32); lm[0] = [cx, cy]
    for k, o in {5: -.45, 9: -.15, 13: .15, 17: .45}.items():
        b = lm[0] + np.array([o * L * .9, -L]); lm[k] = b
        for j in range(1, 4): lm[k + j] = b + np.array([0, -L * .3 * j])
    lm[1] = lm[0] + np.array([-.3 * L, -.2 * L])
    for j in range(2, 5): lm[j] = lm[j - 1] + np.array([-.15 * L, -.2 * L])
    return lm

class FakeLandmarker:
    def detect_for_video(self, img, ts):
        lm = synthetic_hand(640, 640)
        return SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=float(x) / W, y=float(y) / HH) for x, y in lm]])
    def close(self): pass

class FakeMP:
    ImageFormat = SimpleNamespace(SRGB=1)
    @staticmethod
    def Image(**kw): return None

class FakeVCam:
    device = "fake-vcam"
    def __init__(self): self.frames, self.closed = [], False
    def send(self, f): self.frames.append(f.copy())
    def close(self): self.closed = True

vcam = FakeVCam(); views = []; waits = {"n": 0}
KEYS = {12: ord("p"), 24: ord("p"), 36: ord("p"), 50: ord("q")}      # fps -> full -> off -> fps -> quit
def fake_waitKey(_ms):
    waits["n"] += 1; return KEYS.get(waits["n"], 255)

# ---- a photo of a "phone" ----------------------------------------------------------------------------------------
photo = np.full((800, 800, 3), 205, np.uint8); photo += rng.integers(0, 6, photo.shape, dtype=np.uint8)
cv2.rectangle(photo, (310, 220), (490, 580), (35, 30, 30), -1); cv2.rectangle(photo, (322, 250), (478, 550), (160, 110, 60), -1)
cv2.imwrite(str(tmp / "phone.png"), photo)
(tmp / "cfg.json").write_text("{}")

# ---- patch + run -------------------------------------------------------------------------------------------------------
saved = {k: getattr(cv2, k) for k in ("imshow", "waitKey", "getWindowProperty", "namedWindow", "setMouseCallback", "destroyAllWindows")}
saved_h = (H.open_camera, H.make_landmarker, H.open_virtual_cam)
cv2.imshow = lambda name, img: views.append(img.copy()); cv2.waitKey = fake_waitKey
cv2.getWindowProperty = lambda *a: 1.0; cv2.namedWindow = lambda *a: None; cv2.setMouseCallback = lambda *a: None; cv2.destroyAllWindows = lambda: None
H.open_camera = lambda args: FakeCap(); H.make_landmarker = lambda args: (FakeMP(), FakeLandmarker()); H.open_virtual_cam = lambda args, w, h: vcam
out, err = io.StringIO(), io.StringIO()
t0 = time.time()
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        H.main([str(tmp / "phone.png"), "--config", str(tmp / "cfg.json"), "--object-size", "phone", "--no-classify",
                "--perf", "fps", "--fps", "30"])
    crashed = None
except BaseException as e:  # noqa: BLE001
    crashed = e
finally:
    for k, v in saved.items(): setattr(cv2, k, v)
    H.open_camera, H.make_landmarker, H.open_virtual_cam = saved_h
    config.CFG.reset()
wall = time.time() - t0
print(f"ran {len(views)} frames in {wall:.1f} s")
if crashed: print("   main() raised:", type(crashed).__name__, crashed); print(err.getvalue()[-600:])

print("\nloop behaviour:")
check("main() ran to the 'q' key and returned without raising", crashed is None and waits["n"] == 50, crashed)
check("every frame produced a preview AND a virtual-camera frame", len(views) == len(vcam.frames) >= 45, f"{len(views)} / {len(vcam.frames)}")
check("virtual camera was closed on exit", vcam.closed)
check("stream frames have the camera's size", all(f.shape == (HH, W, 3) for f in vcam.frames))
OBJ = (slice(300, 560), slice(560, 720))                  # around the synthetic palm (640, 640)
changed = [int((np.abs(f[OBJ].astype(int) - BASE[OBJ].astype(int)).sum(axis=2) > 30).sum()) for f in vcam.frames]
check("object appears in the stream once hands are tracked (fades in)", changed[-1] > 1500 and changed[0] < changed[-1], f"{changed[0]} -> {changed[-1]} px")

print("\nFPS overlay:")
PANEL_FPS = (slice(10, 72), slice(W - 216, W - 8)); PANEL_FULL = (slice(10, 230), slice(W - 300, W - 8))
diff = lambda i, reg: int((np.abs(views[i][reg].astype(int) - vcam.frames[i][reg].astype(int)).sum(axis=2) > 0).sum())
check("'fps' mode: panel drawn on the preview", all(diff(i, PANEL_FPS) > 800 for i in range(8, 11)), [diff(i, PANEL_FPS) for i in range(8, 11)])
check("'full' mode (after p): bigger panel", all(diff(i, PANEL_FULL) > 3 * diff(i, PANEL_FPS) for i in range(16, 21)), [diff(i, PANEL_FULL) for i in range(16, 19)])
check("'off' mode (after p again): preview == stream in the panel area", all(diff(i, PANEL_FULL) == 0 for i in range(27, 34)), [diff(i, PANEL_FULL) for i in range(27, 30)])
check("third p brings back the compact panel", all(diff(i, PANEL_FPS) > 800 for i in range(40, 48)))
check("overlay is NEVER in the virtual-camera stream", all(np.array_equal(f[PANEL_FULL], BASE[PANEL_FULL]) for f in vcam.frames))
import perf
cnt = lambda reg, col: int((np.abs(reg.astype(int) - np.array(col)).sum(axis=2) < 40).sum())
reg = views[20][PANEL_FPS]; g, a_, r = cnt(reg, perf.GREEN), cnt(reg, perf.AMBER), cnt(reg, perf.RED)
check("the FPS digits are coloured by the thresholds (fake camera ~40 fps -> not red)", (g + a_) > 100 and r == 0,
      f"green {g}px, amber {a_}px, red {r}px")

print("\nexit summary:")
o = out.getvalue()
check("run summary printed on exit", "Performance:" in o and "1% low" in o and "avg" in o, o.strip().splitlines()[-1][:110] if o.strip() else "(nothing)")
check("summary lists the five stages", all(w in o for w in ("hands", "gate", "render", "vcam", "preview")))
check("no stray errors on stderr", err.getvalue().strip() == "", err.getvalue()[:100])

print("\nALL PIPELINE TESTS PASSED" if not fails else f"\n{fails} FAILURE(S)"); sys.exit(1 if fails else 0)
