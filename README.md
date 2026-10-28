# HandDoc Cam - hold a real object (from a photo) in your hands on a video call

You give it a **photo of an object** (phone, mug, laptop, a car...). The app cuts the object out of the photo,
works out **what it is and how big it really is**, and checks whether a person could hold it in their hands.
If yes, the object appears in your hands on your live webcam, through a virtual camera (Zoom / Meet / Teams).
If not, it is **not shown** and a red "SIZE ERROR" banner appears.

    photo -> cut-out -> identify -> real size (cm) -> compare with YOUR hand span -> allow / deny -> warp into hands

Only photos of objects are supported (JPG, PNG, BMP, WebP, TIFF). PDFs and documents are not.

## Setup
1. Python 3.10-3.12.  `pip install -r requirements.txt`
2. A virtual camera driver:
   - **Windows / macOS:** install OBS Studio. If "OBS Virtual Camera" is not selectable in your call app,
     start Virtual Camera in OBS once, then stop it (OBS must not hold the device while this app runs).
   - **Linux:** `sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback exclusive_caps=1`
3. First run downloads two small models (hand landmarks, object classifier) to `~/.handdoc_cam/`.

## Run
    python handdoc_cam.py                              # file picker opens
    python handdoc_cam.py phone.jpg                    # photo of an object
    python handdoc_cam.py photo.jpg --object-size laptop   # skip auto-detect; a name or cm ('35')
    python handdoc_cam.py --no-vcam                    # preview window only (no driver needed)

Pick **OBS Virtual Camera** (or your v4l2loopback device) as the camera in your call app.
**Do this once:** hold your hand flat to the camera with a credit card beside it and press `c` (see Calibration).

## Controls (click the preview window first)
`o` open photo | `s` type the object's real size (cm or a name like `phone`) | `c` calibrate hand | `k` type palm length |
`+`/`-` scale | `p` performance overlay | `r` reload config | `g` size gate on/off | `t` thumbs in front | `h` hide | `d` debug | `m` mirror preview | `q` quit (Esc cancels calibration)

Two hands apart -> the object is shown **between** them. One hand -> it sits **in** your palm. Both at true scale:
bring your hand closer and the object grows, like a real object would. The stream is never mirrored.

## How the object's real size is found
1. **Cut-out:** transparency in the file (PNG) > `rembg` if installed (`--seg rembg`) > OpenCV GrabCut (default; best on a fairly
   plain background with the object roughly centred). If no object can be isolated, the whole photo is used.
2. **Identify:** offline EfficientNet-Lite0 (ImageNet's 1000 everyday classes) -> a label such as "cellular telephone".
3. **Look up the size:** a built-in table maps ~800 label keywords to a typical long side in cm (phone 13, mug 11, laptop 35,
   bicycle 170, car 450...). Add or override entries in `~/.handdoc_cam/object_sizes.txt` (`keyword|keyword: cm`).
4. **Not sure?** If the label is unknown or the classifier is not confident (`--classifier-conf`), the app does **not guess**: an amber
   "SIZE UNKNOWN" banner appears, nothing is drawn, and you press `s` to type a size or name. `--no-classify` skips step 2;
   `--object-size` overrides whatever was detected.

## The access gate
`ratio = object size (cm) / your hand span (cm)` - your span is the open-hand thumb-tip to pinky-tip distance saved by calibration.

| object kind | one hand fails if | two hands fail if |
|---|---|---|
| compact (most things) | long side > **1.8 x** span | long side > **3.6 x** span |
| slender (narrower than 0.35 x span: umbrella, broom, pen) | length > 5 x span | length > 8 x span |

Slender things are gripped by their narrow side, so they get the longer limits. Tune with `--max-ratio-one/two`,
`--max-reach-one/two`, or absolute `--max-one-hand-cm/--max-two-hand-cm`. With an 18 cm span: a phone or mug passes in one hand,
a laptop (1.94x) needs two hands, a car (25x) or an elephant is blocked.

## Live FPS counter and profiler
The local preview shows a live frame-rate panel in the top-right corner (never in the stream your call sees). Press **`p`** to cycle
**FPS -> full -> off** (or start with `--perf full|off`).

* **FPS** (green >= 27, amber >= 20, red below; thresholds `perf_good_fps` / `perf_warn_fps`), ms per frame and the processing time per frame.
* **Full** adds **1% low** (the fps of the slowest 1% of recent frames: it exposes stutter an average hides), **headroom** (how much of the
  33 ms frame budget is still unused - the number that shows an optimisation helped even when the camera caps you at 30 fps),
  a count of slow frames, per-stage bars (hands / gate / render / vcam / preview) and a frame-time graph.
* On exit it prints a summary: `Performance: 5400 frames in 180 s | avg 29.9 fps | 1% low 27.8 fps | mean per frame: hands 8.1 ms, ...`
* The panel is cached and refreshed at 4 Hz, so it costs ~0.02-0.1 ms per frame. Measure your own machine (no camera needed):
  `python tests/bench_pipeline.py [--size 1920x1080]`. Example (a laptop-class CPU, 720p): gate + placement 0.1 ms, warp/shadow/blend 4 ms,
  thumbs mask 0.5 ms (was 2.1 ms before it was restricted to the object's region), profiler + full overlay 0.3 ms; MediaPipe and the camera
  are not included and depend on your hardware - the panel's **hands** bar shows them live.

## Configuration: tune every number without editing code
All sizing, threshold, placement, smoothing, shadow and file-limit numbers (40 of them) live in `config.py`, each with a default,
a valid range and a one-line meaning. Change them three ways (later wins):

    python config.py                                  # list every setting: default, range, meaning
    python config.py --write handdoc_config.json      # start a config file (any subset of keys works)
    python handdoc_cam.py --set ratio_one=2.2 --set palm_lift=0.1     # one-off override, repeatable
    python handdoc_cam.py --print-config              # the values currently in effect

* **Config file:** `--config my.json`, else `./handdoc_config.json`, else `~/.handdoc_cam/config.json` (first one found). Example:
  `handdoc_config.example.json`. Keys starting with `_` are comments.
* **Live tuning:** edit the JSON file while the app runs, click the preview, press **`r`**. Gate, placement, shadow, fade, smoothing and
  scale settings apply immediately. A bad edit (broken JSON, unknown name, out-of-range or contradictory value) is rejected with a message,
  and the running values stay as they were.
* **Precedence:** built-in defaults < config file < `--set` < dedicated flags (`--max-ratio-one`, `--palm-cm`, `--span-cm`...).
* Typos get a hint (`ratio_onee` -> "did you mean ratio_one?"). Validation covers type, range, NaN/inf and combinations such as
  `ratio_one <= ratio_two`.

Most useful knobs: `ratio_one` / `ratio_two` (how big is "holdable"), `palm_lift` / `two_hand_lift` (where the object sits),
`foreshorten_ref` (how fast it narrows when you turn your hand), `smooth_min_cutoff` / `smooth_beta` (steadiness vs lag),
`shadow_*`, `thumb_radius`, `grabcut_*` (cut-out detail vs speed).

## Calibration (hands differ)
Press `c`: hold your hand open and flat, palm to camera, with a credit/ID card (85.6 mm long edge) flat beside your palm at the same
distance. The view freezes; click both ends of the card's long edge. Palm length (yellow) and span (magenta) are measured against
the card and saved to `~/.handdoc_cam/profile_<name>.json` (`--profile alice` for several people). Spread your fingers wide.
Alternatives: `k` or `--palm-cm 9.2` (span then estimated as 1.8 x palm), `--span-cm`. Uncalibrated it assumes a 10 cm palm / 18 cm span.

## Bad files never crash the stream
Opening a photo goes through `safe_load()`, which cannot raise. Missing path, folder, empty or corrupted file, PDF, HEIC, unsupported
format, tiny image, file over 100 MB, no read permission, a crashing cut-out or classifier: each shows a red **FILE ERROR** banner with
the reason, **keeps your previous object**, and the video continues. A failing file dialog (no tkinter) tells you to pass the photo on the
command line. A wrong extension is fine if the contents are a valid image. If rendering an object ever fails, it is dropped with a message
instead of stopping the stream.

## Tests (no webcam needed)
    python tests/test_object_gate.py     # synthetic photos: phone, mug, laptop, umbrella, car, elephant, unknown -> pass/block (+ tests/object_demo.png)
    python tests/test_file_loading.py    # ~40 bad/odd file cases must give a message, never an exception
    python tests/test_config.py          # defaults == old behaviour, validation, reload, and each setting's effect
    python tests/test_perf.py            # FPS / 1% low / headroom maths on a fake clock + overlay drawing
    python tests/test_pipeline_smoke.py  # the REAL main() loop with a fake camera, fake hand tracker and fake virtual camera
    python tests/test_code_quality.py    # every function typed + documented, mypy + pyflakes clean
    python tests/bench_pipeline.py       # per-frame cost of each step vs the 33 ms budget (informational)

The object test replaces only the classifier model with a stub; cut-out, size table, gate and rendering are the real code.

## Code layout
| file | role |
|---|---|
| `handdoc_cam.py` | the app: camera loop, object cut-out and size lookup, access gate, hand-anchored rendering, virtual camera |
| `config.py` | every tunable number (40), validated, from JSON / `--set` / live reload |
| `perf.py` | frame profiler (`FrameProfiler`) and the cached FPS overlay (`PerfOverlay`) |
| `tests/` | gate, file-loading, config, profiler, end-to-end smoke and code-quality tests |

All functions and classes have type hints and docstrings (enforced by `tests/test_code_quality.py`); `mypy` is clean.

## Limits worth knowing
* Recognition is limited to what ImageNet knows and the size table covers common objects; anything else asks you. Typical sizes are
  averages (a "bottle" may be 15 or 35 cm): use `s` when you know better.
* The gate is about **size**, not weight (an anvil can pass). It assumes the object is at the same depth as your hand.
* The object is a flat cut-out warped in 2D, not a 3D model: it looks right facing the camera, not turned side-on.
* Cut-out quality depends on the photo: plain background and good contrast work best; try `--seg rembg` for busy scenes.
* Only thumbs are drawn in front of the object in two-hand mode.

## Troubleshooting
Jittery/laggy: `--infer-width 480` or 720p. Hand not found: more light, whole hand in frame, `--conf 0.4`. Wrong camera: `--camera 1`.
Classifier download blocked: use `--no-classify --object-size ...`. Object cut badly: `--seg rembg`, `--seg none`, or crop the photo.
