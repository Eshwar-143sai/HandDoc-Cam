#!/usr/bin/env python3
"""
HandDoc Cam - AR virtual camera that puts a real OBJECT from a photo into your hands.

Give it a photo of an object (phone, mug, laptop, car...). It cuts the object out, identifies it, looks up
its real-world size and checks it against YOUR hand span. If a person could hold it, it appears in your
hands on the live webcam stream (Zoom / Meet / Teams via a virtual camera); if not, it is blocked.

  webcam -> MediaPipe hand landmarks -> access gate (real size vs hand span) -> warp into hands -> virtual camera

Hotkeys (focus the preview window):
  o  open a photo           s  set the object's real size (cm or a name like 'phone')
  c  calibrate YOUR hand with a credit card (click its two ends)   k  type your palm length
  +/-  object scale         g  size gate on/off      t  thumbs in front on/off
  h  hide object            d  debug landmarks       m  mirror preview only (stream is never mirrored)
  r  reload handdoc_config.json (all tunable numbers: python config.py)
  p  performance overlay: FPS -> full (stage timings + frame-time graph) -> off
  q / ESC  quit (ESC cancels calibration)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import config
from config import CFG, ConfigError
from perf import FrameProfiler, PerfOverlay

from typing import Any, Callable, Sequence

Frame = np.ndarray       # (H, W, 3) uint8 BGR image
Quad = np.ndarray        # (4, 2) float32 corners ordered top-left, top-right, bottom-right, bottom-left
Landmarks = np.ndarray   # (21, 2) float32 MediaPipe hand landmarks, in pixels
Args = argparse.Namespace
State = dict[str, Any]   # mutable app state shared by the main loop and its helpers

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path.home() / ".handdoc_cam" / "hand_landmarker.task"
PALM_IDX = [0, 5, 9, 13, 17]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# All tunable numbers (gate ratios, hand-size defaults, placement offsets, smoothing, shadow, file limits)
# live in config.py as CFG.<name>; change them via handdoc_config.json, --set name=value, or `r` at runtime.


def load_profile(path: Path) -> tuple[float, float | None] | None:
    """-> (palm_cm, span_cm or None) or None"""
    try:
        d = json.loads(path.read_text())
        palm = float(d["palm_cm"])
        span = float(d["span_cm"]) if d.get("span_cm") else None
        if not CFG.palm_cm_min <= palm <= CFG.palm_cm_max:
            return None
        if span is not None and not CFG.span_cm_min <= span <= CFG.span_cm_max:
            span = None
        return palm, span
    except Exception:  # noqa: BLE001
        return None


def save_profile(path: Path, palm_cm: float, span_cm: float | None = None) -> None:
    """Write the calibrated palm length (and optional hand span, cm) to the profile JSON, creating folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"palm_cm": round(palm_cm, 2)}
    if span_cm:
        data["span_cm"] = round(span_cm, 2)
    path.write_text(json.dumps(data))


def _arg(args: Args, name: str, default: Any) -> Any:
    """Explicit command-line value if given, otherwise the (live) config value."""
    v = getattr(args, name, None)
    return default if v is None else v


def size_gate(long_cm: float, other_cm: float, mode: str, one_hand_cm: float, two_hand_cm: float,
              span_cm: float | None = None) -> tuple[bool, str]:
    """The threshold rule. Object 'width' = its LONG side (orientation in the photo does not matter).
    Fails when long_cm > limit, where the limits are normally ratio x hand span (1.8x one hand, 3.6x two).
    mode: 'palm' (one hand), 'two' (two hands) or 'none' (no hands yet -> most permissive limit)."""
    dims = f"{long_cm:.1f} x {other_cm:.1f} cm"
    rat = f" = {long_cm / span_cm:.2f}x your hand span" if span_cm else ""
    if mode == "palm":
        if long_cm <= one_hand_cm:
            return True, ""
        if long_cm <= two_hand_cm:
            return False, f"Too large for one hand ({dims}{rat}; limit {one_hand_cm:.0f} cm) - use two hands"
        return False, f"Object exceeds maximum hand-holding size ({dims}{rat}; limit {two_hand_cm:.0f} cm)"
    if long_cm <= two_hand_cm:
        return True, ""
    return False, f"Object exceeds maximum hand-holding size ({dims}{rat}; limit {two_hand_cm:.0f} cm)"


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
class Texture:
    """BGRA image plus a mip pyramid so heavy downscaling stays crisp."""

    def __init__(self, bgra: np.ndarray) -> None:
        """Build the mip pyramid (each level half the size of the one before) from a BGRA image."""
        self.levels = [bgra]
        while max(self.levels[-1].shape[:2]) > 96:
            self.levels.append(
                cv2.resize(self.levels[-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            )

    @property
    def aspect(self) -> float:
        """Width divided by height of the full-resolution image."""
        h, w = self.levels[0].shape[:2]
        return w / h

    def pick(self, need_px: float) -> np.ndarray:
        """Smallest pyramid level whose longer side is still >= ``need_px``: sharp without aliasing."""
        for lv in reversed(self.levels):  # smallest first
            if max(lv.shape[:2]) >= need_px:
                return lv
        return self.levels[0]


def _to_bgra(img: np.ndarray) -> np.ndarray:
    """Convert a gray / BGR / BGRA image to 8-bit BGRA and shrink it so its longer side is at most 2000 px."""
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1.0, float(img.max())))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    m = max(img.shape[:2])
    if m > 2000:
        s = 2000 / m
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------- #
# The input photo shows a real-world OBJECT (phone, mug, laptop, car...).
#   photo -> cut the object out -> identify it -> look up its REAL size -> gate on that size
# --------------------------------------------------------------------------- #
# keyword(s): typical LONG side of the real object in cm. Longest matching keyword wins.
# Matched against classifier labels (ImageNet names such as "cellular telephone", "sports car").
OBJECT_SIZE_TABLE = """
# --- pocket / desk
coin|penny|nickel|dime|button|bead|thimble|pill|capsule: 3
key|padlock|combination lock|safety pin|usb|flash drive|lighter|matchstick|stopwatch|digital watch|wristwatch|watch: 6
eraser|sharpener|lipstick|lip balm|wallet|billfold|credit card|business card|id card: 9
cellular telephone|cell phone|mobile phone|smartphone|phone|ipod|hand-held computer|pager|calculator|computer mouse|mouse: 13
remote control|remote|clicker|pen|ballpoint|fountain pen|marker|highlighter|toothbrush|comb|razor: 15
sunglass|sunglasses|spectacles|eyeglasses|glasses|monocle|scissors|stapler|pill bottle|lotion|sunscreen|perfume|spray bottle: 15
pencil|pencil case|pencil box|hairbrush|brush|hair spray: 19
ruler|slide rule|tape player|cassette: 30
# --- kitchen
coffee mug|mug|cup|teacup|espresso|measuring cup|beaker|goblet|wine glass|cocktail shaker|beer glass|glass: 11
water bottle|pop bottle|beer bottle|soda bottle|bottle|whiskey jug|jug: 24
wine bottle: 31
teapot|coffeepot|coffee maker|kettle|pitcher|water jug: 22
bowl|mixing bowl|soup bowl|plate|dish|bucket|pail|flowerpot: 22
spoon|ladle|spatula|whisk|fork|chopstick|knife|cleaver|letter opener|corkscrew|can opener|strainer|colander: 25
frying pan|skillet|wok|saucepan|pot|dutch oven|caldron|cauldron|chopping board: 35
toaster|waffle iron|blender|mixer|espresso maker|hair dryer: 30
microwave|oven|rotisserie: 50
dishwasher|washing machine|washer|stove: 85
refrigerator|icebox: 180
# --- food
banana|cucumber|zucchini|corn|carrot|artichoke|mushroom|broccoli|cauliflower|cabbage|lettuce|bell pepper|pepper: 18
orange|lemon|lime|apple|granny smith|tomato|peach|pomegranate|custard apple: 8
strawberry|cherry|grape|fig|olive|egg|raspberry|acorn: 4
pineapple|watermelon|cantaloupe|melon|pumpkin|squash|jackfruit|baguette|french loaf|bread|loaf: 30
hotdog|hot dog|hamburger|cheeseburger|burger|sandwich|burrito|taco|bagel|pretzel|doughnut|donut|cookie|muffin|cupcake|ice cream|ice lolly|popsicle: 14
pizza|cake|birthday cake|trifle: 30
# --- clothes / bags
shoe|sandal|loafer|clog|boot|running shoe|cowboy boot|sock|glove|mitten: 30
hat|cap|bonnet|sombrero|cowboy hat|shower cap|bathing cap|helmet|crash helmet|football helmet: 30
backpack|purse|handbag|pouch|bag|plastic bag|mailbag|lunchbox|briefcase|basket|shopping basket|hamper: 40
suitcase|trunk|luggage|punching bag: 70
jersey|t-shirt|sweatshirt|cardigan|jean|pajama|poncho|cloak|fur coat|trench coat|lab coat|suit|kimono|gown|robe|scarf|stole|bow tie|swimsuit: 70
# --- electronics / paper
laptop|notebook|netbook: 35
tablet|ipad|e-reader|kindle|book|comic book|binder|envelope|menu|notepad|diary|magazine|newspaper|folder|jigsaw puzzle|notebook paper: 25
keyboard|computer keyboard|typewriter keyboard|space bar: 45
camera|reflex camera|polaroid camera|camcorder|binoculars|microphone|joystick|loupe|lens cap: 15
printer|radio|boombox|cd player|loudspeaker|speaker|modem|router|projector|desktop computer|oscilloscope|amplifier|photocopier|scanner: 40
monitor|crt screen|screen|computer monitor: 55
television|television set|tv: 90
# --- home
lamp|table lamp|lampshade|vase|candle|bell|hourglass|clock|wall clock|analog clock|digital clock|alarm clock|globe|trophy: 30
mirror|picture frame|frame|poster|painting|picture: 45
toilet tissue|toilet paper|paper towel|tissue|soap|soap dispenser|soap bar: 12
pillow|cushion|towel|bath towel: 60
quilt|comforter|sleeping bag|blanket: 200
teddy|teddy bear|doll|puppet|toy: 28
rubik|cube|spinning top|top: 6
# --- tools / sport
hammer|screwdriver|wrench|plane|pliers|chisel|flashlight|torch|power drill|drill|saw|lantern: 28
basketball|soccer ball|volleyball|rugby ball|football|ball: 24
tennis ball|ping-pong ball|golf ball|baseball|croquet ball: 7
dumbbell|barbell: 35
racket|racquet|golf club|hockey stick|baseball bat|bat|cricket bat: 80
umbrella|parasol|broom|mop|shovel|spade|rake|axe|ax|hatchet|crutch|cane|walking stick|staff|paddle|oar|fishing rod|rod|pole|flagpole|broomstick: 90
ski|snowboard|surfboard|ladder|stepladder: 180
# --- instruments
harmonica|ocarina|maraca: 14
flute|oboe|bassoon|panpipe|pan flute: 60
violin|fiddle|trumpet|cornet|saxophone|sax|french horn|trombone|accordion|drum|banjo: 60
guitar|acoustic guitar|electric guitar: 100
cello|harp|piano|grand piano|upright piano|organ: 150
# --- furniture
stool|barstool: 50
chair|rocking chair|folding chair|barber chair|throne|park bench|bench: 90
desk|table|dining table|pool table|billiard table|altar|wardrobe|chiffonier|china cabinet|filing cabinet|file cabinet|bookcase|bookshelf|dresser|chest|cabinet|cupboard|entertainment center: 130
crib|cradle|bassinet: 130
sofa|studio couch|couch|loveseat|bed|four-poster|bathtub|tub: 200
# --- animals
bee|ladybug|ant|fly|beetle|cricket|grasshopper|mantis|cockroach|bug|weevil|cicada|dragonfly|butterfly|moth|monarch|lacewing: 5
spider|scorpion|tick|centipede|snail|slug|hermit crab: 8
finch|sparrow|robin|goldfinch|chickadee|bulbul|junco|jay|magpie|hummingbird|dipper: 15
frog|toad|newt|salamander|eft|bullfrog|axolotl|goldfish|gecko|chameleon|anole|lizard|iguana|agama: 15
hen|cock|rooster|chicken|duck|drake|partridge|grouse|quail|owl|hawk|vulture|parrot|macaw|cockatoo|toucan|penguin|bittern: 45
flamingo|crane|stork|heron|pelican|swan|albatross|goose|peacock|bald eagle: 90
ostrich: 200
cat|tabby|tiger cat|persian cat|siamese cat|egyptian cat|rabbit|hare|guinea pig|hamster|squirrel|marmot|beaver|porcupine|hedgehog|mink|weasel|otter|skunk|badger|meerkat|mongoose|armadillo|opossum: 42
chihuahua|pomeranian|yorkshire terrier|maltese|toy terrier|pug|shih-tzu|papillon|miniature pinscher|toy poodle|pekinese: 28
dog|terrier|spaniel|retriever|hound|poodle|collie|shepherd|bulldog|husky|malamute|corgi|dachshund|beagle|basset|pointer|setter|schnauzer|rottweiler|doberman|boxer|mastiff|great dane|newfoundland|samoyed|chow|akita|labrador|whippet|greyhound|wolfhound: 70
macaque|baboon|langur|marmoset|capuchin|monkey|gibbon|lemur|guenon|titi: 60
fox|wolf|coyote|jackal|hyena|dingo|lynx|cougar|puma|leopard|jaguar|cheetah|snake|cobra|python|boa|viper|rattlesnake: 110
lion|tiger|bear|gorilla|chimpanzee|orangutan|seal|sea lion|dolphin|crocodile|alligator|komodo dragon: 220
horse|zebra|ox|bison|water buffalo|ram|ibex|gazelle|llama|camel|dromedary|deer|elk|moose|hog|wild boar|pig|cow|bull|donkey|mule|shark|whale|killer whale: 260
elephant|hippopotamus|hippo|rhinoceros|rhino|giraffe: 350
turtle|tortoise|terrapin|crab|lobster|crayfish|starfish|jellyfish|conch: 30
# --- vehicles
bicycle|bike|mountain bike|tricycle|unicycle|moped|scooter|motor scooter: 170
motorcycle|motorbike|snowmobile|go-kart|golfcart: 210
car|sports car|cab|taxi|convertible|jeep|limousine|minivan|model t|racer|race car|beach wagon|station wagon|pickup|police van|minibus|ambulance|fire engine|garbage truck|tow truck|moving van|tractor|trailer truck|truck: 450
bus|school bus|trolleybus|streetcar|tram|trolley|freight car|passenger car|locomotive|bullet train|tank|forklift|harvester|canoe|speedboat|gondola|lifeboat|catamaran|boat: 800
airliner|warplane|airship|space shuttle|aircraft carrier|liner|container ship|submarine|ship|schooner|yawl|fireboat|pirate: 3000
# --- huge
wheelbarrow|barrow|shopping cart|cart|handcart|pushcart: 110
bell cote|lighthouse|bridge|dam|pier|dock|castle|palace|church|monastery|mosque|stupa|obelisk|triumphal arch|skyscraper|tower|barn|greenhouse|boathouse|house|mobile home|library|cinema|restaurant|bakery|barbershop|bookshop|butcher shop|grocery store|prison|stadium|coral reef|cliff|valley|volcano|alp|geyser|seashore|mountain|snowfield|lakeside|promontory|sandbar: 10000
"""
_SIZE_RULES: list[tuple[str, float, re.Pattern[str]]] | None = None


def _size_rules() -> list[tuple[str, float, re.Pattern[str]]]:
    """[(keyword, cm, compiled regex)] sorted so the most specific (longest) keyword wins.
    Add/override entries in ~/.handdoc_cam/object_sizes.txt using the same 'kw1|kw2: cm' format."""
    global _SIZE_RULES
    if _SIZE_RULES is None:
        text = OBJECT_SIZE_TABLE
        user = MODEL_PATH.parent / "object_sizes.txt"
        if user.exists():
            text += "\n" + user.read_text(errors="ignore")  # later lines win on ties
        rules: list[tuple[str, float, re.Pattern[str]]] = []
        for line in text.splitlines():
            line = line.split("#")[0].strip() if line.strip().startswith("#") else line.strip()
            if not line or ":" not in line:
                continue
            kws, cm_text = line.rsplit(":", 1)
            try:
                cm = float(cm_text)
            except ValueError:
                continue
            for kw in kws.split("|"):
                kw = kw.strip().lower()
                if kw:
                    rules.append((kw, cm, re.compile(r"(?<![a-z])" + re.escape(kw) + r"s?(?![a-z])")))
        rules.reverse()  # stable sort below keeps later (user) entries first among equals
        rules.sort(key=lambda r: -len(r[0]))
        _SIZE_RULES = rules
    return _SIZE_RULES


def lookup_object_size(label: str) -> tuple[float, str] | None:
    """'cellular telephone' -> (13.0, 'cellular telephone'); None when unknown."""
    lab = (label or "").lower()
    for kw, cm, rx in _size_rules():
        if rx.search(lab):
            return cm, kw
    return None


def parse_size_input(text: str) -> tuple[float, str] | None:
    """User typed '15', '15cm' or an object name ('laptop') -> (cm, source) or None."""
    t = (text or "").strip().lower().replace("cm", "").strip()
    if not t:
        return None
    try:
        v = float(t)
        return (v, "typed cm") if v > 0 else None
    except ValueError:
        hit = lookup_object_size(text)
        return (hit[0], f"typed '{hit[1]}'") if hit else None


# ---- cut the object out of its photo -------------------------------------------------------------
def _grabcut_alpha(bgr: np.ndarray) -> np.ndarray:
    """Foreground mask (0 or 255) from OpenCV GrabCut, seeded with a rectangle just inside the photo border."""
    h, w = bgr.shape[:2]
    k = CFG.grabcut_max_side / max(h, w) if max(h, w) > CFG.grabcut_max_side else 1.0
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else bgr
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    mg = CFG.grabcut_margin
    rect = (int(mg * sw), int(mg * sh), int((1 - 2 * mg) * sw), int((1 - 2 * mg) * sh))
    cv2.grabCut(small, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), CFG.grabcut_iters, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) if k < 1 else m


def _rembg_alpha(bgr: np.ndarray) -> np.ndarray | None:
    """Alpha channel from the optional ``rembg`` package, or None when it is not installed or fails."""
    try:  # optional: pip install rembg  (better cut-outs on busy backgrounds)
        from PIL import Image
        from rembg import remove

        out = remove(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return np.array(out)[:, :, 3]
    except Exception:  # noqa: BLE001
        return None


def extract_object(img: np.ndarray, method: str = "auto") -> tuple[np.ndarray, bool, str]:
    """-> (BGRA cut-out cropped tight to the object, ok, how).
    Order: real transparency in the file > rembg (if installed) > GrabCut. ok=False means we
    could not isolate an object and the whole photo is used."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1.0, float(img.max())))
    bgr = img[:, :, :3]
    alpha: np.ndarray | None = None
    how = "none"
    if img.shape[2] == 4 and (img[:, :, 3] < 250).mean() > 0.02 and (img[:, :, 3] > 5).mean() > 0.02:
        alpha, how = img[:, :, 3].copy(), "file alpha"
    elif method in ("auto", "rembg"):
        alpha = _rembg_alpha(bgr)
        how = "rembg" if alpha is not None else "none"
    if alpha is None and method in ("auto", "grabcut", "rembg"):
        alpha, how = _grabcut_alpha(bgr), "grabcut"
    ok = alpha is not None
    if ok and alpha is not None:
        m: np.ndarray = (alpha > 127).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            areas = st[1:, cv2.CC_STAT_AREA]
            keep = [i + 1 for i, a in enumerate(areas) if a >= 0.10 * areas.max()]
            m = np.isin(lab, keep).astype(np.uint8)
        frac = m.mean()
        if not 0.01 <= frac <= 0.985:
            ok = False
    if not ok:  # fall back to the whole photo
        return _to_bgra(bgr), False, how
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.erode(m, np.ones((3, 3), np.uint8))  # trim background fringe
    a = cv2.GaussianBlur((m * 255).astype(np.uint8), (0, 0), 1.2)
    ys, xs = np.where(a > 40)  # solid part of the mask; the soft edge may be trimmed a pixel
    pad = 1  # tiny: padding would inflate the width of thin objects and skew their real short side
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad + 1)
    cut = np.dstack([bgr, a])[y0:y1, x0:x1]
    return _to_bgra(cut), True, how


# ---- what is it, and how big really? --------------------------------------------------------------
CLASSIFIER_URL = ("https://storage.googleapis.com/mediapipe-models/image_classifier/"
                  "efficientnet_lite0/float32/1/efficientnet_lite0.tflite")
CLASSIFIER_PATH = MODEL_PATH.parent / "efficientnet_lite0.tflite"


class ObjectSizer:
    """Identify the object (offline EfficientNet-Lite0, ImageNet classes) and map the label to a real size.
    `classify_fn(bgr) -> [(label, score), ...]` can be injected (used by the tests)."""

    def __init__(self, min_conf: float = 0.10, enabled: bool = True,
                 classify_fn: Callable[[np.ndarray], list[tuple[str, float]]] | None = None) -> None:
        """Store the settings; the classifier model is only loaded on first use."""
        self.min_conf, self.enabled, self.fn = min_conf, enabled, classify_fn
        self.error: str = ""
        self._clf: Any = None   # mediapipe ImageClassifier, created lazily
        self._mp: Any = None    # the mediapipe module, imported lazily

    def _load(self) -> None:
        """Download (first run only) and open the EfficientNet-Lite0 image classifier."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not (CLASSIFIER_PATH.exists() and CLASSIFIER_PATH.stat().st_size > 1_000_000):
            CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading object classifier to {CLASSIFIER_PATH} ...")
            urllib.request.urlretrieve(CLASSIFIER_URL, CLASSIFIER_PATH.with_suffix(".part"))
            CLASSIFIER_PATH.with_suffix(".part").replace(CLASSIFIER_PATH)
        self._mp = mp
        self._clf = vision.ImageClassifier.create_from_options(vision.ImageClassifierOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(CLASSIFIER_PATH)), max_results=6))

    def top(self, bgr: np.ndarray) -> list[tuple[str, float]]:
        """The classifier's best (label, score) guesses for a BGR photo, highest score first."""
        if self.fn is not None:
            return self.fn(bgr)
        if self._clf is None:
            self._load()
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._clf.classify(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        return [(c.category_name, c.score) for c in res.classifications[0].categories]

    def estimate(self, bgr: np.ndarray) -> tuple[str | None, float, float | None]:
        """-> (label, score, cm or None). Takes the best-scoring label whose real size we know."""
        if not self.enabled and self.fn is None:
            return None, 0.0, None
        try:
            cands = self.top(bgr)
        except Exception as e:  # noqa: BLE001
            self.error = f"classifier unavailable ({type(e).__name__}: {e})"
            return None, 0.0, None
        for label, score in cands:
            if score < self.min_conf:
                break
            hit = lookup_object_size(label)
            if hit:
                return label, float(score), hit[0]
        return (cands[0][0], float(cands[0][1]), None) if cands else (None, 0.0, None)


class LoadError(Exception):
    """A problem with the user's file, worded so it can be shown on screen exactly as written."""


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def read_image_file(path: str | Path | None) -> np.ndarray:
    """Validate and decode a photo. Raises LoadError (never anything else) with a friendly reason for:
    empty path, missing file, folder, no permission, PDF/HEIC/other unsupported format, empty file,
    huge file, corrupted data, tiny image."""
    raw = str(path or "").strip().strip("\"'")  # tolerate quotes from drag-and-drop into a terminal
    if not raw:
        raise LoadError("No file chosen")
    p = Path(raw).expanduser()
    name = p.name or raw
    try:
        if not p.exists():
            raise LoadError(f"File not found: {raw}")
        if p.is_dir():
            raise LoadError(f"'{name}' is a folder, not an image file")
        size = p.stat().st_size
    except PermissionError:
        raise LoadError(f"No permission to read {name}") from None
    except OSError as e:
        raise LoadError(f"Cannot access {name}: {e.strerror or e}") from None
    ext = p.suffix.lower()
    if ext == ".pdf":
        raise LoadError("PDFs are not supported - give a photo of the object you want to hold (JPG/PNG)")
    if ext in (".heic", ".heif"):
        raise LoadError(f"{name}: HEIC photos are not supported - convert to JPG or PNG first")
    if size == 0:
        raise LoadError(f"{name} is empty (0 bytes)")
    if size > CFG.max_file_mb * 2**20:
        raise LoadError(f"{name} is too large ({size / 2**20:.0f} MB, limit {CFG.max_file_mb} MB)")
    try:
        data = np.fromfile(p, dtype=np.uint8)  # also works for non-ASCII paths on Windows
    except PermissionError:
        raise LoadError(f"No permission to read {name}") from None
    except (OSError, MemoryError) as e:
        raise LoadError(f"Could not read {name}: {getattr(e, 'strerror', None) or type(e).__name__}") from None
    try:
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except (cv2.error, MemoryError):
        img = None
    if img is None or img.size == 0 or img.ndim not in (2, 3):
        hint = "" if ext in IMAGE_EXTS else f" (format '{ext or 'none'}' is not supported - use PNG or JPG)"
        raise LoadError(f"{name} is not a readable image - the file may be corrupted{hint}")
    h, w = img.shape[:2]
    if min(h, w) < CFG.min_side_px:
        raise LoadError(f"{name} is too small to use ({w}x{h} px)")
    return img


def _as_bgr(img: np.ndarray) -> np.ndarray:
    """Any decoded image (gray, BGR, BGRA, 16-bit) -> 8-bit BGR."""
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1.0, float(img.max())))
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 2:
        raise LoadError("Unsupported image channel layout")
    return np.ascontiguousarray(img[:, :, :3])


class ObjectItem:
    """A photo of a real object, cut out, with its real-world size."""

    def __init__(self, path: str | Path, args: Args, sizer: ObjectSizer) -> None:
        """Load, cut out and size the photo at ``path``. Raises LoadError for any problem with the file."""
        img = read_image_file(path)
        self.path, self.name = str(path), Path(str(path).strip().strip("\"'")).name
        bgr = _as_bgr(img)
        try:
            cut, self.seg_ok, self.seg_how = extract_object(img, getattr(args, "seg", "auto"))
        except Exception as e:  # noqa: BLE001  (GrabCut/rembg trouble must not lose the photo)
            cut, self.seg_ok, self.seg_how = _to_bgra(bgr), False, f"failed: {type(e).__name__}"
        self._tex = Texture(cut)
        self.label, self.conf, self.size_cm, self.size_src, self.note = None, 0.0, None, "", ""
        if getattr(args, "object_size", None):
            if not self.apply_text(args.object_size):
                self.note = f"--object-size '{args.object_size}' not understood"
        if self.size_cm is None:
            self.label, self.conf, cm = sizer.estimate(bgr)   # never raises: errors land in sizer.error
            self.note = sizer.error or self.note
            if cm:
                self.size_cm, self.size_src = cm, f"auto {self.conf:.0%}"

    def apply_text(self, text: str) -> bool:
        """Set the real size from typed text: cm ('15') or an object name ('phone'). False if not understood."""
        r = parse_size_input(text)
        if not r:
            return False
        self.size_cm, self.size_src = r
        if r[1] != "typed cm":  # a typed NAME ('phone') labels the object; a bare number ('30') must not
            self.label = self.label or text.strip()
        return True

    def texture(self) -> Texture:
        """The cut-out as a mip-mapped texture, ready to warp."""
        return self._tex


def safe_load(path: Any, args: Args, sizer: ObjectSizer) -> tuple[ObjectItem | None, str]:
    """-> (ObjectItem | None, error_message). NEVER raises, so a bad file cannot take the video loop down."""
    try:
        return ObjectItem(path, args, sizer), ""
    except LoadError as e:
        return None, str(e)
    except MemoryError:
        return None, "Not enough memory to open that image - try a smaller photo"
    except Exception as e:  # noqa: BLE001
        return None, f"Unexpected error opening {Path(str(path)).name}: {type(e).__name__}: {e}"


def object_limits(short_cm: float, span_cm: float, args: Args) -> tuple[float, float, bool]:
    """Slender things (umbrella, broom: narrower than ~1/3 of your span) are gripped by their narrow
    side, so they get the longer 'reach' limits. Returns (one_hand_cm, two_hand_cm, slender)."""
    slender = short_cm <= CFG.slender_frac * span_cm
    r1 = _arg(args, "max_reach_one", CFG.reach_one) if slender else _arg(args, "max_ratio_one", CFG.ratio_one)
    r2 = _arg(args, "max_reach_two", CFG.reach_two) if slender else _arg(args, "max_ratio_two", CFG.ratio_two)
    return (_arg(args, "max_one_hand_cm", None) or r1 * span_cm, _arg(args, "max_two_hand_cm", None) or r2 * span_cm, slender)


def object_quad_two(hands: list[Landmarks], aspect: float, px_per_cm: float, long_cm: float,
                    scale: float) -> Quad:
    """True-scale object centred between two hands, tilted with the line joining them."""
    hs = sorted(hands[:2], key=lambda lm: palm_center(lm)[0])
    a, b = palm_center(hs[0]), palm_center(hs[1])
    r = (b - a) / (np.linalg.norm(b - a) + 1e-6)
    u = np.array([r[1], -r[0]], np.float32)
    long_px = long_cm * px_per_cm * scale
    W, Hh = (long_px, long_px / aspect) if aspect >= 1 else (long_px * aspect, long_px)
    c = (a + b) / 2 + u * (CFG.two_hand_lift * Hh)
    return np.array([c + u * Hh / 2 - r * W / 2, c + u * Hh / 2 + r * W / 2,
                     c - u * Hh / 2 + r * W / 2, c - u * Hh / 2 - r * W / 2], np.float32)


def _resolve_object(doc: ObjectItem, hands: list[Landmarks], state: State, args: Args,
                    r: dict[str, Any]) -> dict[str, Any]:
    """Gate and place one object: fills ``r`` with the verdict, banner text and (if allowed) the target quad."""
    aspect = doc.texture().aspect
    size_cm, src = doc.size_cm, doc.size_src
    label = (doc.label or "object").split(",")[0]
    if size_cm is None:  # we do not know what it is / how big it is -> do not guess silently
        why = doc.note or "I could not tell what this object is"
        r.update(pending=True, size_info=f"{label}: real size unknown",
                 msg=f"{why}. Press s and type its size in cm or a name (e.g. 15 or phone)")
        return r
    short_cm = size_cm * min(aspect, 1 / aspect)
    lim1, lim2, slender = object_limits(short_cm, state["span_cm"], args)
    r.update(doc_long=size_cm, doc_other=short_cm, lim1=lim1, lim2=lim2,
             size_info=f"{label} ~{size_cm:.0f}x{short_cm:.0f} cm ({src}{', slender' if slender else ''})")
    if not state["show"]:
        return r
    sep_ok = False
    if len(hands) >= 2:
        d = np.linalg.norm(palm_center(hands[0]) - palm_center(hands[1]))
        sep_ok = d > CFG.two_hand_min_sep * np.mean([palm_len(h) for h in hands[:2]])
    cand = "two" if sep_ok else ("palm" if hands else "none")
    ok, msg = (True, "") if not state["gate"] else size_gate(size_cm, short_cm, cand, lim1, lim2, state["span_cm"])
    if not ok:
        r.update(blocked=True, msg=msg.replace("Object exceeds", f"A {label} exceeds").replace(
            "Too large for one hand", f"A {label} is too large for one hand"))
    elif cand == "two":
        px_per_cm = max(palm_len(h) for h in hands[:2]) / state["palm_cm"]
        r.update(mode="two", target=object_quad_two(hands, aspect, px_per_cm, size_cm, state["scale"]))
    elif cand == "palm":
        big = max(hands, key=palm_len)
        px_per_cm = palm_len(big) / state["palm_cm"]
        h_px = size_cm * px_per_cm / (1.0 if aspect <= 1 else aspect)
        r.update(mode="palm", target=palm_quad(big, aspect, state["scale"], height_px=h_px))
    return r


def start_text_dialog(prompt: str) -> subprocess.Popen[str]:
    """Open a text-entry dialog in a separate process so the video keeps running."""
    code = (
        "import tkinter as tk\nfrom tkinter import simpledialog\n"
        "r=tk.Tk();r.withdraw();r.attributes('-topmost',True)\n"
        f"v=simpledialog.askstring('HandDoc Cam',{prompt!r})\n"
        "print('TEXT::'+(v or ''))\n"
    )
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)


def read_text(proc: subprocess.Popen[str]) -> str:
    """Text typed into a finished dialog ('' if it was cancelled)."""
    out = proc.stdout.read() if proc.stdout else ""
    for line in out.splitlines():
        if line.startswith("TEXT::"):
            return line[len("TEXT::"):].strip()
    return ""


def start_file_picker() -> subprocess.Popen:
    """Tk dialog in a separate process so the video stream never stalls."""
    code = (
        "import tkinter as tk\nfrom tkinter import filedialog\n"
        "r=tk.Tk();r.withdraw();r.attributes('-topmost',True)\n"
        "p=filedialog.askopenfilename(title='Choose a photo of an object',filetypes=["
        "('Photos','*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff'),('All files','*.*')])\n"
        "print('PICKED::'+p)\n"
    )
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)


def read_picker(proc: subprocess.Popen[str]) -> str | None:
    """-> path ('' if the user cancelled) or None if the dialog itself failed (e.g. no tkinter)."""
    try:
        out = proc.stdout.read() if proc.stdout else ""
    except (OSError, ValueError):
        return None
    for line in out.splitlines():
        if line.startswith("PICKED::"):
            return line[len("PICKED::"):].strip()
    return None


def start_float_dialog(prompt: str, initial: float) -> subprocess.Popen[str]:
    """Open a number-entry dialog (palm length) in a separate process so the video keeps running."""
    code = (
        "import tkinter as tk\nfrom tkinter import simpledialog\n"
        "r=tk.Tk();r.withdraw();r.attributes('-topmost',True)\n"
        f"v=simpledialog.askfloat('HandDoc Cam',{prompt!r},initialvalue={initial},minvalue={CFG.palm_cm_min},maxvalue={CFG.palm_cm_max})\n"
        "print('VALUE::'+('' if v is None else str(v)))\n"
    )
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)


def read_float(proc: subprocess.Popen[str]) -> float | None:
    """The number from a finished number dialog; None if it was cancelled or invalid."""
    out = proc.stdout.read() if proc.stdout else ""
    for line in out.splitlines():
        if line.startswith("VALUE::"):
            try:
                return float(line[len("VALUE::"):])
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------- #
# Hand geometry -> target quad
# --------------------------------------------------------------------------- #
def palm_center(lm: Landmarks) -> np.ndarray:
    """Centre of the palm in pixels: the mean of the wrist and the four finger-base landmarks."""
    return lm[PALM_IDX].mean(0)


def palm_len(lm: Landmarks) -> float:
    """Wrist to middle-finger-base distance in pixels: the live ruler that tracks distance to the camera."""
    return float(np.linalg.norm(lm[9] - lm[0])) + 1e-6


def hand_span(lm: np.ndarray) -> float:
    """Thumb tip (4) to pinky tip (20), pixels. Changes as you spread/close your fingers."""
    return float(np.hypot(*(lm[4] - lm[20])))


def palm_quad(lm: np.ndarray, aspect: float, scale: float, height_px: float | None = None) -> np.ndarray:
    """Card floating above one palm. Rotates with the hand, foreshortens when it turns."""
    L = palm_len(lm)
    u = (lm[9] - lm[0]) / L  # "up" along the hand
    r = np.array([-u[1], u[0]], np.float32)  # "right"
    foreshorten = float(np.clip(np.linalg.norm(lm[5] - lm[17]) / (CFG.foreshorten_ref * L),
                                CFG.foreshorten_min, CFG.foreshorten_max))
    H = (height_px if height_px else CFG.palm_default_height * L) * scale
    W = H * float(np.clip(aspect, 0.01, 100.0)) * foreshorten   # only guards against degenerate input
    c = palm_center(lm) + u * (CFG.palm_lift * H)
    return np.array(
        [c + u * H / 2 - r * W / 2, c + u * H / 2 + r * W / 2,
         c - u * H / 2 + r * W / 2, c - u * H / 2 - r * W / 2], np.float32)


class OneEuro:
    """Adaptive low-pass: heavy smoothing when still, low lag when moving fast."""

    def __init__(self, min_cutoff: float | None = None, beta: float | None = None,
                 d_cutoff: float | None = None) -> None:
        """Create the filter; explicit arguments override the config values."""
        self._fixed = (min_cutoff, beta, d_cutoff)  # explicit values win over the config
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None
        self.configure()

    def configure(self) -> None:
        """(Re)read smoothing settings from the config - called at start and after a live reload."""
        mc, beta, dc = self._fixed
        self.mc = CFG.smooth_min_cutoff if mc is None else mc
        self.beta = CFG.smooth_beta if beta is None else beta
        self.dc = CFG.smooth_d_cutoff if dc is None else dc

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Smoothing factor of a low-pass filter with cutoff ``cutoff`` (Hz) and time step ``dt`` (s)."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def reset(self) -> None:
        """Forget all history; the next sample passes through unchanged."""
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filter one sample (a vector of coordinates) taken at time ``t`` seconds; returns the smoothed vector."""
        if self.x is None or self.dx is None or self.t is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t
        self.t = t
        dx = (x - self.x) / max(dt, 1e-4)
        self.dx = self.dx + self._alpha(self.dc, dt) * (dx - self.dx)
        cutoff = self.mc + self.beta * float(np.linalg.norm(self.dx) / np.sqrt(len(x) / 2))
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        return self.x.copy()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def composite(frame: Frame, tex: Texture, quad: Quad, opacity: float,
              shadow: bool = True) -> tuple[int, int, int, int] | None:
    """Warp `tex` onto `quad` and alpha-blend into `frame` (in place).
    Returns the (x0, y0, x1, y1) region touched, or None if the quad was unusable."""
    H, W = frame.shape[:2]
    quad = quad.astype(np.float32)
    if not np.all(np.isfinite(quad)) or np.abs(quad).max() > 6 * max(W, H):
        return None
    if cv2.contourArea(quad) < 400 or not cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(np.int32)):
        return None
    pad = int(0.03 * max(W, H)) + 8 + int(max(abs(CFG.shadow_dx), abs(CFG.shadow_dy)) + 2 * CFG.shadow_blur)
    x0, y0 = max(0, int(quad[:, 0].min()) - pad), max(0, int(quad[:, 1].min()) - pad)
    x1, y1 = min(W, int(quad[:, 0].max()) + pad), min(H, int(quad[:, 1].max()) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    need = float(max(np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)))
    lv = tex.pick(need)
    th, tw = lv.shape[:2]
    src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad - np.array([x0, y0], np.float32))
    warped = cv2.warpPerspective(lv, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    if shadow and CFG.shadow_strength > 0:
        s = cv2.GaussianBlur(alpha, (0, 0), CFG.shadow_blur) if CFG.shadow_blur > 0 else alpha
        s = cv2.warpAffine(s, np.array([[1, 0, CFG.shadow_dx], [0, 1, CFG.shadow_dy]], np.float32), (s.shape[1], s.shape[0]))
        roi *= (1.0 - CFG.shadow_strength * opacity * s)[..., None]
    a = (alpha * opacity)[..., None]
    roi = roi * (1.0 - a) + warped[:, :, :3].astype(np.float32) * a
    frame[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return x0, y0, x1, y1


def thumb_mask(hands: list[Landmarks], shape: tuple[int, ...],
               roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Soft mask of the thumbs, pasted back on top so fingers 'grip' the object.

    With ``roi=(x0, y0, x1, y1)`` only that region is computed and returned (same pixels as slicing the full-frame
    mask, but without blurring the whole frame - the main-loop saving). The region is padded internally so the blur
    sees the same neighbours it would in the full frame.
    """
    h, w = shape[:2]
    rx0, ry0, rx1, ry1 = roi if roi is not None else (0, 0, w, h)
    pad = 10  # > the blur radius (8 px at sigma 2.5), so the cropped result is identical to the full-frame one
    cx0, cy0, cx1, cy1 = max(0, rx0 - pad), max(0, ry0 - pad), min(w, rx1 + pad), min(h, ry1 + pad)
    m = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
    off = np.array([cx0, cy0], np.int32)
    for lm in hands:
        r = max(3, int(palm_len(lm) * CFG.thumb_radius))
        pts = lm[[1, 2, 3, 4]].astype(np.int32) - off
        for a_, b_ in zip(pts[:-1], pts[1:]):
            cv2.line(m, (int(a_[0]), int(a_[1])), (int(b_[0]), int(b_[1])), 255, 2 * r)
        for q in pts[1:]:
            cv2.circle(m, (int(q[0]), int(q[1])), r, 255, -1)
    blurred = cv2.GaussianBlur(m, (0, 0), 2.5)
    return blurred[ry0 - cy0:ry1 - cy0, rx0 - cx0:rx1 - cx0]


def draw_calib(c: dict, mirror: bool, ref_cm: float) -> np.ndarray:
    """Frozen snapshot with the measured palm segment; user clicks the two ends of the card."""
    img = c["img"].copy()
    lm = c["lm"]
    cv2.line(img, tuple(lm[0].astype(int)), tuple(lm[9].astype(int)), (0, 255, 255), 3)
    cv2.line(img, tuple(lm[4].astype(int)), tuple(lm[20].astype(int)), (255, 0, 255), 3)
    for q in c["pts"]:
        cv2.circle(img, q, 6, (0, 255, 0), -1)
    if len(c["pts"]) == 2:
        cv2.line(img, c["pts"][0], c["pts"][1], (0, 255, 0), 2)
    if mirror:
        img = cv2.flip(img, 1)
    h = img.shape[0]
    lines = [f"CALIBRATION: click both ends of the card's LONG edge ({ref_cm:.2f} cm)  [{len(c['pts'])}/2]",
             "Yellow = palm (wrist to middle-finger base). Spread your fingers wide: thumb-to-pinky span is saved too.",
             "r = redo points    Esc = cancel"]
    for i, t in enumerate(lines):
        y = h - 70 + i * 24
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def resolve_access(doc: ObjectItem | None, hands: list[Landmarks], state: State,
                   args: Args) -> dict[str, Any]:
    """THE ACCESS GATE. Runs before any perspective warp.

        if object_size <= ratio x hand_span:  ALLOW -> return the target quad to warp onto
        else:                                 DENY  -> blocked=True + message, nothing is rendered
        size unknown:                         ASK   -> pending=True, nothing is rendered until the user types it

    Returns dict(mode, target, blocked, pending, msg, size_info, doc_long, doc_other, lim1, lim2)."""
    r = dict(mode="none", target=None, blocked=False, pending=False, msg="", size_info="",
             doc_long=0.0, doc_other=0.0)
    r["lim1"] = _arg(args, "max_one_hand_cm", None) or _arg(args, "max_ratio_one", CFG.ratio_one) * state["span_cm"]
    r["lim2"] = _arg(args, "max_two_hand_cm", None) or _arg(args, "max_ratio_two", CFG.ratio_two) * state["span_cm"]
    if doc is None:
        return r
    return _resolve_object(doc, hands, state, args, r)


def draw_banner(img: Frame, msg: str, title: str = "SIZE ERROR",
                color: tuple[int, int, int] = (30, 30, 200), row: int = 0) -> None:
    """Warning bar across the top (no emoji: OpenCV fonts can't render them). `row` stacks several bars."""
    h, w = img.shape[:2]
    msg = msg if len(msg) <= w // 9 else msg[: max(8, w // 9 - 3)] + "..."
    y0 = int(h * 0.10) + row * 70
    y1 = y0 + 64
    strip = img[y0:min(h, y1)]   # blend only the banner strip, not the whole frame
    if strip.size:
        cv2.addWeighted(strip, 0.15, np.full_like(strip, color), 0.85, 0, dst=strip)
    cv2.putText(img, title, (16, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, msg, (16, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_debug(img: Frame, hands: list[Landmarks]) -> None:
    """Draw hand skeletons, the palm ruler (yellow) and the thumb-pinky span (magenta) onto ``img``."""
    for lm in hands:
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(lm[a].astype(int)), tuple(lm[b].astype(int)), (0, 255, 0), 1)
        for p in lm:
            cv2.circle(img, tuple(p.astype(int)), 3, (0, 0, 255), -1)
        cv2.line(img, tuple(lm[0].astype(int)), tuple(lm[9].astype(int)), (0, 255, 255), 2)   # palm ruler
        cv2.line(img, tuple(lm[4].astype(int)), tuple(lm[20].astype(int)), (255, 0, 255), 2)  # span


# --------------------------------------------------------------------------- #
# Camera + model plumbing
# --------------------------------------------------------------------------- #
class Grabber(threading.Thread):
    """Always keeps only the newest frame -> low latency."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        """Wrap an open camera; call ``start()`` to begin grabbing."""
        super().__init__(daemon=True)
        self.cap, self.lock, self.running = cap, threading.Lock(), True
        self.frame: Frame | None = None
        self.seq: int = 0

    def run(self) -> None:
        """Thread body: keep reading frames, retaining only the newest."""
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.seq = f, self.seq + 1
            else:
                time.sleep(0.01)

    def read(self) -> tuple[int, Frame | None]:
        """(frame counter, copy of the newest frame) - the frame is None until the first one arrives."""
        with self.lock:
            return self.seq, (None if self.frame is None else self.frame.copy())


def open_camera(args: Args) -> cv2.VideoCapture:
    """Open the webcam at the requested size and rate (MJPG, one-frame buffer); exits with advice if it cannot."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera index {args.camera}. Try --camera 1 (or 2...). "
                 "Close other apps using the webcam.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def ensure_model() -> str:
    """Path of the MediaPipe hand-landmark model, downloaded on first run."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return str(MODEL_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {MODEL_PATH} ...")
    tmp = MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(MODEL_PATH)
    return str(MODEL_PATH)


def make_landmarker(args: Args) -> tuple[Any, Any]:
    """Create the MediaPipe HandLandmarker (video mode, two hands). Returns (mediapipe module, landmarker)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError:
        sys.exit("Missing dependency:  pip install mediapipe")
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.conf,
        min_hand_presence_confidence=args.conf,
        min_tracking_confidence=args.conf,
    )
    return mp, vision.HandLandmarker.create_from_options(opts)


def open_virtual_cam(args: Args, w: int, h: int) -> Any:
    """Start the virtual camera at ``w`` x ``h`` (None with --no-vcam); exits with install help if no driver is found."""
    if args.no_vcam:
        return None
    try:
        import pyvirtualcam
    except ImportError:
        sys.exit("Missing dependency:  pip install pyvirtualcam   (or run with --no-vcam)")
    try:
        cam = pyvirtualcam.Camera(width=w, height=h, fps=args.fps, fmt=pyvirtualcam.PixelFormat.BGR,
                                  backend=args.backend, device=args.device)
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"Could not start the virtual camera: {e}\n\n"
            "  Windows/macOS: install OBS Studio (it provides the virtual camera driver). If it is\n"
            "                 not listed yet, start 'Virtual Camera' in OBS once, then stop it so\n"
            "                 OBS isn't holding the device while this app runs.\n"
            "  Linux:         sudo apt install v4l2loopback-dkms && "
            "sudo modprobe v4l2loopback exclusive_caps=1\n"
            "  Or run with --no-vcam to just use the preview window.")
    print(f"Virtual camera running: {cam.device}  ({w}x{h} @ {args.fps} fps)")
    return cam


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (``argv`` defaults to sys.argv). Config-backed flags default to None so the live config applies."""
    p = argparse.ArgumentParser(description="AR virtual camera: hold a real object (from a photo) in your hands.",
                                epilog="All tunable numbers: python config.py  |  python handdoc_cam.py --list-config",
                                allow_abbrev=False)
    p.add_argument("doc", nargs="?", help="photo of the object to hold (omit to pick with a dialog)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--backend", default=None, help="pyvirtualcam backend (obs, v4l2loopback, ...)")
    p.add_argument("--device", default=None, help="virtual camera device name/path")
    p.add_argument("--no-vcam", action="store_true", help="preview only, no virtual camera")
    p.add_argument("--infer-width", type=int, default=640, help="width fed to the hand model")
    p.add_argument("--conf", type=float, default=0.5, help="hand detection/tracking confidence")
    p.add_argument("--opacity", type=float, default=1.0)
    p.add_argument("--palm-cm", type=float, default=None,
                   help="your wrist-to-middle-finger-base length in cm (overrides the saved profile)")
    p.add_argument("--profile", default="default", help="name of the saved hand profile (per person)")
    p.add_argument("--calib-ref-cm", type=float, default=None,
                   help="long edge of the reference object for 'c' calibration (default: credit card 8.56)")
    p.add_argument("--span-cm", type=float, default=None,
                   help="your open-hand thumb-tip to pinky-tip span in cm (default: calibrated, else 1.8 x palm)")
    p.add_argument("--max-ratio-one", type=float, default=None,
                   help="one hand: fail if object long side > RATIO x your hand span (default: config ratio_one)")
    p.add_argument("--max-ratio-two", type=float, default=None,
                   help="two hands: fail if object long side > RATIO x your hand span (default: config ratio_two)")
    p.add_argument("--max-one-hand-cm", type=float, default=None,
                   help="absolute one-hand limit in cm (overrides the ratio rule)")
    p.add_argument("--max-two-hand-cm", type=float, default=None,
                   help="absolute two-hand limit in cm (overrides the ratio rule)")
    p.add_argument("--seg", choices=["auto", "grabcut", "rembg", "none"], default="auto",
                   help="how to cut the object out of its photo (auto: file alpha > rembg if installed > grabcut)")
    p.add_argument("--object-size", default=None,
                   help="real size of the object in the photo: cm ('15') or a name ('phone', 'laptop')")
    p.add_argument("--no-classify", action="store_true", help="do not auto-identify the object (use --object-size / s)")
    p.add_argument("--classifier-conf", type=float, default=0.10, help="minimum classifier confidence (default 0.10)")
    p.add_argument("--max-reach-one", type=float, default=None,
                   help="slender objects (umbrella...): one-hand limit in hand spans (default: config reach_one)")
    p.add_argument("--max-reach-two", type=float, default=None,
                   help="slender objects: two-hand limit in hand spans (default: config reach_two)")
    config.add_cli_options(p)
    p.add_argument("--perf", choices=["off", "fps", "full"], default="fps",
                   help="live performance overlay on the preview: fps (default), full (stage timings + graph), off; "
                        "press p to cycle while running")
    p.add_argument("--no-gate", action="store_true", help="disable the size gate")
    p.add_argument("--banner-in-stream", action="store_true",
                   help="also draw the size-error banner into the virtual camera stream")
    p.add_argument("--mirror-preview", action="store_true", help="mirror the local preview only")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the app: load config, open camera and models, then loop: hands -> access gate -> render -> virtual camera + preview."""
    config.bootstrap(argv)   # config file -> --set -> CFG (may print/exit for --print-config / --list-config)
    args = parse_args(argv)
    mp, landmarker = make_landmarker(args)
    cap = open_camera(args)
    grab = Grabber(cap)
    grab.start()

    t0 = time.time()
    first = grab.read()[1]
    while first is None:
        if time.time() - t0 > 8:
            sys.exit("Camera opened but delivered no frames.")
        time.sleep(0.02)
        first = grab.read()[1]
    H, W = first.shape[:2]
    vcam = open_virtual_cam(args, W, H)

    sizer = ObjectSizer(min_conf=args.classifier_conf, enabled=not args.no_classify)
    doc = None
    picker = None

    filt = OneEuro()
    win = "HandDoc Cam - preview (q to quit)"
    state = dict(scale=1.0, thumbs=True, show=True, debug=False, mirror=args.mirror_preview,
                 gate=not args.no_gate, load_error=("", 0.0), render_err=False,
                 perf_mode=args.perf)

    def open_path(path: str) -> ObjectItem | None:
        """Load a photo; on ANY problem keep the current object, show the reason, keep the video running."""
        item, err = safe_load(path, args, sizer)
        if err:
            state["load_error"] = (err, time.monotonic() + 8.0)
            print("[file]", err, file=sys.stderr)
        else:
            state["load_error"], state["render_err"] = ("", 0.0), False
        return item

    if args.doc:
        doc = open_path(args.doc)
    if doc is None:  # nothing (valid) given on the command line -> ask
        picker = start_file_picker()
    gate_msg, size_info = "", ""
    prof_path = MODEL_PATH.parent / f"profile_{args.profile}.json"
    saved = load_profile(prof_path)
    state.update(palm_cm=CFG.default_palm_cm, span_cm=CFG.default_palm_cm * CFG.span_per_palm, palm_src="default",
                 calib=None, dialog=None, flash=("", 0.0))
    if saved:
        state.update(palm_cm=saved[0], span_cm=saved[1] or saved[0] * CFG.span_per_palm, palm_src="saved profile")
    if args.palm_cm:
        pc = float(np.clip(args.palm_cm, CFG.palm_cm_min, CFG.palm_cm_max))
        state.update(palm_cm=pc, span_cm=pc * CFG.span_per_palm, palm_src="--palm-cm")
    if args.span_cm:
        state.update(span_cm=float(np.clip(args.span_cm, CFG.span_cm_min, CFG.span_cm_max)))

    def set_palm(v: float, src: str, span: float | None = None) -> None:
        """Store a new palm length (and span) in the live state and save it to the profile."""
        state.update(palm_cm=float(v), span_cm=float(span) if span else float(v) * CFG.span_per_palm, palm_src=src)
        save_profile(prof_path, v, span)
        state["flash"] = (f"Hand saved: palm {v:.1f} cm, span {state['span_cm']:.1f} cm ({src})",
                          time.monotonic() + 4)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Calibration screen: collect the two clicks on the reference card (mapped back if the preview is mirrored)."""
        c = state["calib"]
        if c is None or event != cv2.EVENT_LBUTTONDOWN or len(c["pts"]) >= 2:
            return
        c["pts"].append((W - 1 - x if state["mirror"] else x, y))

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    mode, quad, vis = "none", None, 0.0
    last_seq, last_t, last_ts = -1, time.monotonic(), 0
    prof = FrameProfiler(target_fps=args.fps, history=CFG.perf_history)
    overlay = PerfOverlay()   # FPS panel for the local preview (never drawn into the stream)
    infer_scale = min(1.0, args.infer_width / W)

    try:
        while True:
            # file dialog finished?
            if picker is not None and picker.poll() is not None:
                path = read_picker(picker)
                picker = None
                if path is None:
                    state["load_error"] = ("File dialog unavailable (is tkinter installed?). "
                                           "Start with:  python handdoc_cam.py photo.jpg", time.monotonic() + 8.0)
                elif path:
                    new_item = open_path(path)
                    if new_item is not None:
                        doc = new_item

            if state["dialog"] is not None and state["dialog"].poll() is not None:
                v = read_float(state["dialog"])
                state["dialog"] = None
                if v and CFG.palm_cm_min <= v <= CFG.palm_cm_max:
                    set_palm(v, "typed")

            if state.get("tdialog") is not None and state["tdialog"].poll() is not None:
                txt = read_text(state["tdialog"])
                state["tdialog"] = None
                if txt and doc is not None:
                    if doc.apply_text(txt):
                        state["flash"] = (f"Object size set: ~{doc.size_cm:.0f} cm ({doc.size_src})", time.monotonic() + 4)
                    else:
                        state["flash"] = (f"Don't know '{txt}' - type a size in cm (e.g. 15)", time.monotonic() + 4)

            seq, frame = grab.read()
            if frame is None or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            now = time.monotonic()
            dt = max(now - last_t, 1e-4)
            last_t = now
            prof.tick()  # one tick per processed camera frame -> fps, 1% low, stage timings

            small = cv2.resize(frame, None, fx=infer_scale, fy=infer_scale) if infer_scale < 1 else frame
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            last_ts = max(last_ts + 1, int(now * 1000))
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), last_ts)
            hands = [np.array([[p.x * W, p.y * H] for p in lm], np.float32)
                     for lm in res.hand_landmarks]
            prof.mark("infer")

            # ---- access gate (size check BEFORE any warp) + target quad
            try:
                acc = resolve_access(doc, hands, state, args)
            except Exception as e:  # noqa: BLE001  (defensive: a bad object must not stop the stream)
                print(f"[object] dropped after error: {type(e).__name__}: {e}", file=sys.stderr)
                state["load_error"] = (f"Problem with this object ({type(e).__name__}); open another photo",
                                       now + 8.0)
                doc = None
                acc = resolve_access(None, hands, state, args)
            new_mode, target, blocked = acc["mode"], acc["target"], acc["blocked"]
            gate_msg, size_info, pending = acc["msg"], acc["size_info"], acc["pending"]
            doc_long, lim1, lim2 = acc["doc_long"], acc["lim1"], acc["lim2"]

            if new_mode != mode and new_mode != "none":
                filt.reset()
            mode = new_mode
            if target is not None:
                quad = filt(target.ravel().astype(np.float64), now).reshape(4, 2).astype(np.float32)
                vis = min(1.0, vis + dt * CFG.fade_in)
            else:
                vis = max(0.0, vis - dt * CFG.fade_out)
            prof.mark("gate")

            # ---- render the stream frame (never mirrored, no HUD)
            out = frame.copy()
            if doc is not None and quad is not None and vis > 0:
                try:
                    roi = composite(out, doc.texture(), quad, args.opacity * vis)
                except Exception as e:  # noqa: BLE001
                    roi = None
                    if not state["render_err"]:
                        state["render_err"] = True
                        print(f"[render] skipped frame(s): {type(e).__name__}: {e}", file=sys.stderr)
                if roi and mode == "two" and state["thumbs"] and hands:
                    x0, y0, x1, y1 = roi
                    m = (thumb_mask(hands, frame.shape, roi=(x0, y0, x1, y1)).astype(np.float32) / 255.0)[..., None]
                    out[y0:y1, x0:x1] = (out[y0:y1, x0:x1] * (1 - m) + frame[y0:y1, x0:x1] * m).astype(np.uint8)

            prof.mark("render")
            if blocked and args.banner_in_stream:
                draw_banner(out, gate_msg)
            elif pending and args.banner_in_stream:
                draw_banner(out, gate_msg, "SIZE UNKNOWN", (0, 140, 255))
            if vcam is not None:
                vcam.send(out)
            prof.mark("send")

            c = state["calib"]
            if c is not None and len(c["pts"]) == 2:
                ref_px = float(np.hypot(c["pts"][0][0] - c["pts"][1][0], c["pts"][0][1] - c["pts"][1][1]))
                if ref_px < 20:
                    state["flash"] = ("Points too close - click the two ends of the card", now + 3)
                    c["pts"].clear()
                else:
                    px_per_cm_c = ref_px / _arg(args, "calib_ref_cm", CFG.card_long_cm)
                    palm = c["palm_px"] / px_per_cm_c
                    span = c["span_px"] / px_per_cm_c
                    if CFG.palm_cm_min <= palm <= CFG.palm_cm_max and CFG.span_cm_min <= span <= CFG.span_cm_max:
                        set_palm(palm, "calibrated", span)
                        state["calib"] = None
                    elif CFG.palm_cm_min <= palm <= CFG.palm_cm_max:
                        state["flash"] = (f"Span {span:.1f} cm looks off - spread fingers wide and flat, retry", now + 4)
                        c["pts"].clear()
                    else:
                        state["flash"] = (f"{palm:.1f} cm looks wrong - hand flat, card beside palm, retry", now + 4)
                        c["pts"].clear()

            # ---- local preview with HUD
            view = out.copy()
            if state["debug"]:
                draw_debug(view, hands)
            if state["mirror"]:
                view = cv2.flip(view, 1)
            label = {"none": "NO HAND" if doc else "NO OBJECT", "palm": "IN HAND", "two": "BETWEEN HANDS"}[mode]
            if blocked:
                label = "BLOCKED"
            elif pending:
                label = "SIZE UNKNOWN"
            cv2.putText(view, f"{label}  {'-> ' + vcam.device if vcam else 'preview only'}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, f"{label}  {'-> ' + vcam.device if vcam else 'preview only'}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if blocked and not args.banner_in_stream:
                draw_banner(view, gate_msg)
            elif pending and not args.banner_in_stream:
                draw_banner(view, gate_msg, "SIZE UNKNOWN", (0, 140, 255))
            if state["load_error"][1] > now:
                draw_banner(view, state["load_error"][0], "FILE ERROR", (30, 30, 200),
                            row=1 if (blocked or pending) else 0)
            if size_info:
                cv2.putText(view, f"{size_info}  gate {'on' if state['gate'] else 'off'}  "
                                  f"limits {lim1:.0f}/{lim2:.0f} cm  ratio {doc_long / state['span_cm']:.2f}x span",
                            (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, f"{size_info}  gate {'on' if state['gate'] else 'off'}  "
                                  f"limits {lim1:.0f}/{lim2:.0f} cm  ratio {doc_long / state['span_cm']:.2f}x span",
                            (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            hud3 = (f"palm {state['palm_cm']:.1f} cm, span {state['span_cm']:.1f} cm ({state['palm_src']})"
                    + ("  - press c to calibrate YOUR hand" if state["palm_src"] == "default" else ""))
            cv2.putText(view, hud3, (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, hud3, (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            if state["flash"][1] > now:
                cv2.putText(view, state["flash"][0], (12, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, state["flash"][0], (12, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 1, cv2.LINE_AA)
            if hands:
                big = max(hands, key=palm_len)
                pp, sp = palm_len(big), hand_span(big)
                pxcm = pp / state["palm_cm"]
                hud4 = (f"live ruler: palm {pp:.0f}px -> {pxcm:.1f} px/cm | "
                        f"thumb-pinky {sp:.0f}px (~{sp / pxcm:.1f} cm, changes with finger spread)")
                cv2.putText(view, hud4, (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, hud4, (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(view, "o open photo  s object size  +/- scale  g gate  c calibrate  k palm  r config  p perf  t thumbs  h hide  d debug  m mirror  q quit",
                        (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if state["calib"] is not None:
                view = draw_calib(state["calib"], state["mirror"], _arg(args, "calib_ref_cm", CFG.card_long_cm))
            overlay.draw(view, prof.snapshot(), state["perf_mode"], CFG.perf_good_fps, CFG.perf_warn_fps)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            prof.mark("ui")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if state["calib"] is not None:  # calibration screen owns the keyboard
                if key in (27, ord("q")):
                    state["calib"] = None
                elif key == ord("r"):
                    state["calib"]["pts"].clear()
                continue
            if key in (ord("q"), 27):
                break
            if key == ord("o") and picker is None:
                picker = start_file_picker()
            elif key in (ord("+"), ord("=")):
                state["scale"] = min(CFG.scale_max, state["scale"] * CFG.scale_step)
            elif key in (ord("-"), ord("_")):
                state["scale"] = max(CFG.scale_min, state["scale"] / CFG.scale_step)
            elif key == ord("p"):  # performance overlay: fps -> full -> off
                state["perf_mode"] = {"off": "fps", "fps": "full", "full": "off"}[state["perf_mode"]]
            elif key == ord("r"):  # reload handdoc_config.json without restarting
                try:
                    changed = config.reload()
                    filt.configure()
                    state["flash"] = ((f"Config reloaded: {', '.join(changed[:4])}{'...' if len(changed) > 4 else ''}"
                                       if changed else "Config reloaded (no changes)"), now + 4)
                except ConfigError as e:
                    why = str(e).strip().lstrip("- ").splitlines()[0] if str(e).strip() else "invalid config"
                    print("[config] reload failed, keeping current values:", e, file=sys.stderr)
                    state["flash"] = (f"Config NOT reloaded: {why}", now + 8)
            elif key == ord("t"):
                state["thumbs"] = not state["thumbs"]
            elif key == ord("h"):
                state["show"] = not state["show"]
            elif key == ord("d"):
                state["debug"] = not state["debug"]
            elif key == ord("m"):
                state["mirror"] = not state["mirror"]
            elif key == ord("g"):
                state["gate"] = not state["gate"]
            elif key == ord("c"):
                if hands:
                    big = max(hands, key=palm_len)
                    state["calib"] = dict(img=frame.copy(), lm=big.copy(), palm_px=palm_len(big), span_px=hand_span(big), pts=[])
                else:
                    state["flash"] = ("Show your hand (flat, palm to camera) first, then press c", now + 3)
            elif key == ord("s") and doc is not None and state.get("tdialog") is None:
                state["tdialog"] = start_text_dialog(
                    "Real size of this object: type cm (e.g. 15) or a name (e.g. phone, laptop, mug)")
            elif key == ord("k") and state["dialog"] is None:
                state["dialog"] = start_float_dialog(
                    "Palm length in cm: wrist crease to the knuckle at the base of the middle finger",
                    round(state["palm_cm"], 1))
    finally:
        print(prof.summary())
        grab.running = False
        cap.release()
        if vcam is not None:
            vcam.close()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
