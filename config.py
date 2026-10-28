#!/usr/bin/env python3
"""
Every tunable number of HandDoc Cam lives here: name, default, valid range and what it does.

Change them without touching the main script, in any of three ways (later wins):

    1. a JSON file       ./handdoc_config.json  or  ~/.handdoc_cam/config.json  or  --config my.json
    2. the command line  --set ratio_one=2.2 --set palm_lift=0.1      (repeatable)
    3. while running     edit the JSON file, click the preview window, press  r  to reload

    python config.py                     # table of all settings (default, range, meaning)
    python config.py --write my.json     # start a config file from the defaults
    python handdoc_cam.py --print-config # the values currently in effect

Unknown names, wrong types, out-of-range values and contradictory combinations are rejected with a
message that names the problem; a rejected reload leaves the running values untouched.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

USER_DIR = Path.home() / ".handdoc_cam"
CONFIG_FILENAME = "handdoc_config.json"


class ConfigError(ValueError):
    """Raised for any invalid config; the message is written to be shown to a person as-is."""


@dataclass(frozen=True)
class Param:
    """One tunable setting: its default, the valid range and what it controls."""
    name: str
    default: float
    lo: float
    hi: float
    group: str
    help: str
    kind: str = "float"  # "float" or "int"


PARAMS = [
    # ---- the access gate: how big a thing may be, in multiples of YOUR hand span ----------------------
    Param("ratio_one", 1.8, 0.5, 6.0, "gate", "one hand: block objects whose long side is above this x your hand span"),
    Param("ratio_two", 3.6, 1.0, 12.0, "gate", "two hands: same rule, for objects held between both hands"),
    Param("reach_one", 5.0, 1.0, 15.0, "gate", "one hand, SLENDER objects (umbrella, broom): max length in hand spans"),
    Param("reach_two", 8.0, 2.0, 25.0, "gate", "two hands, slender objects: max length in hand spans"),
    Param("slender_frac", 0.35, 0.05, 1.0, "gate", "an object is 'slender' if its short side <= this x your hand span"),
    Param("two_hand_min_sep", 1.6, 0.5, 4.0, "gate", "hands count as a two-hand hold when apart by more than this many palm lengths"),
    # ---- hand size & calibration ---------------------------------------------------------------------
    Param("default_palm_cm", 10.0, 4.0, 16.0, "hand", "palm length (wrist to middle-finger knuckle) assumed until you calibrate"),
    Param("span_per_palm", 1.8, 1.4, 2.4, "hand", "open-hand thumb-to-pinky span / palm length; estimates your span if only palm is known"),
    Param("card_long_cm", 8.56, 3.0, 30.0, "hand", "long edge of the calibration reference (credit card = 8.56 cm)"),
    Param("palm_cm_min", 4.0, 2.0, 10.0, "hand", "calibration accepts palm lengths from this (cm)"),
    Param("palm_cm_max", 16.0, 10.0, 25.0, "hand", "... up to this (cm)"),
    Param("span_cm_min", 12.0, 6.0, 20.0, "hand", "calibration accepts hand spans from this (cm)"),
    Param("span_cm_max", 26.0, 18.0, 40.0, "hand", "... up to this (cm)"),
    # ---- where the object is drawn ---------------------------------------------------------------------
    Param("palm_lift", 0.35, -0.5, 1.0, "placement", "one hand: raise the object's centre above the palm centre, in object heights"),
    Param("palm_default_height", 3.4, 0.5, 10.0, "placement", "one hand: object height in palm lengths when its real size is unknown"),
    Param("foreshorten_ref", 0.85, 0.4, 1.5, "placement", "palm width / palm length of a flat hand; sets how fast the object narrows as you turn"),
    Param("foreshorten_min", 0.30, 0.05, 1.0, "placement", "the object never narrows below this fraction of its width"),
    Param("foreshorten_max", 1.15, 0.6, 2.0, "placement", "... or widens beyond this fraction"),
    Param("two_hand_lift", 0.0, -1.0, 1.0, "placement", "two hands: raise the object above the midpoint between your hands, in object heights"),
    Param("thumb_radius", 0.22, 0.05, 0.8, "placement", "thumbs drawn in front of the object: radius in palm lengths"),
    Param("scale_step", 1.08, 1.01, 1.5, "placement", "each + / - key press multiplies the object scale by this"),
    Param("scale_min", 0.3, 0.05, 1.0, "placement", "smallest scale reachable with the - key"),
    Param("scale_max", 4.0, 1.0, 20.0, "placement", "largest scale reachable with the + key"),
    # ---- smoothing & look ------------------------------------------------------------------------------
    Param("smooth_min_cutoff", 1.5, 0.05, 30.0, "look", "jitter smoothing when still (lower = steadier, laggier); One-Euro min cutoff, Hz"),
    Param("smooth_beta", 0.03, 0.0, 2.0, "look", "how fast smoothing relaxes when you move (higher = less lag in motion)"),
    Param("smooth_d_cutoff", 1.0, 0.1, 20.0, "look", "One-Euro derivative cutoff, Hz"),
    Param("fade_in", 8.0, 0.5, 100.0, "look", "object fades in at this rate (1/s) when a hand appears"),
    Param("fade_out", 5.0, 0.5, 100.0, "look", "object fades out at this rate (1/s) when hands disappear"),
    Param("shadow_strength", 0.35, 0.0, 1.0, "look", "drop-shadow darkness (0 = none)"),
    Param("shadow_blur", 6.0, 0.0, 40.0, "look", "drop-shadow softness, px"),
    Param("shadow_dx", 5.0, -60.0, 60.0, "look", "drop-shadow offset right, px"),
    Param("shadow_dy", 8.0, -60.0, 60.0, "look", "drop-shadow offset down, px"),
    # ---- performance overlay ----------------------------------------------------------------------------
    Param("perf_good_fps", 27.0, 5.0, 240.0, "perf", "overlay shows the fps in green at or above this (a '30 fps' webcam jitters around 29-30)"),
    Param("perf_warn_fps", 20.0, 1.0, 240.0, "perf", "... amber at or above this, red below"),
    Param("perf_history", 300, 30, 3000, "perf", "recent frames used for the 1% low and the stutter count", "int"),
    # ---- cut-out & files -------------------------------------------------------------------------------
    Param("grabcut_max_side", 480, 160, 2000, "files", "GrabCut works on a copy shrunk to this many px (higher = finer, slower)", "int"),
    Param("grabcut_iters", 5, 1, 30, "files", "GrabCut refinement passes", "int"),
    Param("grabcut_margin", 0.03, 0.0, 0.3, "files", "GrabCut assumes the object is inside the photo minus this border (fraction)"),
    Param("max_file_mb", 100, 1, 2000, "files", "refuse photos larger than this many MB", "int"),
    Param("min_side_px", 24, 4, 500, "files", "refuse photos with a side shorter than this many px", "int"),
]
BY_NAME = {p.name: p for p in PARAMS}

# (check, message) - evaluated on the full candidate set, so two individually valid values can still clash
_CROSS = [
    (lambda v: v["ratio_one"] <= v["ratio_two"], "ratio_one must not exceed ratio_two"),
    (lambda v: v["reach_one"] <= v["reach_two"], "reach_one must not exceed reach_two"),
    (lambda v: v["palm_cm_min"] < v["palm_cm_max"], "palm_cm_min must be below palm_cm_max"),
    (lambda v: v["span_cm_min"] < v["span_cm_max"], "span_cm_min must be below span_cm_max"),
    (lambda v: v["palm_cm_min"] <= v["default_palm_cm"] <= v["palm_cm_max"],
     "default_palm_cm must lie between palm_cm_min and palm_cm_max"),
    (lambda v: v["foreshorten_min"] < v["foreshorten_max"], "foreshorten_min must be below foreshorten_max"),
    (lambda v: v["scale_min"] < v["scale_max"], "scale_min must be below scale_max"),
    (lambda v: v["perf_warn_fps"] < v["perf_good_fps"], "perf_warn_fps must be below perf_good_fps"),
]


def _coerce(p: Param, raw: Any) -> float | int:
    """Turn a JSON value or a command-line string into a valid number for `p`, or raise ConfigError."""
    if isinstance(raw, bool):
        raise ConfigError(f"{p.name}: expected a number, got {raw!r}")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"{p.name}: expected a number, got {raw!r}") from None
    if v != v or v in (float("inf"), float("-inf")):
        raise ConfigError(f"{p.name}: {raw!r} is not a finite number")
    if p.kind == "int":
        if not v.is_integer():
            raise ConfigError(f"{p.name}: expected a whole number, got {raw!r}")
        v = int(v)
    if not p.lo <= v <= p.hi:
        raise ConfigError(f"{p.name}: {v:g} is outside the allowed range {p.lo:g} .. {p.hi:g}")
    return v


class Config:
    """Holds the live values as plain attributes (CFG.ratio_one ...). Mutated in place so every module sees updates."""

    # One typed attribute per entry in PARAMS (kept in sync by tests/test_code_quality.py) so editors and
    # type checkers know every setting.
    ratio_one: float
    ratio_two: float
    reach_one: float
    reach_two: float
    slender_frac: float
    two_hand_min_sep: float
    default_palm_cm: float
    span_per_palm: float
    card_long_cm: float
    palm_cm_min: float
    palm_cm_max: float
    span_cm_min: float
    span_cm_max: float
    palm_lift: float
    palm_default_height: float
    foreshorten_ref: float
    foreshorten_min: float
    foreshorten_max: float
    two_hand_lift: float
    thumb_radius: float
    scale_step: float
    scale_min: float
    scale_max: float
    smooth_min_cutoff: float
    smooth_beta: float
    smooth_d_cutoff: float
    fade_in: float
    fade_out: float
    shadow_strength: float
    shadow_blur: float
    shadow_dx: float
    shadow_dy: float
    perf_good_fps: float
    perf_warn_fps: float
    perf_history: int
    grabcut_max_side: int
    grabcut_iters: int
    grabcut_margin: float
    max_file_mb: int
    min_side_px: int

    def __init__(self) -> None:
        """Start with every setting at its default."""
        self.reset()

    def reset(self) -> None:
        """Restore every setting to its default."""
        for p in PARAMS:
            setattr(self, p.name, p.default)

    def as_dict(self) -> dict[str, float | int]:
        """All current values as a plain dict - the format of the JSON config file."""
        return {p.name: getattr(self, p.name) for p in PARAMS}

    def apply(self, mapping: dict) -> list[str]:
        """Validate `mapping` as a whole and apply it all-or-nothing. Returns the names whose value changed."""
        errors, candidate = [], self.as_dict()
        for key, raw in mapping.items():
            if str(key).startswith("_"):      # "_comment" etc. are allowed in files
                continue
            p = BY_NAME.get(key)
            if p is None:
                near = difflib.get_close_matches(str(key), BY_NAME, n=2, cutoff=0.5)
                errors.append(f"unknown setting '{key}'" + (f" - did you mean {' or '.join(near)}?" if near else
                                                            " (run: python config.py  to list all settings)"))
                continue
            try:
                candidate[key] = _coerce(p, raw)
            except ConfigError as e:
                errors.append(str(e))
        if not errors:
            errors += [msg for ok, msg in _CROSS if not ok(candidate)]
        if errors:
            raise ConfigError("; ".join(errors) if len(errors) == 1 else "\n  - " + "\n  - ".join(errors))
        changed = [k for k, v in candidate.items() if getattr(self, k) != v]
        for k, v in candidate.items():
            setattr(self, k, v)
        return changed


CFG = Config()


@dataclass
class Source:
    """Where the active settings came from: the config file (if any) and the --set overrides applied on top."""
    path: Path | None = None   # the config file in use (None = defaults / --set only)
    sets: dict | None = None   # --set overrides, re-applied on top of the file at every reload


SOURCE = Source(None, {})


# ---------------------------------------------------------------------------------------------------------
def load_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON config file into a dict; any problem becomes a ConfigError with a readable message."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {p}") from None
    except PermissionError:
        raise ConfigError(f"no permission to read config file: {p}") from None
    except json.JSONDecodeError as e:
        raise ConfigError(f"{p.name} is not valid JSON (line {e.lineno}, column {e.colno}: {e.msg})") from None
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {p}: {e}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{p.name} must contain a JSON object like {{\"ratio_one\": 2.0}}")
    return data


def parse_set_items(items: Iterable[str] | None) -> dict[str, str]:
    """Turn ['a=1', 'b=2'] from repeated --set flags into {'a': '1', 'b': '2'}."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ConfigError(f"--set expects NAME=VALUE, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def discover() -> Path | None:
    """First config file found in ./ or ~/.handdoc_cam/, or None."""
    for cand in (Path.cwd() / CONFIG_FILENAME, USER_DIR / "config.json"):
        if cand.is_file():
            return cand
    return None


def add_cli_options(p: argparse.ArgumentParser) -> None:
    """Register --config, --set, --print-config and --list-config on ``p``."""
    g = p.add_argument_group("configuration (see config.py)")
    g.add_argument("--config", metavar="FILE", default=None,
                   help=f"JSON file of settings (default: ./{CONFIG_FILENAME} or ~/.handdoc_cam/config.json if present)")
    g.add_argument("--set", metavar="NAME=VALUE", action="append", default=[],
                   help="override one setting; repeatable, e.g. --set ratio_one=2.2 --set palm_lift=0.1")
    g.add_argument("--print-config", action="store_true", help="print the settings in effect as JSON and exit")
    g.add_argument("--list-config", action="store_true", help="list every setting with default, range and meaning, then exit")


def describe() -> str:
    """A table of every setting with its default, valid range and meaning, grouped by purpose."""
    lines, group = [], None
    for p in PARAMS:
        if p.group != group:
            group = p.group
            lines.append(f"\n[{group}]")
        lines.append(f"  {p.name:<19} {p.default:<7g} ({p.lo:g}..{p.hi:g})  {p.help}")
    return "\n".join(lines).lstrip("\n")


def write_example(path: str | Path) -> None:
    """Write every default to ``path`` as a starting config file."""
    data: dict[str, Any] = {"_comment": "Any subset of these works. Run `python config.py` for what each one does."}
    data.update(CFG.as_dict())
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _say(text: str) -> None:
    """print() that tolerates a closed pipe, e.g. when output is piped into ``head``."""
    try:
        print(text)
        sys.stdout.flush()
    except BrokenPipeError:  # e.g. `... --list-config | head`
        pass


def bootstrap(argv: Sequence[str] | None = None) -> Source:
    """Called once at start-up: file -> --set -> CFG. Handles --list-config / --print-config. Exits on bad input."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_cli_options(pre)
    ns, _ = pre.parse_known_args(argv)
    if ns.list_config:
        _say(describe())
        sys.exit(0)
    CFG.reset()
    try:
        sets = parse_set_items(ns.set)
        path = Path(ns.config) if ns.config else discover()
        if path is not None:
            try:
                CFG.apply(load_file(path))
            except ConfigError as e:
                if ns.config:  # the user asked for this file explicitly -> hard error
                    raise ConfigError(f"{e}") from None
                print(f"[config] ignoring {path}: {e}", file=sys.stderr)  # auto-found file: warn, use defaults
                CFG.reset()
                path = None
        CFG.apply(sets)
    except ConfigError as e:
        sys.exit(f"Config error: {e}")
    SOURCE.path, SOURCE.sets = path, sets
    if ns.print_config:
        _say(json.dumps(CFG.as_dict(), indent=2))
        sys.exit(0)
    return SOURCE


def reload() -> list[str]:
    """Re-read the config file (+ the original --set overrides) while the app runs.
    All-or-nothing: on any error the running values are unchanged. Returns the names that changed."""
    if SOURCE.path is None:
        raise ConfigError(f"no config file in use - create {CONFIG_FILENAME} next to the script "
                          "(python handdoc_cam.py --print-config > handdoc_config.json) and restart once")
    fresh = Config()
    fresh.apply(load_file(SOURCE.path))
    fresh.apply(SOURCE.sets or {})
    return CFG.apply(fresh.as_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="List HandDoc Cam settings or write a starter config file.")
    ap.add_argument("--write", metavar="FILE", help="write all defaults to FILE as a starting point")
    a = ap.parse_args()
    if a.write:
        if Path(a.write).exists():
            sys.exit(f"{a.write} already exists - not overwriting")
        write_example(a.write)
        print(f"wrote {a.write}")
    else:
        print(describe())
