"""A batch: several videos analysed together, added one at a time (e.g. one video per robot setup).

Each video keeps its own calibration and robot setup (turret click, level, hood / rpm / flywheel mass) in its own
<video>.session.json. The batch file only lists the videos and holds the settings they share (target, ball, tracking).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import fields
from pathlib import Path

from .pipeline import Session
from .settings import Settings

LABELS = ("robot", "hood_deg", "flywheel_rpm", "flywheel_mass_g")
PER_VIDEO = ("turret_px", "shoot_right", "level_adjust_deg", "tag_id", "fps_override") + LABELS
# changing these only needs Re-judge; any other shared setting needs Process again
REJUDGE = ("ball_diameter_in", "cell_bottom_x", "cell_bottom_y", "cell_open_in", "cell_lean_deg",
           "burst_gap_s", "max_extrapolate_s")


def shared_names() -> list[str]:
    return [f.name for f in fields(Settings) if f.name not in PER_VIDEO]


def with_shared(dst: Settings, src: Settings, names: tuple[str, ...] | list[str] | None = None) -> Settings:
    """A copy of dst with the batch-wide settings (or just `names`) taken from src."""
    d = json.loads(dst.to_json())
    s = json.loads(src.to_json())
    for k in shared_names() if names is None else names:
        d[k] = s[k]
    return Settings.from_dict(d)


def needs_reprocess(processed: Settings, current: Settings) -> bool:
    a, b = json.loads(processed.to_json()), json.loads(current.to_json())
    return any(a[k] != b[k] for k in a if k not in REJUDGE + LABELS)


def labels_from_filename(stem: str) -> dict:
    """'robot1_2571rpm_152g_2' -> robot robot1, 2571 rpm, 152 g (also '30deg' or 'hood30' for the hood)."""
    out: dict = {}
    for raw in stem.split("_"):
        tok = raw.lower()
        if m := re.fullmatch(r"(\d+(?:\.\d+)?)rpm", tok):
            out["flywheel_rpm"] = float(m[1])
        elif m := re.fullmatch(r"(\d+(?:\.\d+)?)g", tok):
            out["flywheel_mass_g"] = float(m[1])
        elif m := re.fullmatch(r"hood(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)deg", tok):
            out["hood_deg"] = float(m[1] or m[2])
        elif "robot" not in out and re.fullmatch(r"[a-z][a-z0-9-]*", tok) and not tok.startswith("img"):
            out["robot"] = raw
    return out if any(k != "robot" for k in out) else {}


class Batch:
    def __init__(self, path: Path, settings: Settings, videos: list[Path] | None = None) -> None:
        self.path = path
        self.settings = settings
        self.videos: list[Path] = videos or []

    @property
    def name(self) -> str:
        return self.path.name.removesuffix(".batch.json")

    def save(self) -> None:
        def rel(v: Path) -> str:
            try:
                return os.path.relpath(v, self.path.parent)
            except ValueError:             # different drive on Windows
                return str(v)
        d = {"videos": [rel(v) for v in self.videos], "settings": json.loads(self.settings.to_json())}
        self.path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Batch":
        d = json.loads(path.read_text(encoding="utf-8"))
        vids = [(path.parent / v).resolve() for v in d.get("videos", [])]
        return Batch(path, Settings.from_dict(d.get("settings", {})), vids)

    def add(self, video: Path) -> bool:
        video = video.resolve()
        if video in self.videos:
            return False
        self.videos.append(video)
        return True

    def find(self, name: str) -> Path | None:
        return next((v for v in self.videos if v.name == name), None)

    @staticmethod
    def session_path(video: Path) -> Path:
        return video.with_suffix(".session.json")

    def load_session(self, video: Path) -> Session | None:
        p = self.session_path(video)
        return Session.load(p) if p.exists() else None
