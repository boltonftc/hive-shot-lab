"""Run the whole thing on a video: calibrate from tags, detect + track every frame, fit, judge, group."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import analyze
from .calibrate import Calibration, calibrate, find_tags_in_video
from .detect import Detector
from .settings import Settings
from .track import Track, Tracker

IDLE_STRIDE = 8          # while nothing is flying, only look at every 8th frame (a ball moves ~7 in meanwhile)
IDLE_AFTER_S = 0.3       # ...starting this long after the last yellow blob


def open_video(path: str | Path) -> cv2.VideoCapture:
    # Software decoding only: GPU (D3D11) decoding of iPhone HEVC returned grey frames and spammed errors.
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    return cap


@dataclass
class Session:
    video: str
    fps_file: float
    fps: float                     # the frame rate actually used (after the gravity check)
    n_frames: int
    width: int
    height: int
    settings: Settings
    calib: Calibration
    shots: list[analyze.Shot] = field(default_factory=list)
    tracks: dict[int, Track] = field(default_factory=dict)   # by track id, for drawing
    rejected: int = 0
    seconds: float = 0.0
    level_fix_deg: float = 0.0     # gravity check: the shots say the picture should turn this much more than the tag says
    level_tracks: int = 0          # how many shots that was measured from

    # ---------------------------------------------------------------- save / load
    def save(self, path: Path) -> None:
        d = {
            "video": self.video, "fps_file": self.fps_file, "fps": self.fps, "n_frames": self.n_frames,
            "width": self.width, "height": self.height, "rejected": self.rejected, "seconds": self.seconds,
            "level_fix_deg": self.level_fix_deg, "level_tracks": self.level_tracks,
            "settings": json.loads(self.settings.to_json()),
            "calib": {"H": self.calib.H.tolist(), "px_per_in": self.calib.px_per_in, "shape_err_pct": self.calib.shape_err_pct,
                      "tags_used": self.calib.tags_used, "x_sign": self.calib.x_sign, "tag_roll_deg": self.calib.tag_roll_deg},
            "shots": [asdict(s) for s in self.shots],
            "tracks": {str(k): {"frames": t.frames, "x": t.x, "y": t.y, "u": t.u, "v": t.v} for k, t in self.tracks.items()},
        }
        path.write_text(json.dumps(d), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Session":
        d = json.loads(path.read_text(encoding="utf-8"))
        c = d["calib"]
        H = np.array(c["H"])
        calib = Calibration(H, np.linalg.inv(H), c["px_per_in"], c.get("shape_err_pct", 0.0), c["tags_used"], c["x_sign"],
                            c.get("tag_roll_deg", 0.0))
        tracks = {int(k): Track(int(k), v["frames"], v["x"], v["y"], v["u"], v["v"], 0, False) for k, v in d["tracks"].items()}
        return Session(d["video"], d["fps_file"], d["fps"], d["n_frames"], d["width"], d["height"],
                       Settings.from_dict(d["settings"]), calib, [analyze.Shot(**s) for s in d["shots"]],
                       tracks, d["rejected"], d["seconds"], d.get("level_fix_deg", 0.0), d.get("level_tracks", 0))

    def reevaluate(self) -> None:
        """Re-judge every shot against the (possibly edited) virtual cell. No video needed."""
        analyze.evaluate(self.shots, self.settings)
        analyze.group_bursts(self.shots, self.settings)


def process_video(path: str | Path, settings: Settings,
                  progress: Callable[[int, int], None] | None = None,
                  cancelled: Callable[[], bool] = lambda: False) -> Session:
    t_start = time.time()
    cap = open_video(path)
    if not cap.isOpened():
        raise ValueError(f"Can't open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_file = settings.fps_override or cap.get(cv2.CAP_PROP_FPS) or 240.0

    found = find_tags_in_video(cap)
    calib = calibrate(found, settings)
    det = Detector(settings, calib, w, h)
    trk = Tracker(settings)

    idle_after = max(1, int(round(IDLE_AFTER_S * fps_file)))
    last_seen = -10 ** 9
    f = 0
    while True:
        idle = f - last_seen > idle_after
        if idle and f % IDLE_STRIDE:
            if not cap.grab():
                break
            f += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect(frame)
        if dets:
            last_seen = f
            world = calib.to_world(np.array([[d[0], d[1]] for d in dets]))
        else:
            world = np.zeros((0, 2))
        trk.step(f, world, dets)
        f += 1
        if progress and f % 240 == 0:
            progress(f, n)
            if cancelled():
                break
    cap.release()
    if progress:
        progress(n, n)

    tracks = trk.finish()
    fps = fps_file * analyze.guess_time_scale(tracks, fps_file, settings)

    lean, n_level = analyze.measure_tilt(tracks, fps)      # a check only: drag and lean trade off, so the tag sets 'up'

    shots, rejected = [], 0
    for tr in tracks:
        sh = analyze.fit_track(tr, fps, settings)
        if sh and analyze.is_shot(sh, settings):
            shots.append(sh)
        else:
            rejected += 1
    session = Session(str(path), fps_file, fps, max(n, f), w, h, settings, calib, shots,
                      {t.id: t for t in tracks}, rejected, time.time() - t_start,
                      -lean if n_level else 0.0, n_level)
    session.reevaluate()
    return session
