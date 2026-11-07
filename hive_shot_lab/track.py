"""Link per-frame detections into tracks, one track per ball in flight.

Prediction uses the ball's own last few points (no gravity constant), so tracking works even if the
video file reports the wrong frame rate (common with iPhone slo-mo exports).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .settings import Settings

STILL_STEP_IN = 0.25     # moving less than this per frame = sitting still (waiting in the turret, resting on the floor)
FLIGHT_TOL_IN = 3.0      # a point this far off the smooth flight curve doesn't belong to the flight
MERGE_AREA = 1.6         # a blob this much bigger than the ball's usual blob = two balls overlapping in the picture
MAX_OCCLUDED = 40        # frames a ball may stay hidden inside another ball's blob


@dataclass
class Track:
    id: int
    frames: list[int] = field(default_factory=list)
    x: list[float] = field(default_factory=list)      # world inches
    y: list[float] = field(default_factory=list)
    u: list[float] = field(default_factory=list)      # image pixels (for drawing)
    v: list[float] = field(default_factory=list)
    missed: int = 0
    active: bool = True
    occluded: int = 0                                  # frames hidden inside another ball's blob
    areas: list[float] = field(default_factory=list)   # blob sizes (not saved; used to spot merged blobs)

    def add(self, f: int, x: float, y: float, u: float, v: float, area: float = 0.0) -> None:
        self.frames.append(f); self.x.append(x); self.y.append(y); self.u.append(u); self.v.append(v)
        self.areas.append(area)
        self.missed = 0
        self.occluded = 0

    def drop_last(self) -> None:
        for name in ("frames", "x", "y", "u", "v", "areas"):
            getattr(self, name).pop()

    def typical_area(self) -> float:
        return float(np.median(self.areas[-10:])) if len(self.areas) >= 3 else 0.0

    def predict(self, f: int) -> tuple[float, float]:
        n = len(self.frames)
        if n == 1:
            return self.x[-1], self.y[-1]
        k = min(n, 4 if f - self.frames[-1] <= 2 else 10)       # coasting through a gap: use a longer stretch
        fr = np.array(self.frames[-k:], float)
        deg = 2 if k >= 3 else 1
        px = np.polyfit(fr, self.x[-k:], deg)
        py = np.polyfit(fr, self.y[-k:], deg)
        return float(np.polyval(px, f)), float(np.polyval(py, f))

    def moving(self) -> bool:
        if len(self.frames) < 2:
            return False
        step = np.hypot(self.x[-1] - self.x[-2], self.y[-1] - self.y[-2]) / (self.frames[-1] - self.frames[-2])
        return bool(step > 4 * STILL_STEP_IN)

    def travel(self) -> float:
        return float(np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0])) if self.x else 0.0

    def trim_still(self) -> None:
        """Cut off the start/end where the ball sits still, so only the flight is left to fit."""
        if len(self.frames) < 5:
            return
        step = np.hypot(np.diff(self.x), np.diff(self.y)) / np.diff(self.frames)
        moving = step >= STILL_STEP_IN
        run = moving[:-2] & moving[1:-1] & moving[2:]          # 3 moving steps in a row: not just jitter
        if not run.any():
            return
        a = int(np.argmax(run))
        b = len(run) - int(np.argmax(run[::-1])) + 3
        for name in ("frames", "x", "y", "u", "v"):
            setattr(self, name, getattr(self, name)[a:b])

    def trim_to_flight(self, tol_in: float = FLIGHT_TOL_IN) -> None:
        """Keep the stretch from the start that one smooth curve explains; cut where it stops fitting
        (the ball bounced, or the tracker jumped to another yellow object)."""
        n = len(self.frames)
        if n < 8:
            return
        f = np.asarray(self.frames, float) - self.frames[0]       # frame index, so a wrong file fps doesn't matter
        x, y = np.asarray(self.x), np.asarray(self.y)
        end = min(n, 16)
        while end < n:
            deg = 3 if end >= 12 else 2
            px, py = np.polyfit(f[:end], x[:end], deg), np.polyfit(f[:end], y[:end], deg)
            bad = np.hypot(np.polyval(px, f[end:]) - x[end:], np.polyval(py, f[end:]) - y[end:]) > tol_in
            b = np.r_[bad, True, True]                             # running off the end counts as misses
            run = b[:-2] & b[1:-1] & b[2:]                         # 3 misses in a row: not just one noisy point
            if not run.any():
                end = n
                break
            new_end = end + int(np.argmax(run))
            if new_end <= end:
                break
            end = new_end
        if end < n:
            for name in ("frames", "x", "y", "u", "v"):
                setattr(self, name, getattr(self, name)[:end])


class Tracker:
    def __init__(self, settings: Settings):
        self.s = settings
        self.tracks: list[Track] = []
        self._next = 0

    def step(self, f: int, dets_world: np.ndarray, dets_px: list[tuple[float, float, float]]) -> None:
        active = [t for t in self.tracks if t.active]
        typical = [t.typical_area() for t in active]
        preds = [t.predict(f) for t in active]
        pairs = []
        for ti, t in enumerate(active):
            px, py = preds[ti]
            # Only a ball already flying gets a wider search after missed frames; a still blob (robot arm,
            # ball resting in the turret) must not jump onto a ball passing nearby.
            gate = self.s.gate_in * (1 + t.missed) if t.moving() else self.s.gate_in
            for di, (x, y) in enumerate(dets_world):
                d = float(np.hypot(x - px, y - py))
                if d <= gate:
                    pairs.append((d, ti, di))
        pairs.sort()
        used_t, owner = set(), {}
        for _, ti, di in pairs:          # greedy: closest prediction/detection pairs first
            if ti in used_t or di in owner:
                continue
            used_t.add(ti); owner[di] = ti
            x, y = dets_world[di]
            active[ti].add(f, float(x), float(y), dets_px[di][0], dets_px[di][1], dets_px[di][2])
        near = 2 * self.s.ball_diameter_in
        for ti, t in enumerate(active):
            if ti in used_t:
                continue
            # Two balls flying close together show up as ONE bigger blob: keep both tracks alive, but record
            # neither (the blob's center is between the balls), and pick them up again when they separate.
            merged = next((di for di, oi in owner.items()
                           if typical[ti] and typical[oi] and dets_px[di][2] > MERGE_AREA * max(typical[ti], typical[oi])
                           and np.hypot(*(np.asarray(dets_world[di]) - preds[ti])) < near), None)
            if merged is not None:
                other = active[owner[merged]]
                if other.frames and other.frames[-1] == f:
                    other.drop_last()
                    other.occluded += 1
                t.occluded += 1
                if t.occluded > MAX_OCCLUDED:
                    t.active = False
                continue
            t.missed += 1
            if t.missed > self.s.max_missed_frames:
                t.active = False
        for di, (x, y) in enumerate(dets_world):
            if di not in owner:
                t = Track(self._next); self._next += 1
                t.add(f, float(x), float(y), dets_px[di][0], dets_px[di][1], dets_px[di][2])
                self.tracks.append(t)

    def finish(self) -> list[Track]:
        for t in self.tracks:
            t.active = False
            t.trim_still()
            t.trim_to_flight()
        return [t for t in self.tracks
                if len(t.frames) >= self.s.min_track_points and t.travel() >= self.s.min_travel_in]
