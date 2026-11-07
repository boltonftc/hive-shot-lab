"""From tracks to shots: fit each flight, find where it crosses the cell opening, group into bursts."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .settings import GRAVITY_IN_S2, Settings
from .track import Track

LAUNCH_RADIUS_IN = 18.0   # a shot must be first seen this close to the turret center


def _flight(p: list[float] | np.ndarray, t: np.ndarray, tilt: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Projectile with air drag that slows it in proportion to its speed (k per second).

    p = [x0, y0, vx, vy, k], t = seconds since the first sighting. k = 0 is a plain parabola.
    tilt (radians) leans gravity toward +x; used only to measure a tilted camera.
    """
    x0, y0, vx, vy, k = p
    gx, gy = GRAVITY_IN_S2 * math.sin(tilt), -GRAVITY_IN_S2 * math.cos(tilt)
    if k < 1e-6:
        return x0 + vx * t + 0.5 * gx * t * t, y0 + vy * t + 0.5 * gy * t * t
    e = -np.expm1(-k * t) / k                      # (1 - e^(-kt)) / k, accurate for small k
    return x0 + (vx - gx / k) * e + gx / k * t, y0 + (vy - gy / k) * e + gy / k * t


@dataclass
class Shot:
    track_id: int
    frame0: int
    frame1: int
    t0: float                   # first sighting, seconds into the video
    n_points: int
    v0: float                   # speed at first sighting, in/s
    angle_deg: float            # launch angle at first sighting (0 = flat, 90 = straight up)
    apex_y: float               # highest point, inches above the floor
    accel: float                # downward acceleration of a plain parabola fit, in/s^2 (should be close to gravity)
    drag_k: float               # fitted air drag, 1/s (how quickly the ball loses speed)
    fit_rms: float              # how far the points sit from the fitted curve, inches
    cross_s: float | None       # where it crossed the opening line: inches from the bottom edge, along the lean
    cross_x: float | None
    cross_y: float | None
    result: str                 # IN / LOW / HIGH / SHORT / ?
    fps: float = 240.0
    params: list[float] = field(default_factory=list)    # [x0, y0, vx, vy, k] of the drag model, t from first sighting
    burst_id: int = 0
    index_in_burst: int = 0
    burst_size: int = 0
    shot_id: int = 0

    def position(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _flight(self.params, np.asarray(t, float))


def _cross_opening(shot: Shot, t_end: float, s: Settings) -> tuple[float | None, float | None, float | None]:
    """First place the path passes through the opening's line from the shooter's side to the back.

    Returns (distance along the opening from its bottom edge, x, y). Below 0 = under the opening.
    """
    (bx, by), (tx, ty) = s.cell_points()
    dx, dy = tx - bx, ty - by
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    t = np.linspace(0.0, t_end, max(50, int(t_end * 2000)))
    x, y = shot.position(t)
    side = ux * (y - by) - uy * (x - bx)          # > 0: shooter side of the line (front), < 0: behind it
    idx = np.where((side[:-1] > 0) & (side[1:] <= 0))[0]
    if len(idx) == 0:
        return None, None, None
    i = int(idx[0])
    f = side[i] / (side[i] - side[i + 1])
    cx, cy = x[i] + f * (x[i + 1] - x[i]), y[i] + f * (y[i + 1] - y[i])
    return float((cx - bx) * ux + (cy - by) * uy), float(cx), float(cy)


def classify(cross_s: float | None, s: Settings) -> str:
    if cross_s is None:
        return "SHORT"
    r = s.ball_diameter_in / 2
    if cross_s < r:
        return "LOW"
    if cross_s > s.cell_open_in - r:
        return "HIGH"
    return "IN"


def fit_track(tr: Track, fps: float, s: Settings) -> Shot | None:
    t = (np.array(tr.frames, float) - tr.frames[0]) / fps
    x, y = np.array(tr.x), np.array(tr.y)
    cx = np.polyfit(t, x, 2)                       # quick parabola: starting guess + gravity check
    cy = np.polyfit(t, y, 2)
    accel = -2 * cy[0]

    def resid(p):
        fx, fy = _flight(p, t)
        return np.concatenate([fx - x, fy - y])

    start = [cx[2], cy[2], cx[1], cy[1], 0.1]
    sol = least_squares(resid, start, bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0.0],
                                              [np.inf, np.inf, np.inf, np.inf, 5.0]))
    p = sol.x
    fx, fy = _flight(p, t)
    rms = float(np.sqrt(np.mean((fx - x) ** 2 + (fy - y) ** 2)))
    x0, y0, vx, vy, k = p
    g = GRAVITY_IN_S2
    if vy > 0:
        t_apex = math.log1p(k * vy / g) / k if k > 1e-6 else vy / g
        apex = float(_flight(p, np.array([t_apex]))[1][0])
    else:
        apex = float(y0)
    return Shot(track_id=tr.id, frame0=tr.frames[0], frame1=tr.frames[-1], t0=tr.frames[0] / fps,
                n_points=len(tr.frames), v0=float(math.hypot(vx, vy)), angle_deg=float(math.degrees(math.atan2(vy, abs(vx)))),
                apex_y=apex, accel=float(accel), drag_k=float(k), fit_rms=rms,
                cross_s=None, cross_x=None, cross_y=None, result="?", fps=fps, params=p.tolist())


def evaluate(shots: list[Shot], s: Settings) -> None:
    """(Re)compute the cell crossing + result for each shot. Cheap: run again after editing the cell."""
    for sh in shots:
        t_end = (sh.frame1 - sh.frame0) / sh.fps + s.max_extrapolate_s
        sh.cross_s, sh.cross_x, sh.cross_y = _cross_opening(sh, t_end, s)
        sh.result = classify(sh.cross_s, s)


def is_ballistic(sh: Shot) -> bool:
    """A real flight curves down at roughly g and fits a smooth arc."""
    return 0.5 * GRAVITY_IN_S2 <= sh.accel <= 1.6 * GRAVITY_IN_S2 and sh.fit_rms < 1.5


def is_shot(sh: Shot, s: Settings) -> bool:
    """A flight that starts at the turret and rises above it (not a piece of one, or a ball hopping on the floor)."""
    return (is_ballistic(sh) and math.hypot(sh.params[0], sh.params[1] - s.shooter_y) <= LAUNCH_RADIUS_IN
            and sh.apex_y > s.shooter_y + 6)


def measure_tilt(tracks: list[Track], fps: float) -> tuple[float, int]:
    """Which way is really down? Fit long tracks letting gravity lean; return (median lean in degrees, tracks used).

    Positive = gravity leans toward +x, so the picture must be turned by minus that. Trusts the tag's scale:
    with drag in play, one track can't pin down both the lean and the size of gravity.
    """
    leans = []
    for tr in tracks:
        if len(tr.frames) < 30:
            continue
        t = (np.array(tr.frames, float) - tr.frames[0]) / fps
        x, y = np.array(tr.x), np.array(tr.y)
        cx, cy = np.polyfit(t, x, 2), np.polyfit(t, y, 2)
        if not 0.5 * GRAVITY_IN_S2 < -2 * cy[0] < 1.6 * GRAVITY_IN_S2:
            continue

        def resid(p):
            fx, fy = _flight(p[:5], t, p[5])
            return np.concatenate([fx - x, fy - y])

        sol = least_squares(resid, [cx[2], cy[2], cx[1], cy[1], 0.1, 0.0],
                            bounds=([-np.inf] * 4 + [0.0, -0.5], [np.inf] * 4 + [5.0, 0.5]))
        if np.sqrt(np.mean(sol.fun ** 2) * 2) < 1.5:
            leans.append(math.degrees(sol.x[5]))
    return (float(np.median(leans)), len(leans)) if leans else (0.0, 0)


def guess_time_scale(tracks: list[Track], fps: float, s: Settings) -> float:
    """iPhone slo-mo exported 'for playback' is stamped 30 fps though every frame is 1/240 s apart.

    Gravity tells us the truth: if the arcs fall too slowly, time is stretched. Returns the factor to
    multiply the file's fps by (1.0 if it already looks right), snapped to a common slo-mo ratio.
    """
    acc = []
    long_tracks = [tr for tr in tracks if len(tr.frames) >= 10]
    launched = [tr for tr in long_tracks                              # real shots start at the turret; others are
                if math.hypot(tr.x[0], tr.y[0] - s.shooter_y) <= LAUNCH_RADIUS_IN]   # clutter (arms, people, bounces)
    for tr in launched or long_tracks:
        t = (np.array(tr.frames, float) - tr.frames[0]) / fps
        cy = np.polyfit(t, tr.y, 2)
        if cy[0] < 0:
            acc.append(-2 * cy[0])
    if len(acc) < 1:
        return 1.0
    ratio = math.sqrt(GRAVITY_IN_S2 / float(np.median(acc)))
    if 0.8 <= ratio <= 1.25:
        return 1.0
    return min((1 / 8, 1 / 4, 1 / 2, 2, 4, 8), key=lambda r: abs(math.log(r / ratio)))


def group_bursts(shots: list[Shot], s: Settings) -> None:
    shots.sort(key=lambda sh: sh.t0)
    burst = -1
    last_t = -1e9
    members: dict[int, list[Shot]] = {}
    for sh in shots:
        if sh.t0 - last_t > s.burst_gap_s:
            burst += 1
        last_t = sh.t0
        sh.burst_id = burst
        members.setdefault(burst, []).append(sh)
    for i, sh in enumerate(shots):
        sh.shot_id = i + 1
    for b, group in members.items():
        for k, sh in enumerate(group):
            sh.index_in_burst = k + 1
            sh.burst_size = len(group)
