"""Turn one AprilTag seen in the video into a map from image pixels to world inches."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .settings import Settings

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def detect_tags(gray: np.ndarray) -> dict[int, np.ndarray]:
    """Return {tag id: 4x2 corner array (TL, TR, BR, BL as printed)}."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners: the scale depends on them
    det = cv2.aruco.ArucoDetector(_DICT, params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


@dataclass
class Calibration:
    H: np.ndarray            # world (in) -> image (px)
    H_inv: np.ndarray        # image (px) -> world (in)
    px_per_in: float
    shape_err_pct: float     # how far the tag is from a perfect square in the picture (camera not square / tag bent)
    tags_used: list[int]
    x_sign: int              # +1 if world x grows to the right in the image
    tag_roll_deg: float = 0.0  # how far the tag's edges are turned from level in the picture (the grid follows it)

    def to_world(self, pts_px: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_inv).reshape(-1, 2)

    def to_image(self, pts_in: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_in, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)


def pick_tag(found: dict[int, np.ndarray], settings: Settings) -> int:
    if not found:
        raise ValueError("No AprilTag (tag36h11) found in the first seconds of the video")
    if settings.tag_id >= 0:
        if settings.tag_id not in found:
            raise ValueError(f"Tag {settings.tag_id} not found (saw {sorted(found)})")
        return settings.tag_id
    return max(found, key=lambda i: cv2.contourArea(found[i].astype(np.float32)))


def calibrate(found: dict[int, np.ndarray], settings: Settings) -> Calibration:
    """Scale and 'up' from the tag (it stands upright in the flight plane), origin from the clicked turret.

    The camera looks square at the flight plane, so this is scale + rotation + shift only: perspective
    from one small tag would be noisy far away from it.
    """
    tid = pick_tag(found, settings)
    if settings.turret_px is None:
        raise ValueError("Turret not set: press 'Pick turret on video' and click the turret center")
    c = found[tid]
    edges = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    diags = np.array([np.linalg.norm(c[2] - c[0]), np.linalg.norm(c[3] - c[1])]) / math.sqrt(2)
    sides = np.concatenate([edges, diags])
    s = float(sides.mean() / settings.tag_size_in)
    shape_err = float(100 * (sides.max() - sides.min()) / sides.mean())

    u0, v0 = settings.turret_px
    to_turret = np.array([[1 / s, 0, -u0 / s], [0, -1 / s, v0 / s], [0, 0, 1]])   # pixels -> inches, y up
    sign = 1 if settings.shoot_right else -1
    F = np.diag([sign, 1.0])
    q = np.c_[c[:, 0] - u0, v0 - c[:, 1]] @ F               # tag corners with y up, mirrored like the world
    d = np.roll(q, -1, axis=0) - q
    tilt = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 45) % 90 - 45   # each edge's tilt from square
    roll = float(np.mean(tilt))
    a = math.radians(settings.level_adjust_deg - roll)
    L = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ F
    B = np.eye(3)
    B[:2, :2] = L
    B[1, 2] = settings.shooter_y
    H_inv = B @ to_turret
    return Calibration(np.linalg.inv(H_inv), H_inv, s, shape_err, [tid], sign, roll * sign)


def calibrate_from_video(cap: cv2.VideoCapture, settings: Settings, samples: int = 12) -> Calibration:
    return calibrate(find_tags_in_video(cap, samples), settings)


def find_tags_in_video(cap: cv2.VideoCapture, samples: int = 12) -> dict[int, np.ndarray]:
    """Look at frames spread over the first ~2 s and return the median corners of each tag."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    seen: dict[int, list[np.ndarray]] = {}
    span = min(n, 480)
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * span / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        for tid, c in detect_tags(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).items():
            seen.setdefault(tid, []).append(c)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return {tid: np.median(np.stack(cs), axis=0) for tid, cs in seen.items()}
