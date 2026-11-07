"""Find yellow pollen in a frame: colour threshold inside the flight zone, then blobs of a ball-like size."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .calibrate import Calibration
from .settings import Settings

DETECT_SCALE = 0.5   # work on a half-size image: 4x less work, plenty for a ~25 px ball


class Detector:
    def __init__(self, settings: Settings, calib: Calibration, frame_w: int, frame_h: int):
        zone = np.array([[settings.zone_x_min, settings.zone_y_min], [settings.zone_x_max, settings.zone_y_min],
                         [settings.zone_x_max, settings.zone_y_max], [settings.zone_x_min, settings.zone_y_max]])
        poly = calib.to_image(zone)
        x0 = int(max(0, np.floor(poly[:, 0].min()))); x1 = int(min(frame_w, np.ceil(poly[:, 0].max())))
        y0 = int(max(0, np.floor(poly[:, 1].min()))); y1 = int(min(frame_h, np.ceil(poly[:, 1].max())))
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError("The flight zone is outside the video frame -- check the tag positions and zone settings")
        self.roi = (x0, y0, x1, y1)
        self.zone_px = poly
        s = DETECT_SCALE
        mask = np.zeros((int(round((y1 - y0) * s)), int(round((x1 - x0) * s))), np.uint8)
        cv2.fillPoly(mask, [np.round((poly - [x0, y0]) * s).astype(np.int32)], 255)
        self.mask = mask
        r_px = settings.ball_diameter_in / 2 * calib.px_per_in * s
        self.min_area = max(4.0, math.pi * (0.45 * r_px) ** 2)       # partly hidden / far side of the plane
        self.max_area = math.pi * (3.0 * r_px) ** 2                   # allows a motion-blur streak
        self.lo = np.array(settings.hsv_low, np.uint8)
        self.hi = np.array(settings.hsv_high, np.uint8)
        self.kernel = np.ones((3, 3), np.uint8)

    def yellow_mask(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.roi
        small = cv2.resize(frame[y0:y1, x0:x1], (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_AREA)
        m = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), self.lo, self.hi)
        m = cv2.bitwise_and(m, self.mask)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        """Ball candidates as (u, v, area_px) in full-frame pixels."""
        m = self.yellow_mask(frame)
        n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        x0, y0, _, _ = self.roi
        out = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                cx, cy = cents[i]
                out.append((x0 + cx / DETECT_SCALE, y0 + cy / DETECT_SCALE, area / DETECT_SCALE ** 2))
        return out
