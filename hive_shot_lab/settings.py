"""All tunable numbers in one place. Saved as JSON next to each video so a session can be re-run.

World frame (inches): x = horizontal distance from the TURRET CENTER toward the hive cell,
y = height above the FLOOR. The camera looks at this plane from the side.

Nothing has to be measured in the scene: one AprilTag of known size standing upright in the flight plane gives
the scale and which way is up, and a click on the turret gives the origin.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

GRAVITY_IN_S2 = 386.09


@dataclass
class Settings:
    # --- scale reference: one AprilTag (family tag36h11) standing in the flight plane ---
    tag_size_in: float = 6.5                    # printed black square, edge to edge (DECODE field tags: 6.5)
    tag_id: int = -1                            # -1 = use the biggest tag in view

    # --- the ball ---
    ball_diameter_in: float = 2.8
    hsv_low: tuple[int, int, int] = (18, 80, 80)     # OpenCV HSV: H 0..180
    hsv_high: tuple[int, int, int] = (40, 255, 255)

    # --- where to look for flying balls (world inches) ---
    zone_x_min: float = -12.0
    zone_x_max: float = 90.0
    zone_y_min: float = 4.0
    zone_y_max: float = 110.0

    # --- robot / camera ---
    turret_px: tuple[float, float] | None = None   # turret center clicked in the video (pixels)
    shooter_y: float = 14.0                        # turret center height above the floor
    shoot_right: bool = True                       # shots fly toward the right side of the picture
    level_adjust_deg: float = 0.0                  # extra tilt on top of the one read from the tag's edges; normally 0

    # --- robot setup in this video (labels for comparing videos; 0 / "" = not entered) ---
    robot: str = ""
    hood_deg: float = 0.0
    flywheel_rpm: float = 0.0
    flywheel_mass_g: float = 0.0

    # --- virtual hive cell opening (side view: a line segment leaning back) ---
    cell_bottom_x: float = 47.0                 # horizontal distance turret -> bottom edge of the opening
    cell_bottom_y: float = 53.5                 # bottom edge height
    cell_open_in: float = 14.0                  # opening height measured along the lean
    cell_lean_deg: float = 30.0                 # leans back (away from the shooter) from vertical

    # --- tracking / grouping ---
    fps_override: float = 0.0                   # 0 = use the file's frame rate
    burst_gap_s: float = 0.6                    # launches closer together than this are one burst
    min_track_points: int = 8
    max_missed_frames: int = 6
    gate_in: float = 10.0                       # how far (in) a ball may be from its prediction
    min_travel_in: float = 12.0                 # tracks that move less than this are ignored
    max_extrapolate_s: float = 0.6              # how far past the last seen point we extend the fit

    # ------------------------------------------------------------------ helpers
    def cell_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Bottom and top edge of the opening, in world inches."""
        lean = math.radians(self.cell_lean_deg)
        bx, by = self.cell_bottom_x, self.cell_bottom_y
        return (bx, by), (bx + self.cell_open_in * math.sin(lean), by + self.cell_open_in * math.cos(lean))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Settings":
        s = Settings()
        for k, v in d.items():
            if k == "turret_px":
                s.turret_px = (float(v[0]), float(v[1])) if v else None
            elif hasattr(s, k):
                setattr(s, k, tuple(v) if isinstance(getattr(s, k), tuple) else v)
        return s

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Settings":
        return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
