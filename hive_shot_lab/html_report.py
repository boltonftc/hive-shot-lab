"""One self-contained web page: method, procedure, files used, settings, results tables and charts."""
from __future__ import annotations

import base64
import datetime
import html
import io
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from .batch import Batch
from .pipeline import Session
from .report import setup_label
from .settings import Settings

METHOD = [
    ("Scale and level", "One AprilTag (tag36h11) of known size stands upright in the plane the ball flies in. Its size "
     "in the picture gives pixels per inch (sub-pixel corners), and its edges give which way is up. The turret center, "
     "clicked once per video, is the origin: x = inches from the turret toward the cell, y = inches above the floor."),
    ("Finding the ball", "In every frame the yellow pollen is found by colour (HSV threshold) inside a flight zone, on a "
     "half-size image, keeping only ball-sized blobs. While nothing is flying only every 8th frame is checked."),
    ("Tracking", "Each blob is linked to the ball whose predicted position (from its last 4 points) is nearest. The part "
     "where the ball waits still in the turret is cut off, and so is the part after the flight ends (a bounce, or "
     "another yellow object): the path is cut where 3 points in a row land more than 3 in off a smooth curve."),
    ("Physics", "Each path is fitted with a projectile that has air drag (a plain parabola reads the speed about 2% low). "
     "The fit gives launch speed and angle at the first sighting, the highest point, and the whole path."),
    ("What counts as a shot", "The flight falls at 0.5-1.6 g, fits within 1.5 in on average, starts within 18 in of "
     "the turret and rises at least 6 in above it. Anything else (someone carrying a ball, a ball rolling or "
     "bouncing) is ignored."),
    ("Judging", "The cell opening is a line segment in the side view. Where the path's center crosses it decides IN "
     "(at least one ball radius inside both edges), LOW, HIGH, or SHORT (never reaches it). Launches closer together "
     "than the burst gap form a burst; balls are numbered #1, #2, ... within it."),
    ("Checks", "Gravity check: every path should fall at about 1 g; a slow-motion file saved at playback speed is "
     "detected this way and its time scale corrected. Accuracy was tested on a generated video with known answers: "
     "launch speed within 0.3 %, angle within 0.2 deg."),
]

PROCEDURE = [
    "iPhone slow-mo (120 or 240 fps), 1080p, HDR off, on a tripod, square to the flight plane (side view).",
    "Solid background (ideally a green screen) behind the flight path; no other yellow objects in view.",
    "AprilTag upright on a stand in the flight plane, out of the ball's path; its black square measured once.",
    "Video copied to the PC uncompressed (USB cable / AirDrop, 'Keep Originals').",
    "In the app: open the video, check the tag size, click the turret center, check the robot setup, Process "
    "(or Batch process for many videos). Re-judge after changing only the cell.",
]


def _png(fig: Figure) -> str:
    size = fig.get_size_inches().copy()
    fig.set_size_inches(11, 6.5)                    # same size in the report whatever the window size is
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    fig.set_size_inches(size)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _table(df: pd.DataFrame | None, index: bool = False) -> str:
    if df is None or df.empty:
        return "<p class='muted'>(none)</p>"
    return df.to_html(index=index, border=0, classes="tbl", na_rep="", float_format=lambda v: f"{v:g}")


def _files_table(sessions: list[Session]) -> pd.DataFrame:
    rows = []
    for ses in sessions:
        c, s = ses.calib, ses.settings
        rows.append({
            "video": Path(ses.video).name, "folder": str(Path(ses.video).parent), "robot setup": setup_label(s),
            "size": f"{ses.width}x{ses.height}", "frames": ses.n_frames,
            "fps (file / used)": f"{ses.fps_file:.2f} / {ses.fps:.2f}", "tag": ", ".join(map(str, c.tags_used)),
            "px per in": round(c.px_per_in, 3), "tag tilt deg": round(c.tag_roll_deg, 2),
            "tag out of square %": round(c.shape_err_pct, 2),
            "turret px": "" if s.turret_px is None else f"({s.turret_px[0]:.0f}, {s.turret_px[1]:.0f})",
            "shots": len(ses.shots), "ignored tracks": ses.rejected, "processing s": round(ses.seconds),
        })
    return pd.DataFrame(rows)


def _settings_table(s: Settings) -> pd.DataFrame:
    rows = [
        ("AprilTag black square", f"{s.tag_size_in:g} in"), ("Ball diameter", f"{s.ball_diameter_in:g} in"),
        ("Yellow (HSV) low / high", f"{tuple(s.hsv_low)} / {tuple(s.hsv_high)}"),
        ("Turret height", f"{s.shooter_y:g} in"),
        ("Cell opening bottom edge", f"x = {s.cell_bottom_x:g} in, height = {s.cell_bottom_y:g} in"),
        ("Cell opening size / lean", f"{s.cell_open_in:g} in, leaning back {s.cell_lean_deg:g} deg"),
        ("Flight zone", f"x {s.zone_x_min:g} to {s.zone_x_max:g} in, height {s.zone_y_min:g} to {s.zone_y_max:g} in"),
        ("Burst gap", f"{s.burst_gap_s:g} s"), ("Match distance / max missed frames", f"{s.gate_in:g} in / {s.max_missed_frames}"),
        ("Min points / min travel per track", f"{s.min_track_points} / {s.min_travel_in:g} in"),
        ("Path extended past last sighting", f"{s.max_extrapolate_s:g} s"),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def write_report(path: Path, *, title: str, sessions: list[Session], batch: Batch | None, settings: Settings,
                 shots: pd.DataFrame, summary: pd.DataFrame | None, group_name: str,
                 charts: list[tuple[str, Figure]], logo: Path | None = None, accent: str = "#2B5797") -> None:
    esc = html.escape
    n_shots = len(shots)
    n_in = int((shots["result"] == "IN").sum()) if n_shots else 0
    what = f"batch <b>{esc(batch.name)}</b>" if batch else f"video <b>{esc(Path(sessions[0].video).name)}</b>"
    logo_html = ""
    if logo and logo.exists():
        logo_html = f"<img class='logo' src='data:image/png;base64,{base64.b64encode(logo.read_bytes()).decode()}'>"
    charts_html = "".join(f"<figure><img src='data:image/png;base64,{_png(fig)}'><figcaption>{esc(name)}</figcaption>"
                          "</figure>" for name, fig in charts)
    method_html = "".join(f"<li><b>{esc(h)}.</b> {esc(t)}</li>" for h, t in METHOD)
    proc_html = "".join(f"<li>{esc(t)}</li>" for t in PROCEDURE)
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; color: #1b1b1b; background: #f6f8fc; }}
header {{ background: {accent}; color: white; padding: 18px 28px; display: flex; align-items: center; gap: 22px; }}
header h1 {{ margin: 0; font-size: 28px; }} header p {{ margin: 4px 0 0; opacity: .9; }}
.logo {{ height: 70px; background: white; border-radius: 10px; padding: 6px; }}
main {{ padding: 10px 28px 40px; max-width: 1300px; }}
h2 {{ color: {accent}; border-bottom: 3px solid {accent}; padding-bottom: 4px; margin-top: 34px; }}
.kpis {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.kpi {{ background: white; border: 2px solid {accent}; border-radius: 10px; padding: 10px 18px; min-width: 120px; }}
.kpi b {{ display: block; font-size: 26px; }}
table.tbl {{ border-collapse: collapse; background: white; font-size: 13px; margin: 8px 0; }}
table.tbl th {{ background: {accent}; color: white; padding: 6px 8px; text-align: left; }}
table.tbl td {{ padding: 5px 8px; border-bottom: 1px solid #dde3ee; }}
table.tbl tr:nth-child(even) td {{ background: #f2f5fb; }}
figure {{ background: white; border: 1px solid #dde3ee; border-radius: 8px; padding: 10px; margin: 14px 0; }}
figure img {{ max-width: 100%; }} figcaption {{ font-weight: bold; color: {accent}; }}
.muted {{ color: #777; }} .wide {{ overflow-x: auto; }}
</style></head><body>
<header>{logo_html}<div><h1>{esc(title)}</h1>
<p>Report for {what} &middot; {len(sessions)} video(s) &middot; generated {datetime.datetime.now():%Y-%m-%d %H:%M}</p></div></header>
<main>
<h2>Results at a glance</h2>
<div class="kpis"><div class="kpi"><b>{n_shots}</b>shots</div><div class="kpi"><b>{n_in}</b>IN</div>
<div class="kpi"><b>{100 * n_in / max(1, n_shots):.0f}%</b>IN rate</div>
<div class="kpi"><b>{len(sessions)}</b>videos</div></div>
<h3>Summary by {esc(group_name.lower())}</h3>
<p class="muted">opening = where the ball crossed the cell opening, inches up from its bottom edge; sd = standard
deviation (smaller = more consistent).</p>
<div class="wide">{_table(summary, index=True)}</div>
<h2>Charts</h2>{charts_html}
<h2>Method</h2><ul>{method_html}</ul>
<h2>Procedure</h2><ol>{proc_html}</ol>
<h2>Files used</h2><div class="wide">{_table(_files_table(sessions))}</div>
<h2>Settings</h2>{_table(_settings_table(settings))}
<h2>Every shot</h2><div class="wide">{_table(shots)}</div>
</main></body></html>"""
    path.write_text(page, encoding="utf-8")
