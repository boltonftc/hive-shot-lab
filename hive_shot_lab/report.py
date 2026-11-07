"""Tables for the GUI and CSV export."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import Session
from .settings import Settings


def shot_kind(burst_size: int, index: int) -> str:
    return "single" if burst_size <= 1 else f"burst #{index}"


def kind_order(kind: str) -> tuple[int, int]:
    """Sort key: single first, then burst #1, #2, ..."""
    return (0, 0) if kind == "single" else (1, int(kind.split("#")[1]))


def setup_label(s: Settings) -> str:
    parts = [s.robot] if s.robot else []
    if s.hood_deg:
        parts.append(f"hood {s.hood_deg:g} deg")
    if s.flywheel_rpm:
        parts.append(f"{s.flywheel_rpm:g} rpm")
    if s.flywheel_mass_g:
        parts.append(f"{s.flywheel_mass_g:g} g")
    return ", ".join(parts) or "setup not entered"


def shots_dataframe(ses: Session) -> pd.DataFrame:
    st = ses.settings
    rows = []
    for s in ses.shots:
        rows.append({
            "video": Path(ses.video).name, "setup": setup_label(st), "robot": st.robot or None,
            "hood_deg": st.hood_deg or None, "rpm": st.flywheel_rpm or None, "flywheel_g": st.flywheel_mass_g or None,
            "shot": s.shot_id, "burst": s.burst_id + 1, "index": s.index_in_burst, "burst_size": s.burst_size,
            "kind": shot_kind(s.burst_size, s.index_in_burst),
            "time_s": round(s.t0, 3), "v0_in_s": round(s.v0, 1), "v0_ft_s": round(s.v0 / 12, 2),
            "angle_deg": round(s.angle_deg, 1), "apex_in": round(s.apex_y, 1),
            "at_opening_in": None if s.cross_s is None else round(s.cross_s, 2),
            "result": s.result, "points": s.n_points, "fit_rms_in": round(s.fit_rms, 2),
            "drag_per_s": round(s.drag_k, 3), "gravity_check": round(s.accel / 386.09, 2),
        })
    return pd.DataFrame(rows)


def batch_dataframe(sessions: list[Session]) -> pd.DataFrame:
    frames = [shots_dataframe(s) for s in sessions if s.shots]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def group_order(group: str) -> tuple:
    """Sort key for 'single', 'burst #2', a setup label, or 'setup | burst #2'."""
    head, _, kind = group.rpartition(" | ")
    if not head and not (group == "single" or group.startswith("burst #")):
        return (group, (0, 0))
    return (head, kind_order(kind))


def summary_by_kind(df: pd.DataFrame) -> pd.DataFrame:
    return summary_by(df, "kind")


def summary_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per group (ball kind, robot setup, or both): how consistent, how fast, how often IN."""
    if df.empty:
        return df
    g = df.groupby(col)
    out = pd.DataFrame({
        "shots": g.size(),
        "in_rate_%": g["result"].apply(lambda r: round(100 * (r == "IN").mean(), 1)),
        "opening_mean_in": g["at_opening_in"].mean().round(2),
        "opening_sd_in": g["at_opening_in"].std().round(2),
        "v0_mean_ft_s": g["v0_ft_s"].mean().round(2),
        "v0_sd_ft_s": g["v0_ft_s"].std().round(2),
        "angle_mean": g["angle_deg"].mean().round(1),
        "angle_sd": g["angle_deg"].std().round(2),
    })
    return out.loc[sorted(out.index, key=group_order)]
