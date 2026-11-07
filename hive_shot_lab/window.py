"""Hive Shot Lab main window.

Top: header with the team logo and the main buttons. Left: batch (several videos compared together), setup steps
(video, tag size, turret, target, robot setup), with every other number under "Advanced settings".
Tabs: Video (scoreboard, frames with overlays, a flash when a shot reaches the cell), Charts, Shots table, Summary
(the whole batch when one is open).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtCore import QEasingCurve, QObject, QPropertyAnimation, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGraphicsOpacityEffect, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from . import theme
from .theme import APP_NAME, BLUE, NAVY, ORANGE, TEAM, GlowButton
from .calibrate import Calibration, calibrate, find_tags_in_video
from .batch import LABELS, REJUDGE, Batch, labels_from_filename, needs_reprocess, with_shared
from .detect import Detector
from .html_report import write_report
from .pipeline import Session, open_video, process_video
from .report import batch_dataframe, kind_order, setup_label, shot_kind, summary_by
from .settings import Settings

RESULT_BGR = {"IN": (70, 200, 70), "LOW": (0, 150, 255), "HIGH": (255, 140, 40), "SHORT": (60, 60, 230), "?": (200, 200, 200)}
RESULT_QT = {k: QColor(b[2], b[1], b[0]) for k, b in RESULT_BGR.items()}
KIND_COLORS = {"single": "#333333", "burst #1": "tab:blue", "burst #2": "tab:green",
               "burst #3": "tab:orange", "burst #4": "tab:red"}
CHARTS = [("paths", "Paths (side view)"), ("crossing", "Crossing at the cell"), ("speed", "Launch speed"),
          ("in_rate", "IN rate")]
# chart filter lists: (key, title, share of the panel height)
FILTERS = [("group", "Key: data sets", 3), ("rpm", "Filter: flywheel rpm", 2), ("mass", "Filter: flywheel mass", 2)]
RESULT_COLORS = {"IN": "#2DBE4E", "LOW": "#FF8A00", "HIGH": BLUE, "SHORT": "#E03030"}
FLASH_TEXT = {"IN": "IN!  🐝", "LOW": "LOW", "HIGH": "HIGH", "SHORT": "SHORT"}
NAME_EXAMPLE = "robot1_2571rpm_152g"


def kind_color(kind: str) -> str:
    return KIND_COLORS.get(kind, "tab:purple")


def pt(p, dx: float = 0, dy: float = 0) -> tuple[int, int]:
    """OpenCV drawing wants plain Python ints."""
    return int(round(p[0] + dx)), int(round(p[1] + dy))


def outlined_text(img: np.ndarray, text: str, p: tuple[int, int], fs: float, col, th: int) -> None:
    cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, p, cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)


def copy_settings(s: Settings) -> Settings:
    return Settings.from_dict(json.loads(s.to_json()))


def gravity_check_text(ses: Session) -> str:
    return (f"Gravity check (not applied): the shots lean {ses.level_fix_deg:+.1f} deg from the tag's level "
            f"({ses.level_tracks} shots)")


# ====================================================================== settings panel
# attribute: (label, min, max, decimals, tooltip)
FIELDS = {
    "tag_size_in": ("AprilTag black square (in)", 0.5, 24, 3,
                    "Edge to edge of the black square. DECODE field tags: 6.5 in (165.1 mm)"),
    "tag_id": ("Tag id (-1 = biggest in view)", -1, 586, 0, "Pick a specific tag when more than one is in view"),
    "ball_diameter_in": ("Diameter (in)", 1, 10, 2, "Sets the IN margin and the ball-size filter"),
    "shooter_y": ("Turret height above floor (in)", 0, 100, 2, "Height of the turret center you clicked"),
    "level_adjust_deg": ("Extra level adjustment (deg)", -45, 45, 2,
                         "Added to the tilt read from the AprilTag's edges. Normally 0 - use it only if the tag stands crooked"),
    "cell_bottom_x": ("Distance to opening (in)", -100, 400, 2,
                      "Horizontal distance from the turret center to the bottom edge of the opening"),
    "cell_bottom_y": ("Opening bottom height (in)", 0, 200, 2, "Height of the bottom edge of the opening above the floor"),
    "cell_open_in": ("Opening size along lean (in)", 1, 60, 2, ""),
    "cell_lean_deg": ("Lean back from vertical (deg)", -80, 80, 1, ""),
    "zone_x_min": ("x from", -200, 400, 1, ""),
    "zone_x_max": ("x to", -200, 400, 1, ""),
    "zone_y_min": ("height from", -50, 300, 1, ""),
    "zone_y_max": ("height to", -50, 300, 1, ""),
    "fps_override": ("Frame rate override (0 = from file)", 0, 2000, 2,
                     "Normally 0: the gravity check fixes slow-mo files saved at playback speed"),
    "burst_gap_s": ("Burst gap (s)", 0.02, 10, 2, "Launches closer together than this are one burst"),
    "gate_in": ("Match distance (in)", 1, 100, 1, "How far a ball may be from where the tracker expected it"),
    "min_track_points": ("Min points per shot", 3, 500, 0, ""),
    "max_missed_frames": ("Max missed frames", 0, 100, 0, "Frames a ball may be hidden before its track ends"),
    "min_travel_in": ("Min travel (in)", 0, 200, 1, "Tracks that move less than this are ignored"),
    "max_extrapolate_s": ("Extend path past last sighting (s)", 0, 3, 2, ""),
}
# robot setup table rows: (attribute, parameter shown to the user)
SETUP_ROWS = [("robot", "Robot"), ("hood_deg", "Hood angle (deg)"), ("flywheel_rpm", "Flywheel speed (rpm)"),
              ("flywheel_mass_g", "Flywheel mass (g)")]
ADVANCED = [
    ("AprilTag", ["tag_id"]),
    ("Ball", ["ball_diameter_in"]),
    ("Camera", ["level_adjust_deg"]),
    ("Hive cell opening shape (side view)", ["cell_open_in", "cell_lean_deg"]),
    ("Where to look for flying balls (in)", ["zone_x_min", "zone_x_max", "zone_y_min", "zone_y_max"]),
    ("Tracking", ["fps_override", "burst_gap_s", "gate_in", "min_track_points", "max_missed_frames",
                  "min_travel_in", "max_extrapolate_s"]),
]
ADVANCED_COUNTED = [a for _, attrs in ADVANCED for a in attrs] + ["hsv_low", "hsv_high"]


def hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color:#666666;")
    return lbl


class SettingsPanel(QWidget):
    changed = Signal()
    open_video = Signal()
    pick_turret = Signal()
    process = Signal()
    advanced_toggled = Signal(bool)
    new_batch = Signal()
    open_batch = Signal()
    add_to_batch = Signal()
    remove_from_batch = Signal()
    batch_process = Signal()
    fill_setup = Signal()
    batch_video_clicked = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._loading = False
        self._turret_px: tuple[float, float] | None = None
        self.boxes: dict[str, QDoubleSpinBox | QSpinBox] = {}
        lay = QVBoxLayout(self)

        box = QGroupBox("Batch (several videos together)")
        bl = QVBoxLayout(box)
        self.batch_label = QLabel()
        self.batch_label.setWordWrap(True)
        bl.addWidget(self.batch_label)
        self.batch_list = QListWidget()
        self.batch_list.setMaximumHeight(130)
        self.batch_list.itemClicked.connect(lambda it: self.batch_video_clicked.emit(it.data(Qt.ItemDataRole.UserRole)))
        bl.addWidget(self.batch_list)
        row = QHBoxLayout()
        for text, sig, tip in (("New...", self.new_batch, "Start a new batch file"),
                               ("Open...", self.open_batch, "Open a saved batch"),
                               ("Add video...", self.add_to_batch, "Add a video to the batch and open it"),
                               ("Remove", self.remove_from_batch, "Take the selected video out of the batch (no files deleted)")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(sig)
            row.addWidget(b)
            if text == "Add video...":
                self.add_btn = b
            if text == "Remove":
                self.remove_btn = b
        bl.addLayout(row)
        self.batch_process_btn = QPushButton("Batch process videos...")
        self.batch_process_btn.setToolTip("Pick many videos filmed with the same camera setup; each is processed and added "
                                          f"to the batch. Names like {NAME_EXAMPLE} fill in robot, rpm and mass.")
        self.batch_process_btn.clicked.connect(self.batch_process)
        bl.addWidget(self.batch_process_btn)
        lay.addWidget(box)

        form = self._group(lay, "1. Video")
        b = QPushButton("Open video...")
        b.clicked.connect(self.open_video)
        form.addRow(b)
        form.addRow(hint("Side view, slo-mo, on a tripod. One AprilTag standing in the plane the ball flies in."))

        form = self._group(lay, "2. Scale")
        self._spin(form, "tag_size_in")

        form = self._group(lay, "3. Turret")
        row = QHBoxLayout()
        pick = QPushButton("Pick turret on video")
        pick.setToolTip("Then click the center of the turret in the video")
        pick.clicked.connect(self.pick_turret)
        self.turret_label = QLabel()
        row.addWidget(pick)
        row.addWidget(self.turret_label, 1)
        form.addRow(row)
        self.shoot_right = QCheckBox("Shots fly toward the right of the picture")
        self.shoot_right.toggled.connect(self._emit)
        form.addRow(self.shoot_right)
        self._spin(form, "shooter_y")

        form = self._group(lay, "4. Target (virtual hive cell)")
        self._spin(form, "cell_bottom_x")
        self._spin(form, "cell_bottom_y")
        form.addRow(hint("Changing only the target? Press Re-judge - it's instant."))

        box = QGroupBox("5. Robot setup in this video")
        vl = QVBoxLayout(box)
        self.setup_table = QTableWidget(len(SETUP_ROWS), 2)
        self.setup_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.setup_table.verticalHeader().setVisible(False)
        self.setup_table.horizontalHeader().setStretchLastSection(True)
        self.setup_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for r, (_attr, label) in enumerate(SETUP_ROWS):
            p = QTableWidgetItem(label)
            p.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.setup_table.setItem(r, 0, p)
            self.setup_table.setItem(r, 1, QTableWidgetItem(""))
        self.setup_table.resizeColumnToContents(0)
        self.setup_table.resizeRowsToContents()
        self.setup_table.setFixedHeight(self.setup_table.horizontalHeader().sizeHint().height()
                                        + sum(self.setup_table.rowHeight(r) for r in range(len(SETUP_ROWS)))
                                        + 2 * self.setup_table.frameWidth() + 2)
        self.setup_table.itemChanged.connect(self._setup_edited)
        vl.addWidget(self.setup_table)
        fill = QPushButton("Fill from file name")
        fill.setToolTip(f"Read robot / rpm / mass from a name like {NAME_EXAMPLE} (optional _30deg for the hood)")
        fill.clicked.connect(self.fill_setup)
        vl.addWidget(fill)
        vl.addWidget(hint(f"Filled in from the file name (like {NAME_EXAMPLE}). Double-click a value to change it. "
                          "These are labels for comparing videos: no need to re-process after changing them."))
        lay.addWidget(box)

        self.process_btn = QPushButton("Process video  (F5)")
        self.process_btn.setStyleSheet("font-weight:bold; padding:8px;")
        self.process_btn.clicked.connect(self.process)
        lay.addWidget(self.process_btn)

        self.adv_btn = QToolButton()
        self.adv_btn.setCheckable(True)
        self.adv_btn.setAutoRaise(True)
        self.adv_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.adv_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adv_btn.toggled.connect(self._toggle_advanced)
        lay.addWidget(self.adv_btn)
        self.adv = QWidget()
        al = QVBoxLayout(self.adv)
        al.setContentsMargins(0, 0, 0, 0)
        for title, attrs in ADVANCED:
            form = self._group(al, title)
            for attr in attrs:
                self._spin(form, attr)
            if title == "Ball":
                self.hsv: dict[str, list[QSpinBox]] = {}
                for key, label in (("hsv_low", "Yellow low  H / S / V"), ("hsv_high", "Yellow high H / S / V")):
                    row = QHBoxLayout()
                    spins = []
                    for top in (180, 255, 255):
                        sp = QSpinBox()
                        sp.setRange(0, top)
                        sp.valueChanged.connect(self._emit)
                        row.addWidget(sp)
                        spins.append(sp)
                    self.hsv[key] = spins
                    form.addRow(label, row)
                form.addRow(hint("Hover over the video to read H/S/V; turn on 'Yellow mask' to check."))
        row = QHBoxLayout()
        for text, fn in (("Load settings...", self._load), ("Save settings...", self._save), ("Defaults", self._defaults)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        al.addLayout(row)
        self.adv.setVisible(False)
        lay.addWidget(self.adv)
        lay.addStretch(1)
        self.set(Settings())

    def _group(self, parent: QVBoxLayout, title: str) -> QFormLayout:
        box = QGroupBox(title)
        parent.addWidget(box)
        return QFormLayout(box)

    def _spin(self, form: QFormLayout, attr: str) -> None:
        label, lo, hi, dec, tip = FIELDS[attr]
        if isinstance(getattr(Settings(), attr), int):
            w = QSpinBox()
            w.setRange(int(lo), int(hi))
        else:
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setSingleStep(10 ** -min(dec, 1) if dec else 1)
        if tip:
            w.setToolTip(tip)
        w.valueChanged.connect(self._emit)
        self.boxes[attr] = w
        form.addRow(label, w)

    def _toggle_advanced(self, on: bool) -> None:
        self.adv.setVisible(on)
        self.adv_btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.advanced_toggled.emit(on)

    def _update_advanced_label(self) -> None:
        s, d = self.get(), Settings()
        n = sum(getattr(s, a) != getattr(d, a) for a in ADVANCED_COUNTED)
        self.adv_btn.setText("Advanced settings" + (f"  ({n} changed from defaults)" if n else ""))

    # ------------------------------------------------------------------ get / set
    def set(self, s: Settings) -> None:
        self._loading = True
        for attr, w in self.boxes.items():
            w.setValue(getattr(s, attr))
        for key, spins in self.hsv.items():
            for sp, v in zip(spins, getattr(s, key)):
                sp.setValue(int(v))
        self.shoot_right.setChecked(s.shoot_right)
        self.set_setup(s)
        self._set_turret_label(s.turret_px)
        self._loading = False
        self._update_advanced_label()
        self.changed.emit()

    def get(self) -> Settings:
        s = Settings()
        for attr, w in self.boxes.items():
            setattr(s, attr, w.value())
        for key, spins in self.hsv.items():
            setattr(s, key, tuple(sp.value() for sp in spins))
        s.shoot_right = self.shoot_right.isChecked()
        for r, (attr, _label) in enumerate(SETUP_ROWS):
            setattr(s, attr, self._setup_value(r))
        s.turret_px = self._turret_px
        return s

    def set_turret(self, u: float, v: float) -> None:
        self._set_turret_label((round(u, 1), round(v, 1)))
        self._emit()

    def set_setup(self, s: Settings) -> None:
        """Show the robot setup labels of s in the table (blank = not entered)."""
        self.setup_table.blockSignals(True)
        for r, (attr, _label) in enumerate(SETUP_ROWS):
            v = getattr(s, attr)
            self.setup_table.item(r, 1).setText(v if isinstance(v, str) else (f"{v:g}" if v else ""))
        self.setup_table.blockSignals(False)

    def _setup_value(self, row: int) -> str | float:
        text = self.setup_table.item(row, 1).text().strip()
        if SETUP_ROWS[row][0] == "robot":
            return text
        try:
            return max(0.0, float(text)) if text else 0.0
        except ValueError:
            return 0.0

    def _setup_edited(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        r = item.row()
        if SETUP_ROWS[r][0] != "robot" and item.text().strip():
            try:
                float(item.text())
            except ValueError:
                QMessageBox.warning(self, "Robot setup", f"'{item.text()}' isn't a number.")
                self.setup_table.blockSignals(True)
                item.setText("")
                self.setup_table.blockSignals(False)
        self._emit()

    # ------------------------------------------------------------------ internals
    def _set_turret_label(self, px: tuple[float, float] | None) -> None:
        self._turret_px = px
        self.turret_label.setText(f"at pixel ({px[0]:.0f}, {px[1]:.0f})" if px else
                                  "<span style='color:red'>not set</span>")

    def _emit(self, *_a) -> None:
        if not self._loading:
            self._update_advanced_label()
            self.changed.emit()

    def _load(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "Settings (*.json)")
        if fn:
            try:
                self.set(Settings.load(Path(fn)))
            except (OSError, ValueError, TypeError, KeyError) as e:
                QMessageBox.warning(self, "Load settings", f"Couldn't read {fn}:\n{e}")

    def _save(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(self, "Save settings", "settings.json", "Settings (*.json)")
        if fn:
            self.get().save(Path(fn))

    def _defaults(self) -> None:
        self.set(Settings())


# ====================================================================== frame view
class FrameView(QLabel):
    """Shows a video frame scaled to fit; reports the image pixel under the mouse and clicks."""
    hovered = Signal(float, float)
    clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.setStyleSheet("background:#202020; color:#bbbbbb;")
        self.setText("Open a video (Ctrl+O)")
        self._img: QImage | None = None

    def set_frame(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._show()

    def _show(self) -> None:
        if self._img is not None:
            self.setPixmap(QPixmap.fromImage(self._img).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e) -> None:  # noqa: N802 (Qt name)
        super().resizeEvent(e)
        self._show()

    def _to_image(self, e) -> tuple[float, float] | None:
        pm = self.pixmap()
        if self._img is None or pm is None or pm.isNull():
            return None
        ox, oy = (self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2
        u = (e.position().x() - ox) * self._img.width() / pm.width()
        v = (e.position().y() - oy) * self._img.height() / pm.height()
        return (u, v) if 0 <= u < self._img.width() and 0 <= v < self._img.height() else None

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        p = self._to_image(e)
        self.hovered.emit(*(p or (-1.0, -1.0)))

    def mousePressEvent(self, e) -> None:  # noqa: N802
        p = self._to_image(e)
        if p and e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(*p)


# ====================================================================== background processing
class Worker(QObject):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, settings: Settings) -> None:
        super().__init__()
        self.path, self.settings, self.cancel = path, settings, False

    def run(self) -> None:
        try:
            ses = process_video(self.path, self.settings, lambda i, n: self.progress.emit(i, n), lambda: self.cancel)
            self.done.emit(ses)
        except Exception as e:  # shown to the user; the app keeps running
            self.failed.emit(f"{type(e).__name__}: {e}")


class BatchWorker(QObject):
    """Processes several videos one after another; saves each one's results next to it."""
    progress = Signal(int, int)
    video_started = Signal(int, int, str)
    video_done = Signal(object, object)          # video path, Session
    video_failed = Signal(object, str)
    all_done = Signal()

    def __init__(self, jobs: list[tuple[Path, Settings]]) -> None:
        super().__init__()
        self.jobs, self.cancel = jobs, False

    def run(self) -> None:
        n = len(self.jobs)
        for k, (path, s) in enumerate(self.jobs):
            if self.cancel:
                break
            self.video_started.emit(k, n, path.name)
            try:
                ses = process_video(path, s, lambda i, m, k=k: self.progress.emit(int(1000 * (k + i / max(1, m)) / n), 1000),
                                    lambda: self.cancel)
            except Exception as e:  # one bad video shouldn't stop the rest
                self.video_failed.emit(path, f"{type(e).__name__}: {e}")
                continue
            if self.cancel:
                break
            ses.save(path.with_suffix(".session.json"))
            ses.settings.save(path.with_suffix(".settings.json"))
            self.video_done.emit(path, ses)
        self.all_done.emit()


# ====================================================================== main window
class Badge(QLabel):
    """A scoreboard light: big number, small label, team-colored."""

    def __init__(self, label: str, color: str) -> None:
        super().__init__()
        self.label, self.color = label, color
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(96)
        self.set_value("-")

    def set_value(self, value: str) -> None:
        self.setText(f"<div style='font-size:22px; font-weight:900'>{value}</div>"
                     f"<div style='font-size:11px; font-weight:bold'>{self.label}</div>")
        self.setStyleSheet(f"background:{self.color}; color:white; border:2px solid #000; border-radius:10px;"
                           f"padding:4px 10px;")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} · {TEAM}")
        self.setWindowIcon(theme.app_icon())
        self._cross_frames: list[tuple[int, str]] = []
        self.video: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.n_frames = 0
        self.fps_file = 240.0
        self.frame_idx = 0
        self._next_read = -1
        self.frame: np.ndarray | None = None
        self.found: dict[int, np.ndarray] = {}
        self.calib: Calibration | None = None
        self.detector: Detector | None = None
        self.problem = ""
        self.picking = False
        self.session: Session | None = None
        self.settings = Settings()
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.batch: Batch | None = None
        self.batch_sessions: dict[Path, Session | None] = {}     # by resolved video path

        self._build_ui()
        self.debounce = QTimer(self, singleShot=True, interval=300)
        self.debounce.timeout.connect(self._settings_applied)
        self.panel.changed.connect(self.debounce.start)
        self.panel.pick_turret.connect(self._start_pick)
        self.panel.open_video.connect(self.choose_video)
        self.panel.process.connect(self.start_processing)
        self.panel.advanced_toggled.connect(self._show_tuning_overlays)
        self.panel.new_batch.connect(self.new_batch)
        self.panel.open_batch.connect(self.choose_batch)
        self.panel.add_to_batch.connect(self.add_to_batch)
        self.panel.remove_from_batch.connect(self.remove_from_batch)
        self.panel.batch_process.connect(self.batch_process)
        self.panel.fill_setup.connect(self.fill_setup_from_name)
        self.panel.batch_video_clicked.connect(lambda p: self.open_video(Path(p)))
        self._show_tuning_overlays(False)
        self._refresh_batch_list()
        self.play_timer = QTimer(self, interval=33)
        self.play_timer.timeout.connect(self._play_tick)
        self._settings_applied()

    # ------------------------------------------------------------------ layout
    def _build_ui(self) -> None:
        def action(text: str, fn, key: str | None = None) -> QAction:
            a = QAction(text, self)
            a.triggered.connect(fn)
            if key:
                a.setShortcut(QKeySequence(key))
            self.addAction(a)
            return a

        action("Open video", self.choose_video, "Ctrl+O")
        self.a_process = action("Process", self.start_processing, "F5")
        self.a_rejudge = action("Re-judge", self.rejudge)

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_header())

        self.panel = SettingsPanel()
        self.panel.process_btn.setObjectName("primary")
        self.panel.process_btn.setStyleSheet("")
        theme.add_glow(self.panel.process_btn, BLUE)
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_video_page(), "🎥  Video")
        self.tabs.addTab(self._build_charts_tab(), "📈  Charts")
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._table_jump)
        self.tabs.addTab(self.table, "📋  Shots")
        summ = QWidget()
        sl = QVBoxLayout(summ)
        self.summary_text = QLabel()
        self.summary_text.setTextFormat(Qt.TextFormat.RichText)
        self.summary_text.setWordWrap(True)
        self.summary_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary_table = QTableWidget()
        self.summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sl.addWidget(self.summary_text)
        sl.addWidget(self.summary_table, 1)
        self.tabs.addTab(summ, "🏆  Summary")

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        rl.addWidget(split, 1)
        self.setCentralWidget(root)

        sb = self.statusBar()
        self.hover_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        sb.addPermanentWidget(self.hover_label)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cancel_btn)

        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.step(-1))
        QShortcut(QKeySequence("Shift+Right"), self, lambda: self.step(10))
        QShortcut(QKeySequence("Shift+Left"), self, lambda: self.step(-10))
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.toggle_play)

    def _build_header(self) -> QWidget:
        head = QWidget()
        head.setObjectName("header")
        head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        hl = QHBoxLayout(head)
        hl.setContentsMargins(14, 8, 14, 8)
        logo = QLabel()
        logo.setPixmap(theme.logo(theme.LOGO_DARK_BG, 70))
        hl.addWidget(logo)
        titles = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        sub = QLabel(f"{TEAM}  ·  BIOBUZZ pollen shot tracker")
        sub.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(sub)
        hl.addLayout(titles)
        hl.addStretch(1)
        self.btn_open = GlowButton("📂  Open video")
        self.btn_turret = GlowButton("🎯  Pick turret")
        self.btn_process = GlowButton("▶  Process", glow="#7FB2FF", primary=True)
        self.btn_batch = GlowButton("🗂  Batch process")
        self.btn_rejudge = GlowButton("⚖  Re-judge")
        self.btn_export = GlowButton("💾  Export CSV")
        self.btn_report = GlowButton("📄  Report")
        self.btn_about = GlowButton("ℹ")
        for b, fn, tip in ((self.btn_open, self.choose_video, "Open a side-view slow-mo video of shots (Ctrl+O)"),
                           (self.btn_turret, self._start_pick, "Then click the center of the turret in the video"),
                           (self.btn_process, self.start_processing, "Find and track every shot (F5)"),
                           (self.btn_batch, self.batch_process,
                            f"Process many videos at once. Names like {NAME_EXAMPLE} fill in robot, rpm and mass"),
                           (self.btn_rejudge, self.rejudge, "Re-check every shot against the hive cell - instant"),
                           (self.btn_export, self.export_csv, "Save the shots table and summary as CSV"),
                           (self.btn_report, self.export_report,
                            "Save a web-page report: method, files used, tables and charts"),
                           (self.btn_about, lambda: theme.about(self), "About Hive Shot Lab, credits and license")):
            b.setToolTip(tip)
            b.setMinimumHeight(46)
            b.clicked.connect(fn)
            hl.addWidget(b)
        return head

    def _build_video_page(self) -> QWidget:
        page = QWidget()
        pl = QVBoxLayout(page)
        board = QHBoxLayout()
        self.badges = {
            "shots": Badge("SHOTS", NAVY),
            "IN": Badge("IN", RESULT_COLORS["IN"]),
            "LOW": Badge("LOW", RESULT_COLORS["LOW"]),
            "HIGH": Badge("HIGH", RESULT_COLORS["HIGH"]),
            "SHORT": Badge("SHORT", RESULT_COLORS["SHORT"]),
            "rate": Badge("IN RATE", ORANGE),
        }
        for b in self.badges.values():
            board.addWidget(b)
        board.addStretch(1)
        pl.addLayout(board)
        pl.addWidget(self._build_video_tab(), 1)

        self.flash = QLabel(self.view)
        self.flash.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flash.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.flash.hide()
        self._flash_fx = QGraphicsOpacityEffect(self.flash)
        self.flash.setGraphicsEffect(self._flash_fx)
        self._flash_anim = QPropertyAnimation(self._flash_fx, b"opacity", self)
        self._flash_anim.setDuration(1100)
        self._flash_anim.setStartValue(1.0)
        self._flash_anim.setEndValue(0.0)
        self._flash_anim.setEasingCurve(QEasingCurve.Type.InQuad)
        self._flash_anim.finished.connect(self.flash.hide)
        return page

    def _build_charts_tab(self) -> QWidget:
        """'Compare by' plus one sub-tab per chart, each with its own zoom / save toolbar."""
        charts = QWidget()
        cl = QVBoxLayout(charts)
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Compare by:</b>"))
        self.compare = QComboBox()
        self.compare.addItems(["Ball kind (single / burst #n)", "Robot setup", "Robot setup + ball kind", "Video"])
        self.compare.setToolTip("How shots are grouped (and colored) in the charts and the summary")
        self.compare.currentIndexChanged.connect(lambda _i: self.refresh_results())
        row.addWidget(self.compare)
        row.addStretch(1)
        cl.addLayout(row)
        self.chart_tabs = QTabWidget()
        self.figs: dict[str, Figure] = {}
        self.canvases: dict[str, FigureCanvasQTAgg] = {}
        for key, title in CHARTS:
            fig = Figure(figsize=(10, 6), layout="constrained")
            canvas = FigureCanvasQTAgg(fig)
            page = QWidget()
            pl = QVBoxLayout(page)
            pl.addWidget(NavigationToolbar2QT(canvas, page))
            pl.addWidget(canvas, 1)
            self.chart_tabs.addTab(page, title)
            self.figs[key], self.canvases[key] = fig, canvas
        self._hidden: dict[str, set[str]] = {f: set() for f, _t, _c in FILTERS}
        self._legend_groups = False          # the Qt key lists the data sets; the report's images need them drawn in
        self._chart_df, self._chart_sessions = None, []
        body = QHBoxLayout()
        body.addWidget(self.chart_tabs, 1)
        key = QWidget()
        kl = QVBoxLayout(key)
        kl.setContentsMargins(0, 0, 0, 0)
        self.filter_lists: dict[str, QListWidget] = {}
        for f, title, stretch in FILTERS:
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            lw = QListWidget()
            lw.setWordWrap(True)
            lw.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            lw.setToolTip("Untick to hide these shots in every chart (and in the exported report)")
            lw.itemChanged.connect(lambda it, f=f: self._filter_toggled(f, it))
            bl.addWidget(lw, 1)
            brow = QHBoxLayout()
            for text, show in (("All", True), ("None", False)):
                b = QPushButton(text)
                b.clicked.connect(lambda _=False, f=f, show=show: self._filter_set_all(f, show))
                brow.addWidget(b)
            bl.addLayout(brow)
            kl.addWidget(box, stretch)
            self.filter_lists[f] = lw
        self.chart_key = self.filter_lists["group"]
        kl.addWidget(hint("A shot shows only if it's ticked in all three lists. Numbers = shots each choice would add."))
        key.setFixedWidth(290)
        body.addWidget(key)
        cl.addLayout(body, 1)
        return charts

    # ------------------------------------------------------------------ chart key + filters (show / hide shots)
    @staticmethod
    def _filter_keys(df, f: str):
        """Per shot: its data set, rpm or flywheel mass as the text shown in that filter list."""
        if f == "group":
            return df["group"]
        col = {"rpm": "rpm", "mass": "flywheel_g"}[f]
        unit = {"rpm": " rpm", "mass": " g"}[f]
        return df[col].map(lambda v: "not set" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:g}{unit}")

    def _visible(self, df, skip: str | None = None):
        """Which shots pass the filters (optionally ignoring one of them)."""
        mask = np.ones(len(df), bool)
        for f, _t, _s in FILTERS:
            if f != skip and self._hidden[f]:
                mask &= ~self._filter_keys(df, f).isin(self._hidden[f]).to_numpy()
        return mask

    def _update_chart_key(self, df) -> None:
        for f, lw in self.filter_lists.items():
            lw.blockSignals(True)
            lw.clear()
        if df is not None and not df.empty:
            order = self._group_order(df)
            colors = self._group_colors(order)
            for f, lw in self.filter_lists.items():
                keys = self._filter_keys(df, f)
                counts = keys[self._visible(df, skip=f)].value_counts()
                if f == "group":
                    values = order
                else:
                    num = {k: float(k.split()[0]) if k != "not set" else math.inf for k in keys.unique()}
                    values = sorted(num, key=num.get)
                for v in values:
                    it = QListWidgetItem(f"{v}  ({int(counts.get(v, 0))})")
                    if f == "group":
                        swatch = QPixmap(16, 16)
                        swatch.fill(QColor(to_hex(colors[v])))
                        it.setIcon(QIcon(swatch))
                    it.setData(Qt.ItemDataRole.UserRole, v)
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Unchecked if v in self._hidden[f] else Qt.CheckState.Checked)
                    lw.addItem(it)
        for lw in self.filter_lists.values():
            lw.blockSignals(False)

    def _filter_toggled(self, f: str, it: QListWidgetItem) -> None:
        v = it.data(Qt.ItemDataRole.UserRole)
        if it.checkState() == Qt.CheckState.Checked:
            self._hidden[f].discard(v)
        else:
            self._hidden[f].add(v)
        QTimer.singleShot(0, self._redraw_filtered)          # rebuild the lists after this click is handled

    def _filter_set_all(self, f: str, show: bool) -> None:
        lw = self.filter_lists[f]
        self._hidden[f] = set() if show else {lw.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lw.count())}
        self._redraw_filtered()

    def _redraw_filtered(self) -> None:
        self._update_chart_key(self._chart_df)
        self._draw_charts(self._chart_df, self._chart_sessions)

    def _build_video_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

        split = QSplitter()
        self.view = FrameView()
        self.view.hovered.connect(self._hover)
        self.view.clicked.connect(self._frame_clicked)
        split.addWidget(self.view)
        self.shot_list = QListWidget()
        self.shot_list.setFont(QFont("Consolas", 9))
        self.shot_list.itemClicked.connect(lambda it: self.jump_to_shot(it.data(Qt.ItemDataRole.UserRole)))
        split.addWidget(self.shot_list)
        split.setStretchFactor(0, 1)
        split.setSizes([1000, 330])
        lay.addWidget(split, 1)

        row = QHBoxLayout()
        for text, d in (("<< 10", -10), ("< 1", -1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, d=d: self.step(d))
            row.addWidget(b)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        row.addWidget(self.play_btn)
        for text, d in (("1 >", 1), ("10 >>", 10)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, d=d: self.step(d))
            row.addWidget(b)
        self.speed = QComboBox()
        self.speed.addItems(["1 frame/tick", "2 frames/tick", "4 frames/tick", "8 frames/tick", "16 frames/tick"])
        self.speed.setToolTip("Playback speed (about 30 ticks per second)")
        row.addWidget(self.speed)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        row.addWidget(self.slider, 1)
        self.frame_label = QLabel()
        self.frame_label.setMinimumWidth(190)
        row.addWidget(self.frame_label)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.overlay: dict[str, QCheckBox] = {}
        for key, text, on, tip in (
                ("grid", "Grid (1 ft)", True, "World grid every 12 in plus the floor line - check it lines up with reality"),
                ("tags", "Tags", True, "Detected AprilTags (green = used for scale)"),
                ("cell", "Cell + turret", True, "Virtual hive cell opening (magenta) and turret (cyan)"),
                ("paths", "Shot paths", True, "Tracked points and fitted path of shots near this frame"),
                ("all_paths", "All paths", False, "Draw every shot's fitted path at once"),
                ("dets", "Live detections", False, "Run the ball finder on this frame (yellow circles)"),
                ("mask", "Yellow mask", False, "Show which pixels count as 'yellow' (magenta) - for tuning H/S/V")):
            cb = QCheckBox(text)
            cb.setChecked(on)
            cb.setToolTip(tip)
            cb.toggled.connect(self.render)
            self.overlay[key] = cb
            row.addWidget(cb)
        row.addStretch(1)
        lay.addLayout(row)
        return w

    def _show_tuning_overlays(self, on: bool) -> None:
        for key in ("dets", "mask"):
            self.overlay[key].setVisible(on)
            if not on:
                self.overlay[key].setChecked(False)

    # ------------------------------------------------------------------ video
    def choose_video(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Open video", str(self._videos_dir()),
                                            "Video (*.mov *.mp4 *.m4v *.avi *.mkv);;All files (*)")
        if fn:
            self.open_video(Path(fn))

    def _videos_dir(self) -> Path:
        if self.video:
            return self.video.parent
        d = Path.home() / "Videos"
        return d if d.is_dir() else Path.home()

    def open_video(self, path: Path, load_saved: bool = True) -> None:
        if self.thread is not None:
            QMessageBox.information(self, "Open video", "Wait for processing to finish (or Cancel it) first.")
            return
        if self.debounce.isActive():                 # save the last video's pending edits before switching
            self.debounce.stop()
            self._settings_applied()
        cap = open_video(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Open video", f"Can't open {path}")
            return
        if self.cap:
            self.cap.release()
        self.stop_play()
        self.cap, self.video = cap, path
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_file = cap.get(cv2.CAP_PROP_FPS) or 240.0
        self.setWindowTitle(f"{APP_NAME} · {TEAM}  -  {path.name}")
        self.statusBar().showMessage("Looking for AprilTags...")
        QApplication.processEvents()
        self.found = find_tags_in_video(cap)
        self._next_read = -1
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.session = None
        if load_saved:
            sp, ss = self._settings_path(), self._session_path()
            s = None
            try:
                if ss.exists():
                    self.session = Session.load(ss)
                    s = self.session.settings
                if sp.exists():
                    s = Settings.load(sp)                    # newest: saved whenever this video's settings change
            except (OSError, ValueError, TypeError, KeyError) as e:
                QMessageBox.warning(self, "Saved results", f"Couldn't read the saved results next to the video:\n{e}")
                self.session = None
            if s is None:                    # a new video: keep the general settings, not the last video's turret/setup
                s = self.panel.get()
                s.turret_px = None
                for a in LABELS:
                    setattr(s, a, getattr(Settings(), a))
            if not any(getattr(s, a) for a in LABELS):
                for k, val in labels_from_filename(path.stem).items():
                    setattr(s, k, val)
            self.panel.set(s)
        if self._in_batch(path):
            self.panel.set(with_shared(self.panel.get(), self.batch.settings))
            self.batch_sessions[path.resolve()] = self.session      # same object as on screen, so edits show in the batch
            self._refresh_batch_list()
        self._settings_applied()
        self.refresh_results()
        self.seek(0)
        msg = f"Opened {path.name}: {self.n_frames} frames at {self.fps_file:.2f} fps; tags seen: {sorted(self.found) or 'none'}"
        if self.session:
            msg += f"; loaded saved results ({len(self.session.shots)} shots)"
        self.statusBar().showMessage(msg)

    def _session_path(self) -> Path:
        return self.video.with_suffix(".session.json")

    def _settings_path(self) -> Path:
        return self.video.with_suffix(".settings.json")

    def read_frame(self, f: int) -> np.ndarray | None:
        if self.cap is None:
            return None
        if f != self._next_read:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = self.cap.read()
        self._next_read = f + 1 if ok else -1
        return frame if ok else None

    def seek(self, f: int) -> None:
        if self.cap is None:
            return
        before = self.frame_idx
        f = int(np.clip(f, 0, max(0, self.n_frames - 1)))
        frame = self.read_frame(f)
        if frame is None:
            return
        self.frame_idx, self.frame = f, frame
        if self.slider.value() != f:
            self.slider.blockSignals(True)
            self.slider.setValue(f)
            self.slider.blockSignals(False)
        fps = self.session.fps if self.session else self.fps_file
        self.frame_label.setText(f"frame {f} / {self.n_frames - 1}   t = {f / fps:.3f} s")
        self.render()
        if 0 < f - before <= 40:                         # playing or stepping forward, not a big jump
            for cf, result in self._cross_frames:
                if before < cf <= f:
                    self.celebrate(result)

    def celebrate(self, result: str) -> None:
        color = RESULT_COLORS.get(result, ORANGE)
        c = QColor(color)
        size = 96 if result == "IN" else 64
        self.flash.setText(FLASH_TEXT.get(result, result))
        self.flash.setStyleSheet(f"color:white; font-size:{size}px; font-weight:900; "
                                 f"background:rgba({c.red()},{c.green()},{c.blue()},150); "
                                 f"border:6px solid {color}; border-radius:24px;")
        self.flash.setGeometry(self.view.rect().adjusted(40, 40, -40, -40))
        self.flash.show()
        self.flash.raise_()
        self._flash_anim.stop()
        self._flash_anim.start()

    def step(self, d: int) -> None:
        self.seek(self.frame_idx + d)

    def toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.stop_play()
        elif self.cap is not None:
            self.play_timer.start()
            self.play_btn.setText("Pause")

    def stop_play(self) -> None:
        self.play_timer.stop()
        self.play_btn.setText("Play")

    def _play_tick(self) -> None:
        if self.frame_idx >= self.n_frames - 1:
            self.stop_play()
            return
        self.step(2 ** self.speed.currentIndex())

    # ------------------------------------------------------------------ settings -> calibration
    def _start_pick(self) -> None:
        if self.frame is None:
            QMessageBox.information(self, "Pick turret", "Open a video first.")
            return
        self.picking = True
        self.tabs.setCurrentIndex(0)
        self.view.setCursor(Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage("Click the center of the turret in the video")

    def _frame_clicked(self, u: float, v: float) -> None:
        if not self.picking:
            return
        self.picking = False
        self.view.unsetCursor()
        self.panel.set_turret(u, v)
        self.debounce.stop()
        self._settings_applied()
        self.statusBar().showMessage(f"Turret set at pixel ({u:.0f}, {v:.0f})")

    def _settings_applied(self) -> None:
        self.settings = self.panel.get()
        if self.video and self.video.exists():
            self.settings.save(self._settings_path())      # each video remembers its own turret click and setup
        self._sync_setup_labels()
        self.calib, self.detector, self.problem = None, None, ""
        if self.cap is not None:
            try:
                self.calib = calibrate(self.found, self.settings)
                w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.detector = Detector(self.settings, self.calib, w, h)
            except (ValueError, cv2.error) as e:
                self.problem = str(e)
        self._update_info()
        if self._in_batch(self.video):
            self._refresh_batch_list()
        self.render()

    def _update_info(self) -> None:
        if self.cap is None:
            self.info.setText("Open a video to start.")
            return
        parts = []
        if self.calib:
            c = self.calib
            col = "green" if c.shape_err_pct < 3 else "darkorange"
            parts.append(f"<span style='color:{col}'>Scale from tag {c.tags_used[0]}: {c.px_per_in:.2f} px/in "
                         f"(tag looks {c.shape_err_pct:.1f}% out of square)</span>")
            if c.shape_err_pct >= 3:
                parts.append("<span style='color:darkorange'>camera may not be square to the tag, or the tag isn't flat</span>")
            parts.append(f"level from the tag (tilted {c.tag_roll_deg:+.2f} deg in the picture)"
                         + (f" plus {self.settings.level_adjust_deg:+.2f} adjustment" if self.settings.level_adjust_deg else ""))
        if self.problem:
            parts.append(f"<span style='color:red'><b>Problem:</b> {self.problem}</span>")
        if self.session and json.loads(self.session.settings.to_json()) != json.loads(self.settings.to_json()):
            parts.append("<i>Settings differ from the processed results: Re-judge (cell / ball / burst gap) "
                         "or Process again (tag / turret / zone / colour / tracking).</i>")
        self.info.setText(" &nbsp;|&nbsp; ".join(parts))

    # ------------------------------------------------------------------ drawing
    def render(self) -> None:
        if self.frame is None:
            return
        img = self.frame.copy()
        h = img.shape[0]
        th = max(1, round(h / 540))
        fs = 0.4 * th
        s = self.settings
        calib = self.calib or (self.session.calib if self.session else None)

        def P(pts) -> np.ndarray:                                   # world inches -> int image pixels
            return np.round(calib.to_image(np.asarray(pts, float))).astype(np.int32)

        if self.overlay["mask"].isChecked() and self.detector:
            x0, y0, x1, y1 = self.detector.roi
            m = cv2.resize(self.detector.yellow_mask(self.frame), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
            sub = img[y0:y1, x0:x1]
            sub[m] = (sub[m] * 0.25 + np.array([255, 0, 255]) * 0.75).astype(np.uint8)

        if calib is not None:
            zone = [[s.zone_x_min, s.zone_y_min], [s.zone_x_max, s.zone_y_min], [s.zone_x_max, s.zone_y_max],
                    [s.zone_x_min, s.zone_y_max]]
            cv2.polylines(img, [P(zone)], True, (0, 220, 255), th)
            if self.overlay["grid"].isChecked():
                for x in range(math.ceil(s.zone_x_min / 12) * 12, int(s.zone_x_max) + 1, 12):
                    a, b = P([[x, max(0, s.zone_y_min)], [x, s.zone_y_max]])
                    cv2.line(img, pt(a), pt(b), (255, 255, 255), th)
                    outlined_text(img, f'{x}"', pt(a, 3, -4), fs, (255, 255, 255), 1)
                for y in range(max(12, math.ceil(s.zone_y_min / 12) * 12), int(s.zone_y_max) + 1, 12):
                    a, b = P([[s.zone_x_min, y], [s.zone_x_max, y]])
                    cv2.line(img, pt(a), pt(b), (255, 255, 255), th)
                    outlined_text(img, f'{y}"', pt(a, 3, -4), fs, (255, 255, 255), 1)
                a, b = P([[s.zone_x_min, 0], [s.zone_x_max, 0]])
                cv2.line(img, pt(a), pt(b), (0, 0, 0), 3 * th)
                cv2.line(img, pt(a), pt(b), (255, 255, 255), th)
                outlined_text(img, "floor", pt(a, 3, -6), fs, (255, 255, 255), 1)
            if self.overlay["cell"].isChecked():
                r_px = max(2, int(s.ball_diameter_in / 2 * calib.px_per_in))
                (bot, top) = P(list(s.cell_points()))
                cv2.line(img, pt(bot), pt(top), (255, 0, 255), 2 * th)
                for p in (bot, top):
                    cv2.circle(img, pt(p), 2 * th, (255, 0, 255), -1)
                cv2.putText(img, "cell", pt(top, 6, 0), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.4, (255, 0, 255), th)
                tp = P([[0.0, s.shooter_y]])[0]
                cv2.circle(img, pt(tp), r_px, (255, 255, 0), th)
                cv2.putText(img, "turret", pt(tp, r_px + 4, 0), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.4, (255, 255, 0), th)

        if self.overlay["tags"].isChecked():
            used = set(self.calib.tags_used) if self.calib else set()
            for tid, c in self.found.items():
                col = (0, 220, 0) if tid in used else (0, 160, 255)
                cv2.polylines(img, [np.round(c).astype(np.int32)], True, col, th)
                cv2.putText(img, f"tag {tid}", pt(c[0], 0, -6), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.4, col, th)

        if self.session and self.overlay["paths"].isChecked():
            self._draw_shots(img, th, fs)

        if self.overlay["dets"].isChecked() and self.detector and calib:
            r_px = max(3, int(s.ball_diameter_in / 2 * calib.px_per_in))
            for u, v, _ in self.detector.detect(self.frame):
                cv2.circle(img, (int(u), int(v)), r_px + 3, (0, 255, 255), th)

        self.view.set_frame(img)

    def _draw_shots(self, img: np.ndarray, th: int, fs: float) -> None:
        ses, f = self.session, self.frame_idx
        calib, s = ses.calib, ses.settings
        all_paths = self.overlay["all_paths"].isChecked()
        r_px = max(3, int(s.ball_diameter_in / 2 * calib.px_per_in))
        shot_tracks = set()
        for sh in ses.shots:
            shot_tracks.add(sh.track_id)
            span = s.max_extrapolate_s * sh.fps
            near = sh.frame0 - 0.25 * sh.fps <= f <= sh.frame1 + span
            if not (near or all_paths):
                continue
            col = RESULT_BGR.get(sh.result, (200, 200, 200))
            t = np.linspace(0, (sh.frame1 - sh.frame0) / sh.fps + s.max_extrapolate_s, 120)
            x, y = sh.position(t)
            keep = y > 0
            x, y = x[keep], y[keep]
            pts = np.round(calib.to_image(np.c_[x, y])).astype(np.int32)
            cv2.polylines(img, [pts], False, col, th if near else 1)
            if not near:
                continue
            tr = ses.tracks.get(sh.track_id)
            if tr:
                for fr, u, v in zip(tr.frames, tr.u, tr.v):
                    if fr <= f:
                        cv2.circle(img, (int(u), int(v)), th + 1, col, -1)
            if sh.frame0 <= f:
                tt = (f - sh.frame0) / sh.fps
                bx, by = sh.position(np.array([tt]))
                bp = np.round(calib.to_image(np.c_[bx, by])[0]).astype(int)
                cv2.circle(img, pt(bp), r_px, col, th)
            lp = pts[len(pts) // 2]
            label = f"#{sh.shot_id} {shot_kind(sh.burst_size, sh.index_in_burst)} {sh.result} {sh.v0 / 12:.1f} ft/s"
            outlined_text(img, label, pt(lp, 8, -8 - 18 * th * (sh.index_in_burst - 1)), fs * 1.3, col, th)
        for tid, tr in ses.tracks.items():                       # tracks that didn't look like shots
            if tid in shot_tracks or not tr.frames or not (tr.frames[0] <= f <= tr.frames[-1] + 30):
                continue
            last = None
            for fr, u, v in zip(tr.frames, tr.u, tr.v):
                if fr <= f:
                    cv2.circle(img, (int(u), int(v)), th + 3, (0, 0, 0), -1)
                    cv2.circle(img, (int(u), int(v)), th + 1, (255, 255, 255), -1)
                    last = (int(u), int(v))
            if last:
                outlined_text(img, f"track {tid}: not a shot", pt(last, 10, -10), fs * 1.3, (255, 255, 255), th)

    def _hover(self, u: float, v: float) -> None:
        if u < 0 or self.frame is None:
            self.hover_label.setText("")
            return
        b, g, r = self.frame[int(v), int(u)]
        hh, ss, vv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
        text = f"px ({u:.0f}, {v:.0f})  HSV ({hh}, {ss}, {vv})"
        calib = self.calib or (self.session.calib if self.session else None)
        if calib is not None:
            x, y = calib.to_world(np.array([[u, v]]))[0]
            text = f"x = {x:.1f} in, height = {y:.1f} in   |   " + text
        self.hover_label.setText(text)

    # ------------------------------------------------------------------ processing
    def start_processing(self) -> None:
        if self.video is None:
            QMessageBox.information(self, "Process", "Open a video first.")
            return
        if self.thread is not None:
            return
        if self.calib is None:
            QMessageBox.warning(self, "Process", f"Can't process yet:\n{self.problem}")
            return
        s = copy_settings(self.settings)
        s.save(self._settings_path())
        self.stop_play()
        self.thread = QThread(self)
        self.worker = Worker(self.video, s)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.a_process.setEnabled(False)
        self.panel.process_btn.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.btn_process.pulse(True)
        self.statusBar().showMessage("Processing... (you can keep looking at the video)")
        self.thread.start()

    def cancel_processing(self) -> None:
        if self.worker:
            self.worker.cancel = True

    def _on_progress(self, i: int, n: int) -> None:
        self.progress.setValue(int(100 * i / max(1, n)))

    def _finish_thread(self) -> None:
        self.thread.quit()
        self.thread.wait()
        self.thread, self.worker = None, None
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.a_process.setEnabled(True)
        self.panel.process_btn.setEnabled(True)
        for b in (self.btn_process, self.btn_batch):
            b.pulse(False)
            b.setEnabled(True)

    def _on_done(self, ses: Session) -> None:
        cancelled = self.worker.cancel
        self._finish_thread()
        self.session = ses
        self.panel.set(ses.settings)                 # picks up the level correction measured from gravity
        self.debounce.stop()
        self._settings_applied()
        if not cancelled:
            ses.save(self._session_path())
            ses.settings.save(self._settings_path())
            if self.batch and not self._in_batch(self.video) and QMessageBox.question(
                    self, "Batch", f"Add {self.video.name} to the batch '{self.batch.name}'?") == QMessageBox.StandardButton.Yes:
                self.batch.add(self.video)
            if self._in_batch(self.video):
                self.batch_sessions[self.video.resolve()] = ses
                self.batch.settings = with_shared(self.batch.settings, ses.settings)
                self.batch.save()
        self._refresh_batch_list()
        self.refresh_results()
        self._update_info()
        self.render()
        msg = f"{len(ses.shots)} shots found ({ses.rejected} other tracks ignored) in {ses.seconds:.0f} s"
        if cancelled:
            msg = "Cancelled - partial results (not saved): " + msg
        if ses.level_tracks:
            msg += f".  {gravity_check_text(ses)}"
        if abs(ses.fps / ses.fps_file - 1) > 0.01:
            msg += f".  Gravity check: the file says {ses.fps_file:.0f} fps but the balls fall like {ses.fps:.0f} fps - using {ses.fps:.0f}"
        self.statusBar().showMessage(msg)

    def _on_failed(self, err: str) -> None:
        self._finish_thread()
        self.statusBar().showMessage("Processing failed")
        QMessageBox.warning(self, "Processing failed", err)

    def rejudge(self) -> None:
        if not self.session and not self.batch:
            return
        if self.session:
            self.session.settings = with_shared(self.session.settings, self.settings, REJUDGE + LABELS)
            self.session.reevaluate()
            if self.video and self.video.exists():
                self.session.save(self._session_path())
        if self.batch:
            for v, ses in self.batch_sessions.items():
                if ses is None or ses is self.session:
                    continue
                ses.settings = with_shared(ses.settings, self.settings, REJUDGE)
                ses.reevaluate()
                ses.save(self.batch.session_path(v))
            self.batch.settings = with_shared(self.batch.settings, self.settings, REJUDGE)
            self.batch.save()
            self._refresh_batch_list()
        self.refresh_results()
        self._update_info()
        self.render()
        where = f" in all {len(self.batch.videos)} videos of the batch" if self.batch else ""
        self.statusBar().showMessage(f"Re-judged every shot{where} against the current cell / ball / burst settings")

    def fill_setup_from_name(self) -> None:
        if not self.video:
            return
        found = labels_from_filename(self.video.stem)
        if not found:
            QMessageBox.information(self, "Robot setup", f"Can't read a setup from '{self.video.stem}'.\n\nName videos "
                                    f"like {NAME_EXAMPLE} (robot, rpm, flywheel grams; optional _30deg for the hood).")
            return
        s = self.panel.get()
        for k, val in found.items():
            setattr(s, k, val)
        self.panel.set(s)

    def _sync_setup_labels(self) -> None:
        """Hood / rpm / flywheel mass are just labels: copy them into the results right away, no re-processing."""
        ses = self.session
        if not ses or all(getattr(ses.settings, a) == getattr(self.settings, a) for a in LABELS):
            return
        for a in LABELS:
            setattr(ses.settings, a, getattr(self.settings, a))
        if self.video and self.video.exists():
            ses.save(self._session_path())
            ses.settings.save(self._settings_path())
        self._refresh_batch_list()
        self.refresh_results()

    # ------------------------------------------------------------------ batch
    def _in_batch(self, video: Path | None) -> bool:
        return bool(self.batch and video and video.resolve() in self.batch.videos)

    def new_batch(self) -> None:
        fn, _ = QFileDialog.getSaveFileName(self, "New batch", str(self._videos_dir() / "my_test.batch.json"),
                                            "Batch (*.batch.json)")
        if not fn:
            return
        path = Path(fn)
        if not path.name.endswith(".batch.json"):
            path = path.with_name(path.name.removesuffix(".json") + ".batch.json")
        self.batch = Batch(path, copy_settings(self.settings))
        self.batch_sessions = {}
        if self.video and QMessageBox.question(
                self, "New batch", f"Add the open video {self.video.name} to the batch?") == QMessageBox.StandardButton.Yes:
            self.batch.add(self.video)
            self.batch_sessions[self.video.resolve()] = self.session
        self.batch.save()
        self._refresh_batch_list()
        self.refresh_results()
        self.statusBar().showMessage(f"New batch {path.name}. Add videos with 'Add video...'")

    def choose_batch(self) -> None:
        fn, _ = QFileDialog.getOpenFileName(self, "Open batch", str(self._videos_dir()), "Batch (*.batch.json)")
        if not fn:
            return
        try:
            batch = Batch.load(Path(fn))
        except (OSError, ValueError, TypeError, KeyError) as e:
            QMessageBox.warning(self, "Open batch", f"Couldn't read {fn}:\n{e}")
            return
        self.batch = batch
        self.batch_sessions = {}
        bad = []
        for v in batch.videos:
            try:
                self.batch_sessions[v] = batch.load_session(v)
            except (OSError, ValueError, TypeError, KeyError):
                self.batch_sessions[v] = None
                bad.append(v.name)
        first = next((v for v in batch.videos if v.exists()), None)
        if first:
            self.open_video(first)
        self._refresh_batch_list()
        self.refresh_results()
        if bad:
            QMessageBox.warning(self, "Open batch", "Couldn't read the saved results of: " + ", ".join(bad))

    def add_to_batch(self) -> None:
        if not self.batch:
            return
        fns, _ = QFileDialog.getOpenFileNames(self, "Add videos to the batch", str(self._videos_dir()),
                                              "Video (*.mov *.mp4 *.m4v *.avi *.mkv);;All files (*)")
        if not fns:
            return
        for fn in fns:
            v = Path(fn).resolve()
            if self.batch.add(v):
                try:
                    self.batch_sessions[v] = self.batch.load_session(v)
                except (OSError, ValueError, TypeError, KeyError):
                    self.batch_sessions[v] = None
        self.batch.save()
        self.open_video(Path(fns[-1]))
        self._refresh_batch_list()
        self.refresh_results()

    def remove_from_batch(self) -> None:
        it = self.panel.batch_list.currentItem()
        if not self.batch or it is None:
            return
        v = Path(it.data(Qt.ItemDataRole.UserRole))
        self.batch.videos.remove(v)
        self.batch_sessions.pop(v, None)
        self.batch.save()
        self._refresh_batch_list()
        self.refresh_results()
        self.statusBar().showMessage(f"Removed {v.name} from the batch (its files are untouched)")

    def _video_settings(self, v: Path) -> Settings:
        """This video's own settings (turret click, robot setup, ...) plus the batch's shared ones."""
        if self.video and v == self.video.resolve():
            s = copy_settings(self.settings)
        else:
            sp, ses = v.with_suffix(".settings.json"), self.batch_sessions.get(v)
            try:
                s = Settings.load(sp) if sp.exists() else (copy_settings(ses.settings) if ses else Settings())
            except (OSError, ValueError, TypeError, KeyError):
                s = Settings()
            if not any(getattr(s, a) for a in LABELS):
                for k, val in labels_from_filename(v.stem).items():
                    setattr(s, k, val)
        return with_shared(s, self.batch.settings) if self.batch else s

    def batch_process(self) -> None:
        """Process every video of the batch that isn't processed yet (each with its own turret click)."""
        if self.thread is not None:
            return
        if not self.batch:
            start = self.video.parent if self.video else self._videos_dir()
            fns, _ = QFileDialog.getOpenFileNames(self, "Batch process: pick the videos", str(start),
                                                  "Video (*.mov *.mp4 *.m4v *.avi *.mkv);;All files (*)")
            if not fns:
                return
            paths = [Path(f).resolve() for f in fns]
            default = paths[0].parent / f"{paths[0].parent.name}.batch.json"
            fn, _ = QFileDialog.getSaveFileName(self, "Save the batch as", str(default), "Batch (*.batch.json)")
            if not fn:
                return
            bp = Path(fn)
            if not bp.name.endswith(".batch.json"):
                bp = bp.with_name(bp.name.removesuffix(".json") + ".batch.json")
            self.batch, self.batch_sessions = Batch(bp, copy_settings(self.settings)), {}
            for v in paths:
                self.batch.add(v)
                self.batch_sessions[v] = self.batch.load_session(v) if Batch.session_path(v).exists() else None
            self.batch.save()
            self.open_video(paths[0])
            self._refresh_batch_list()
            self.refresh_results()
        self.batch.settings = with_shared(self.batch.settings, self.settings)     # the panel's shared settings
        self.batch.save()
        self._refresh_batch_list()

        videos = [v for v in self.batch.videos if v.exists()]
        todo = [v for v in videos if self.batch_sessions.get(v) is None
                or needs_reprocess(self.batch_sessions[v].settings, self._video_settings(v))]
        if not todo:
            if QMessageBox.question(self, "Batch process", "Every video in the batch is already processed.\n\n"
                                    "Process them all again?") != QMessageBox.StandardButton.Yes:
                return
            todo = videos
        missing = [v for v in todo if self._video_settings(v).turret_px is None]
        if missing:
            ready = [v for v in todo if v not in missing]
            msg = ("The turret isn't set in:\n\n   " + "\n   ".join(v.name for v in missing) +
                   "\n\nClick each one in the batch list, press 'Pick turret on video' and click the turret center. "
                   "A check mark appears next to the video when it's set.")
            if not ready:
                QMessageBox.warning(self, "Batch process", msg)
                return
            if QMessageBox.question(self, "Batch process", msg + f"\n\nProcess the {len(ready)} video(s) that are "
                                    "ready now?") != QMessageBox.StandardButton.Yes:
                return
            todo = ready
        jobs = [(v, self._video_settings(v)) for v in todo]

        self.stop_play()
        self._batch_failed: list[str] = []
        self._batch_shots = 0
        self._batch_done = 0
        self.thread = QThread(self)
        self.worker = BatchWorker(jobs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.video_started.connect(self._batch_video_started)
        self.worker.video_done.connect(self._batch_video_done)
        self.worker.video_failed.connect(self._batch_video_failed)
        self.worker.all_done.connect(self._batch_all_done)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.a_process.setEnabled(False)
        self.panel.process_btn.setEnabled(False)
        self.panel.batch_process_btn.setEnabled(False)
        for b in (self.btn_process, self.btn_batch):
            b.setEnabled(False)
        self.btn_batch.pulse(True)
        self.statusBar().showMessage(f"Batch processing {len(jobs)} videos...")
        self.thread.start()

    def _batch_video_started(self, k: int, n: int, name: str) -> None:
        self.statusBar().showMessage(f"Batch: video {k + 1} of {n}: {name}  ({self._batch_shots} shots so far)")

    def _batch_video_done(self, path: Path, ses: Session) -> None:
        self._batch_shots += len(ses.shots)
        self._batch_done += 1
        self.batch_sessions[path] = ses
        if self.video and self.video.resolve() == path:
            self.session = ses
        self._refresh_batch_list()

    def _batch_video_failed(self, path: Path, err: str) -> None:
        self._batch_failed.append(f"{path.name}: {err}")

    def _batch_all_done(self) -> None:
        cancelled = self.worker.cancel
        self._finish_thread()
        self.panel.batch_process_btn.setEnabled(True)
        setups = {setup_label(s.settings) for s in self._sessions()}
        if len(setups) > 1 and self.compare.currentIndex() == 0:
            self.compare.setCurrentIndex(1)                     # color the charts by robot setup
        if self.video is None and self.batch and self.batch.videos:
            self.open_video(self.batch.videos[0])
        self._refresh_batch_list()
        self.refresh_results()
        msg = (f"Batch done: {self._batch_done} video(s) processed, {self._batch_shots} shots found. The batch now has "
               f"{sum(len(s.shots) for s in self._sessions())} shots in {len(self._sessions())} videos")
        if cancelled:
            msg = "Batch cancelled. " + msg
        self.statusBar().showMessage(msg + (f"; {len(self._batch_failed)} failed" if self._batch_failed else ""))
        if self._batch_failed:
            QMessageBox.warning(self, "Batch process", "These videos couldn't be processed:\n\n" + "\n".join(self._batch_failed))

    def _refresh_batch_list(self) -> None:
        p = self.panel
        p.batch_list.clear()
        has = self.batch is not None
        for w in (p.add_btn, p.remove_btn, p.batch_list):
            w.setVisible(has)
        if not has:
            p.batch_label.setText("No batch open: results are for the open video only. Press <b>New...</b> to collect "
                                  "several videos (e.g. different hood angles or flywheel speeds) into one comparison.")
            return
        sessions = [s for s in self.batch_sessions.values() if s]
        p.batch_label.setText(f"<b>{self.batch.name}</b>: {len(self.batch.videos)} videos, "
                              f"{sum(len(s.shots) for s in sessions)} shots. Click a video to open it. "
                              "✔ = turret set, ✖ = turret still to pick.")
        current = self.video.resolve() if self.video else None
        for v in self.batch.videos:
            ses = self.batch_sessions.get(v)
            vs = self._video_settings(v)
            mark = "✔" if vs.turret_px is not None else "✖"
            col = None
            if not v.exists():
                text, col = f"{mark} {v.name}   video file missing", QColor("red")
            elif ses is None:
                text, col = f"{mark} {v.name}   not processed yet   {setup_label(vs)}", QColor("darkorange")
            else:
                n = len(ses.shots)
                text = f"{mark} {v.name}   {n} shot{'s' if n != 1 else ''}   {setup_label(ses.settings)}"
                if needs_reprocess(ses.settings, vs):
                    text, col = text + "   (settings changed: process again)", QColor("darkorange")
            if vs.turret_px is None and v.exists():
                col = QColor("#C00000")
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, str(v))
            if col is not None:
                it.setForeground(col)
            if v == current:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            p.batch_list.addItem(it)

    # ------------------------------------------------------------------ results
    def _sessions(self) -> list[Session]:
        """What the Charts / Shots / Summary tabs show: the whole batch, or just the open video."""
        if self.batch:
            return [s for v in self.batch.videos if (s := self.batch_sessions.get(v))]
        return [self.session] if self.session else []

    def _with_groups(self, df):
        if df.empty:
            return df
        mode = self.compare.currentIndex()
        df = df.copy()
        df["group"] = (df["kind"] if mode == 0 else df["setup"] if mode == 1
                       else df["setup"] + " | " + df["kind"] if mode == 2 else df["video"])

        def num(v) -> float:
            return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

        setup = [(str(r or ""), num(m), num(rpm), num(h))       # robot, then flywheel mass, rpm, hood
                 for r, m, rpm, h in zip(df["robot"], df["flywheel_g"], df["rpm"], df["hood_deg"])]
        kind = [kind_order(k) for k in df["kind"]]
        df["gkey"] = [(k,) if mode == 0 else (su,) if mode == 1 else (su, k) if mode == 2 else (v,)
                      for su, k, v in zip(setup, kind, df["video"])]
        return df

    @staticmethod
    def _group_order(df) -> list[str]:
        first = dict(zip(df["group"], df["gkey"]))
        return sorted(first, key=first.get)

    def _group_colors(self, order: list[str]) -> dict[str, object]:
        """Ball kind keeps its usual colors; otherwise each robot setup gets its own clearly different color."""
        mode = self.compare.currentIndex()
        if mode == 0:
            return {g: kind_color(g) for g in order}
        base = [g.rsplit(" | ", 1)[0] if mode == 2 else g for g in order]
        setups = list(dict.fromkeys(base))
        pal = colormaps["tab10" if len(setups) <= 10 else "tab20"]
        color = {su: pal(i % pal.N) for i, su in enumerate(setups)}
        return {g: color[b] for g, b in zip(order, base)}

    def refresh_results(self) -> None:
        self.shot_list.clear()
        ses = self.session
        if ses:
            for sh in ses.shots:
                kind = shot_kind(sh.burst_size, sh.index_in_burst)
                it = QListWidgetItem(f"#{sh.shot_id:<3} {kind:<9} {sh.result:<5} {sh.v0 / 12:5.1f} ft/s "
                                     f"{sh.angle_deg:4.1f} deg  t={sh.t0:7.2f}s")
                it.setData(Qt.ItemDataRole.UserRole, sh.shot_id)
                it.setForeground(RESULT_QT.get(sh.result, QColor("gray")))
                self.shot_list.addItem(it)
        sessions = self._sessions()
        df = self._with_groups(batch_dataframe(sessions))
        self._fill_table(self.table, df.drop(columns=["group", "gkey"], errors="ignore"))
        self._fill_summary(df, sessions)
        self._chart_df, self._chart_sessions = df, sessions
        self._update_chart_key(df)
        self._draw_charts(df, sessions)

        shots = [sh for s in sessions for sh in s.shots]
        counts = {k: sum(sh.result == k for sh in shots) for k in ("IN", "LOW", "HIGH", "SHORT")}
        self.badges["shots"].set_value(str(len(shots)) if shots else "-")
        for k, n in counts.items():
            self.badges[k].set_value(str(n) if shots else "-")
        self.badges["rate"].set_value(f"{100 * counts['IN'] / len(shots):.0f}%" if shots else "-")
        self._cross_frames = self._crossing_frames(ses)

    @staticmethod
    def _crossing_frames(ses: Session | None) -> list[tuple[int, str]]:
        """The video frame where each shot reaches the hive cell opening, and its result."""
        out = []
        if not ses:
            return out
        for sh in ses.shots:
            if sh.cross_x is None:
                continue
            t = np.linspace(0, (sh.frame1 - sh.frame0) / sh.fps + ses.settings.max_extrapolate_s, 400)
            x, _ = sh.position(t)
            i = int(np.argmax(x >= sh.cross_x))
            out.append((int(round(sh.frame0 + t[i] * sh.fps)), sh.result))
        return out

    def jump_to_shot(self, shot_id: int) -> None:
        sh = next((s for s in self.session.shots if s.shot_id == shot_id), None) if self.session else None
        if sh is None:
            return
        self.tabs.setCurrentIndex(0)
        if self.cap is None:
            self.statusBar().showMessage("The video for this session isn't open.")
            return
        self.stop_play()
        self.seek(max(0, sh.frame0 - 5))

    def _table_jump(self, row: int, _col: int) -> None:
        headers = [self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())]
        if "shot" not in headers:
            return
        shot_id = int(self.table.item(row, headers.index("shot")).text())
        name = self.table.item(row, headers.index("video")).text() if "video" in headers else ""
        if self.batch and name and (self.video is None or self.video.name != name):
            v = self.batch.find(name)
            if v is None or not v.exists():
                self.statusBar().showMessage(f"Video {name} not found")
                return
            self.open_video(v)
            if self.video is None or self.video.name != name:
                return
        self.jump_to_shot(shot_id)

    @staticmethod
    def _fill_table(table: QTableWidget, df, index_name: str | None = None) -> None:
        table.setSortingEnabled(False)
        table.clear()
        if df is None or df.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        cols = ([index_name] if index_name else []) + list(df.columns)
        table.setColumnCount(len(cols))
        table.setRowCount(len(df))
        table.setHorizontalHeaderLabels(cols)
        for r, (idx, row) in enumerate(df.iterrows()):
            vals = ([idx] if index_name else []) + list(row.values)
            for c, v in enumerate(vals):
                it = QTableWidgetItem()
                if isinstance(v, (int, float, np.integer, np.floating)) and not (isinstance(v, float) and math.isnan(v)):
                    it.setData(Qt.ItemDataRole.DisplayRole, float(v) if isinstance(v, (float, np.floating)) else int(v))
                else:
                    it.setText("" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
                if cols[c] == "result":
                    it.setForeground(RESULT_QT.get(str(v), QColor("gray")))
                table.setItem(r, c, it)
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    @staticmethod
    def _cell_text(s: Settings) -> str:
        r = s.ball_diameter_in / 2
        return (f"Virtual cell: opening bottom at x = {s.cell_bottom_x:g} in, height {s.cell_bottom_y:g} in, "
                f"{s.cell_open_in:g} in tall leaning back {s.cell_lean_deg:g} deg. A ball is IN when its center crosses "
                f"between {r:.1f} and {s.cell_open_in - r:.1f} in from the bottom edge.")

    def _fill_summary(self, df, sessions: list[Session]) -> None:
        group_name = self.compare.currentText()
        table_intro = (f"<br><b>By {group_name.lower()}</b> (opening = where the ball crossed the opening, inches up "
                       "from its bottom edge; sd = standard deviation, smaller is more consistent):")
        n_shots = len(df)
        n_in = int((df["result"] == "IN").sum()) if not df.empty else 0
        totals = f"<b>{n_shots} shots, {n_in} IN ({100 * n_in / max(1, n_shots):.0f}%)</b>"
        if self.batch:
            lines = [f"<b>Batch {self.batch.name}</b>: {len(self.batch.videos)} videos. {totals}"]
            for v in self.batch.videos:
                ses = self.batch_sessions.get(v)
                if ses is None:
                    lines.append(f"&nbsp; {v.name}: not processed yet")
                    continue
                c = ses.calib
                lines.append(f"&nbsp; {v.name}: <b>{setup_label(ses.settings)}</b> - {len(ses.shots)} shots; "
                             f"{c.px_per_in:.2f} px/in, {ses.fps:.0f} fps, tag tilted {c.tag_roll_deg:+.2f} deg")
            lines += [self._cell_text(self.batch.settings), table_intro]
        elif sessions:
            ses = sessions[0]
            c, s = ses.calib, ses.settings
            lines = [
                f"<b>{Path(ses.video).name}</b> &nbsp; {ses.width}x{ses.height}, {ses.n_frames} frames. "
                f"Robot setup: {setup_label(s)}",
                f"Frame rate: file {ses.fps_file:.2f} fps, used <b>{ses.fps:.2f}</b> fps"
                + (" <span style='color:darkorange'>(corrected by the gravity check)</span>" if abs(ses.fps / ses.fps_file - 1) > 0.01 else ""),
                f"Scale and level: tag {c.tags_used[0]}, {c.px_per_in:.2f} px/in (tag {c.shape_err_pct:.1f}% out of square), "
                f"tilted {c.tag_roll_deg:+.2f} deg in the picture"
                + (f", plus {s.level_adjust_deg:+.2f} deg adjustment" if s.level_adjust_deg else "") + ". "
                + (gravity_check_text(ses) + "." if ses.level_tracks else ""),
                self._cell_text(s),
                f"{totals}; {ses.rejected} other moving yellow things ignored; processing took {ses.seconds:.0f} s.",
                table_intro,
            ]
        else:
            self.summary_text.setText("Process a video (or open a saved session or batch) to see the summary.")
            self._fill_table(self.summary_table, None)
            return
        self.summary_text.setText("<br>".join(lines))
        if df.empty:
            summ = None
        else:
            summ = summary_by(df, "group")
            summ = summ.loc[[g for g in self._group_order(df) if g in summ.index]]
        self._fill_table(self.summary_table, summ, group_name)

    def _draw_charts(self, df, sessions: list[Session]) -> None:
        for fig in self.figs.values():
            fig.clear()
        if df is None or df.empty:
            for fig in self.figs.values():
                fig.text(0.5, 0.5, "Process a video to see charts", ha="center", va="center", fontsize=14, color="gray")
            for c in self.canvases.values():
                c.draw_idle()
            return
        s = self.batch.settings if self.batch else sessions[0].settings
        r = s.ball_diameter_in / 2
        kinds = self._group_order(df)
        mode = self.compare.currentIndex()
        by_kind = mode == 0
        colors = self._group_colors(kinds)                # from all groups, so colors don't change when hiding some
        visible = self._visible(df)
        df_all, df = df, df[visible]
        shown = set(df["group"])
        kinds = [k for k in kinds if k in shown]
        if df.empty:
            for fig in self.figs.values():
                fig.text(0.5, 0.5, "Nothing left to show - tick more in the key / filters", ha="center", va="center",
                         fontsize=14, color="gray")
            for c in self.canvases.values():
                c.draw_idle()
            return
        (bx, by), (tx, ty) = s.cell_points()
        colored = "" if by_kind else f"  -  colored by {self.compare.currentText().lower()}"
        counts = df["group"].value_counts()
        legend: dict[str, list] = {}                      # one entry per color: name -> [color, shots]
        for k in kinds:
            name = k.rsplit(" | ", 1)[0] if mode == 2 else k
            legend.setdefault(name, [colors[k], 0])[1] += int(counts[k])
        group_handles = ([Line2D([], [], color=c, lw=3, label=f"{name}  ({n} shots)") for name, (c, n) in legend.items()]
                         if self._legend_groups else [])

        fig = self.figs["paths"]
        ax = fig.add_subplot()
        pairs = [(ses, sh) for ses in sessions for sh in ses.shots]          # same order as the rows of df_all
        for (ses, sh), g, vis in zip(pairs, df_all["group"], visible):
            if not vis:
                continue
            t_seen = (sh.frame1 - sh.frame0) / sh.fps
            x, y = sh.position(np.linspace(0, t_seen, 200))
            ax.plot(x, y, color=colors[g], lw=1.2, alpha=0.85)
            x, y = sh.position(np.linspace(t_seen, t_seen + ses.settings.max_extrapolate_s, 80))
            keep = y > 0
            ax.plot(x[keep], y[keep], color=colors[g], lw=1, alpha=0.6, ls=":")
        ux, uy = (tx - bx) / s.cell_open_in, (ty - by) / s.cell_open_in
        ax.plot([bx, tx], [by, ty], color="magenta", lw=3, label="cell opening")
        ax.plot([bx + r * ux, tx - r * ux], [by + r * uy, ty - r * uy], color="limegreen", lw=6, alpha=0.5,
                label="IN window (ball center)")
        ax.plot([0.0], [s.shooter_y], "o", color="c", ms=8, label="turret")
        ax.axhline(0, color="k", lw=1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x from turret (in)")
        ax.set_ylabel("height (in)")
        ax.set_title("Every shot, side view (dotted = past the last sighting)" + colored)
        ax.grid(alpha=0.3)
        extra = ax.get_legend_handles_labels()[0] + [Line2D([], [], color="gray", ls=":", label="past last sighting")]
        fig.legend(handles=group_handles + extra, loc="outside right upper", fontsize=8)

        fig = self.figs["crossing"]
        ax = fig.add_subplot()
        vals = df["at_opening_in"].dropna()
        lo, hi = min(-2.0, vals.min() if len(vals) else 0), max(s.cell_open_in + 2, vals.max() if len(vals) else 0)
        bins = np.arange(math.floor(lo * 2) / 2, hi + 0.5, 0.5)
        ax.axvspan(r, s.cell_open_in - r, color="limegreen", alpha=0.2, label="IN window")
        for edge in (0, s.cell_open_in):
            ax.axvline(edge, color="magenta", lw=1)
        for k in kinds:
            v = df.loc[df["group"] == k, "at_opening_in"].dropna()
            if len(v):
                ax.hist(v, bins=bins, histtype="step", lw=2, color=colors[k],
                        label=f"{k} (n={len(v)})" if self._legend_groups else None)
        short = int((df["result"] == "SHORT").sum())
        ax.set_title("Where each ball crossed the cell opening" + (f"  ({short} SHORT not shown)" if short else "")
                     + colored)
        ax.set_xlabel("inches up the opening from its bottom edge  (<0 LOW, >opening size HIGH)")
        ax.set_ylabel("shots")
        fig.legend(loc="outside right upper", fontsize=8)

        fig = self.figs["speed"]
        ax = fig.add_subplot()
        data = [df.loc[df["group"] == k, "v0_ft_s"].values for k in kinds]
        ax.boxplot(data, tick_labels=kinds, showfliers=False)
        rng = np.random.default_rng(0)
        for i, (k, d) in enumerate(zip(kinds, data), start=1):
            ax.plot(i + rng.uniform(-0.12, 0.12, len(d)), d, "o", color=colors[k], alpha=0.7, ms=5)
        ax.set_title("Launch speed" + colored)
        ax.set_ylabel("ft/s")
        ax.grid(axis="y", alpha=0.3)
        if not by_kind:
            ax.tick_params(axis="x", labelrotation=20, labelsize=8)

        fig = self.figs["in_rate"]
        ax = fig.add_subplot()
        rates, labels = [], []
        for k in kinds:
            sub = df[df["group"] == k]
            n_in = int((sub["result"] == "IN").sum())
            rates.append(100 * n_in / len(sub))
            labels.append(f"{n_in}/{len(sub)}")
        bars = ax.bar(kinds, rates, color=[colors[k] for k in kinds])
        for b, lab in zip(bars, labels):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, lab, ha="center", fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("% IN")
        ax.set_title("IN rate" + colored)
        if not by_kind:
            ax.tick_params(axis="x", labelrotation=20, labelsize=8)

        for c in self.canvases.values():
            c.draw_idle()

    # ------------------------------------------------------------------ files
    def export_csv(self) -> None:
        sessions = self._sessions()
        if not sessions:
            QMessageBox.information(self, "Export", "Nothing to export yet - process a video first.")
            return
        default = (self.batch.path.with_name(self.batch.name + ".shots.csv") if self.batch
                   else Path(sessions[0].video).with_suffix(".shots.csv"))
        fn, _ = QFileDialog.getSaveFileName(self, "Export shots CSV", str(default), "CSV (*.csv)")
        if not fn:
            return
        df = self._with_groups(batch_dataframe(sessions))
        df.drop(columns=["group", "gkey"], errors="ignore").to_csv(fn, index=False)
        summary_path = Path(fn).with_name(Path(fn).stem + "_summary.csv")
        if not df.empty:
            summary_by(df, "group").to_csv(summary_path, index_label=self.compare.currentText())
        self.statusBar().showMessage(f"Saved {fn} and {summary_path.name}")

    def _report_branding(self) -> tuple[str, Path | None, str]:
        """Title, logo and accent colour of the HTML report."""
        return f"{APP_NAME} report  ·  {TEAM}", theme.LOGO_LIGHT_BG, BLUE

    def export_report(self, fn: str | bool | None = None) -> None:
        sessions = self._sessions()
        if not sessions:
            QMessageBox.information(self, "Report", "Nothing to report yet - process a video first.")
            return
        if not isinstance(fn, str):                       # menu actions / buttons pass a bool
            default = (self.batch.path.with_name(self.batch.name + "_report.html") if self.batch
                       else Path(sessions[0].video).with_suffix(".report.html"))
            fn, _ = QFileDialog.getSaveFileName(self, "Export HTML report", str(default), "Web page (*.html)")
            if not fn:
                return
        self.refresh_results()
        df = self._with_groups(batch_dataframe(sessions))
        summ = None
        if not df.empty:
            summ = summary_by(df, "group")
            summ = summ.loc[[g for g in self._group_order(df) if g in summ.index]]
            summ.index.name = self.compare.currentText()
        title, logo, accent = self._report_branding()
        self._legend_groups = True                       # images in the report need their own legend
        self._draw_charts(self._chart_df, self._chart_sessions)
        try:
            write_report(Path(fn), title=title, sessions=sessions, batch=self.batch,
                         settings=self.batch.settings if self.batch else sessions[0].settings,
                         shots=df.drop(columns=["group", "gkey"], errors="ignore"), summary=summ,
                         group_name=self.compare.currentText(),
                         charts=[(title_, self.figs[key]) for key, title_ in CHARTS], logo=logo, accent=accent)
        finally:
            self._legend_groups = False
            self._draw_charts(self._chart_df, self._chart_sessions)
        self.statusBar().showMessage(f"Saved the report {fn}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(fn)))

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.worker:
            self.worker.cancel = True
        if self.thread:
            self.thread.quit()
            self.thread.wait(5000)
        super().closeEvent(e)


if __name__ == "__main__":
    main()
