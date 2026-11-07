"""Hive Shot Lab look: 36563 Prime Symmetry colours, logos, buttons that press and glow, splash screen, About box."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEasingCurve, QRect, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QMessageBox, QPushButton, QSplashScreen, QWidget

from . import __version__

BRANDING = Path(__file__).resolve().parent / "branding"
LOGO_LIGHT_BG = BRANDING / "prime_symmetry_logo_light_bg.png"
LOGO_DARK_BG = BRANDING / "prime_symmetry_logo_dark_bg.png"
ICON = BRANDING / "prime_symmetry_icon_36563.png"

ORANGE = "#FCAB01"
ORANGE_DARK = "#B37800"
BLUE = "#0058FC"
BLUE_DARK = "#003BA8"
NAVY = "#0B1730"
LIGHT = "#F4F7FF"
SPLASH_W, SPLASH_H = 760, 520

APP_NAME = "Hive Shot Lab"
TEAM = "36563 Prime Symmetry"
COPYRIGHT = "Copyright © 2026 Bolton Robotics FTC"
LICENSE_LINE = "Licensed under Apache 2.0 — Free to use with attribution"
TRADEMARK = ("FIRST® and FIRST® Tech Challenge are registered trademarks of FIRST®, which is not overseeing, "
             "involved with, or responsible for this software.")
REPO_URL = "https://github.com/boltonftc/hive-shot-lab"
# (project, what it does for us, license) - full texts in THIRD_PARTY_NOTICES.md
CREDITS = [
    ("OpenCV", "video reading, finding the ball, AprilTag detection", "Apache 2.0"),
    ("AprilTag", "APRIL Robotics Lab, University of Michigan — the tag family", "BSD 2-Clause"),
    ("FFmpeg", "decoding phone video (inside OpenCV)", "LGPL 2.1"),
    ("Qt / PySide6", "The Qt Company — the app window", "LGPL 3"),
    ("NumPy", "math on the tracked points", "BSD 3-Clause"),
    ("SciPy", "fitting the flight path with air drag", "BSD 3-Clause"),
    ("pandas", "shot tables and CSV files", "BSD 3-Clause"),
    ("Matplotlib", "charts", "Matplotlib License"),
    ("PyInstaller", "packaging the Windows download", "GPL 2 + bootloader exception"),
]

QSS = f"""
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{ background:{LIGHT}; }}
QWidget#header {{ background:{NAVY}; }}
QLabel#title {{ color:white; font-size:26px; font-weight:900; }}
QLabel#subtitle {{ color:{ORANGE}; font-size:13px; font-weight:bold; }}

QPushButton {{ background:{ORANGE}; color:#111; font-weight:bold; border:2px solid #000;
              border-bottom:6px solid {ORANGE_DARK}; border-radius:10px; padding:6px 12px; }}
QPushButton:hover {{ background:#FFC233; }}
QPushButton:pressed {{ border-bottom:2px solid {ORANGE_DARK}; padding-top:10px; padding-bottom:2px; }}
QPushButton:disabled {{ background:#C9C9C9; color:#777; border-bottom-color:#999; }}
QPushButton#primary {{ background:{BLUE}; color:white; border-bottom-color:{BLUE_DARK}; font-size:15px; }}
QPushButton#primary:hover {{ background:#2F78FF; }}
QPushButton#primary:pressed {{ border-bottom:2px solid {BLUE_DARK}; }}

QGroupBox {{ background:white; border:2px solid {BLUE}; border-radius:10px; margin-top:14px; padding-top:6px; }}
QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 6px; color:white; background:{BLUE};
                    border-radius:6px; font-weight:bold; }}

QTabWidget::pane {{ border:3px solid {BLUE}; border-radius:8px; background:white; }}
QTabBar::tab {{ background:#DCE6FF; color:{NAVY}; font-weight:bold; font-size:14px; padding:8px 18px;
               border-top-left-radius:10px; border-top-right-radius:10px; margin-right:4px; }}
QTabBar::tab:selected {{ background:{ORANGE}; color:#111; }}
QTabBar::tab:hover {{ background:#FFD566; }}

QProgressBar {{ border:2px solid {NAVY}; border-radius:6px; text-align:center; font-weight:bold; }}
QProgressBar::chunk {{ background:{ORANGE}; }}
QStatusBar {{ background:{NAVY}; color:white; }}
QStatusBar QLabel {{ color:white; }}
"""


class GlowButton(QPushButton):
    """A button that lights up when you point at it, and can pulse (e.g. while processing)."""

    def __init__(self, text: str, glow: str = ORANGE, primary: bool = False) -> None:
        super().__init__(text)
        if primary:
            self.setObjectName("primary")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._glow = glow_effect(self, glow, 0)
        self.setGraphicsEffect(self._glow)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(lambda v: self._glow.setBlurRadius(v))
        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(6.0)
        self._pulse.setEndValue(40.0)
        self._pulse.setDuration(700)
        self._pulse.setLoopCount(-1)
        self._pulse.setEasingCurve(QEasingCurve.Type.SineCurve)
        self._pulse.valueChanged.connect(lambda v: self._glow.setBlurRadius(v))

    def _glow_to(self, radius: float) -> None:
        if self._pulse.state() == QVariantAnimation.State.Running:
            return
        self._anim.stop()
        self._anim.setStartValue(self._glow.blurRadius())
        self._anim.setEndValue(radius)
        self._anim.start()

    def enterEvent(self, e) -> None:  # noqa: N802 (Qt name)
        self._glow_to(30.0)
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._glow_to(0.0)
        super().leaveEvent(e)

    def pulse(self, on: bool) -> None:
        if on:
            self._pulse.start()
        else:
            self._pulse.stop()
            self._glow.setBlurRadius(0)


def glow_effect(widget, color: str, radius: float) -> QGraphicsDropShadowEffect:
    fx = QGraphicsDropShadowEffect(widget)
    fx.setOffset(0, 0)
    fx.setBlurRadius(radius)
    fx.setColor(QColor(color))
    return fx


def add_glow(button: QPushButton, glow: str = ORANGE) -> None:
    """Give an existing button (e.g. one made by the settings panel) a soft glow."""
    button.setGraphicsEffect(glow_effect(button, glow, 18))


def app_icon() -> QIcon:
    return QIcon(str(ICON))


def logo(path: Path, height: int) -> QPixmap:
    return QPixmap(str(path)).scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)


def _canvas() -> tuple[QPixmap, QPainter]:
    pm = QPixmap(SPLASH_W, SPLASH_H)
    pm.fill(QColor("white"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor(BLUE))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(4, 4, SPLASH_W - 8, SPLASH_H - 8, 24, 24)
    return pm, p


def _text(p: QPainter, y: int, h: int, text: str, color: str, size: int, weight=QFont.Weight.Normal,
          italic: bool = False) -> None:
    p.setPen(QColor(color))
    f = QFont("Segoe UI", size, weight)
    f.setItalic(italic)
    p.setFont(f)
    p.drawText(QRect(40, y, SPLASH_W - 80, h), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)


def splash_pages() -> list[QPixmap]:
    """Created by -> app title -> special thanks."""
    pages = []

    pm, p = _canvas()
    _text(p, 24, 30, "Created by", NAVY, 15, QFont.Weight.Bold)
    lg = logo(LOGO_LIGHT_BG, 270)
    p.drawPixmap((SPLASH_W - lg.width()) // 2, 58, lg)
    _text(p, 335, 26, f"{TEAM}  ·  Bolton Robotics FTC", NAVY, 13, QFont.Weight.Bold)
    _text(p, 365, 22, COPYRIGHT, "#333333", 11)
    _text(p, 389, 22, LICENSE_LINE, "#333333", 11)
    _text(p, 425, 60, TRADEMARK, "#666666", 9, italic=True)
    p.end()
    pages.append(pm)

    pm, p = _canvas()
    lg = logo(LOGO_LIGHT_BG, 330)
    p.drawPixmap((SPLASH_W - lg.width()) // 2, 30, lg)
    _text(p, 380, 60, APP_NAME, NAVY, 30, QFont.Weight.Black)
    _text(p, 440, 30, f"BIOBUZZ pollen shot tracker  ·  FTC team 36563  ·  v{__version__}", ORANGE_DARK, 13,
          QFont.Weight.Bold)
    p.end()
    pages.append(pm)

    pm, p = _canvas()
    _text(p, 26, 44, "Special Thanks To…", NAVY, 22, QFont.Weight.Black)
    y = 84
    for name, what, lic in CREDITS:
        p.setPen(QColor(BLUE))
        p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        p.drawText(QRect(50, y, 160, 40), Qt.AlignmentFlag.AlignVCenter, name)
        p.setPen(QColor("#222222"))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(QRect(215, y, 380, 40), Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, what)
        p.setPen(QColor("#777777"))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRect(600, y, 125, 40), Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, lic)
        y += 44
    p.end()
    pages.append(pm)
    return pages


class Splash(QSplashScreen):
    """Shows the splash pages in turn; after the first one a click skips to the app."""
    done = Signal()
    HOLD_MS = (4000, 2500, 4500)

    def __init__(self) -> None:
        self.pages = splash_pages()
        super().__init__(self.pages[0])
        self.page = 0
        self.timer = QTimer(self, singleShot=True)
        self.timer.timeout.connect(self.next_page)

    def start(self) -> None:
        self.timer.start(self.HOLD_MS[0])

    def next_page(self) -> None:
        self.page += 1
        if self.page >= len(self.pages):
            self.timer.stop()
            self.done.emit()
            return
        self.setPixmap(self.pages[self.page])
        self.showMessage("Click to skip", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, QColor("#999999"))
        self.timer.start(self.HOLD_MS[self.page])

    def mousePressEvent(self, e) -> None:  # noqa: N802 (Qt name)
        if self.page > 0 and self.page < len(self.pages):     # the "Created by" page can't be skipped
            self.page = len(self.pages) - 1
            self.next_page()


def about(parent: QWidget) -> None:
    rows = "".join(f"<tr><td><b>{n}</b></td><td>{w}</td><td><i>{lic}</i></td></tr>" for n, w, lic in CREDITS)
    QMessageBox.about(parent, f"About {APP_NAME}",
                      f"<h2>{APP_NAME} <small>v{__version__}</small></h2>"
                      f"<p>BIOBUZZ pollen shot tracker by FTC team {TEAM}, Bolton Robotics FTC.<br>"
                      f"<a href='{REPO_URL}'>{REPO_URL}</a></p>"
                      f"<p>{COPYRIGHT}<br>{LICENSE_LINE}</p>"
                      f"<h3>Special Thanks To…</h3><table cellspacing='6'>{rows}</table>"
                      f"<p style='color:#666'><i>{TRADEMARK}</i></p>")
