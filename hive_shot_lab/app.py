"""Start Hive Shot Lab:  python -m hive_shot_lab [VIDEO]"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from . import theme
from .window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(theme.APP_NAME)
    app.setStyleSheet(theme.QSS)
    app.setWindowIcon(theme.app_icon())
    splash = theme.Splash()
    splash.show()
    app.processEvents()
    w = MainWindow()
    w.resize(1600, 950)
    if len(sys.argv) > 1:
        w.open_video(Path(sys.argv[1]))

    def reveal() -> None:
        w.show()
        splash.finish(w)

    splash.done.connect(reveal)
    splash.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
