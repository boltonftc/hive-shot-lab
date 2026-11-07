# Third-party notices

Hive Shot Lab is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
It is built on the open-source projects below. Thank you to everyone who makes them.

| Project | What it does in Hive Shot Lab | License | Home |
|---|---|---|---|
| OpenCV (`opencv-python-headless`) | Reads the video, finds the yellow ball, detects the AprilTag | Apache 2.0 | https://opencv.org/ |
| AprilTag | The tag family (tag36h11) and detection method, used through OpenCV | BSD 2-Clause | https://april.eecs.umich.edu/software/apriltag |
| FFmpeg (bundled inside the OpenCV package) | Decodes phone video (HEVC / H.264) | LGPL 2.1 | https://ffmpeg.org/ |
| Qt for Python (PySide6) | The app window, buttons, splash screen | LGPL v3 | https://www.qt.io/qt-for-python |
| NumPy | Math on the tracked points | BSD 3-Clause | https://numpy.org/ |
| SciPy | Fits each flight path with air drag (`least_squares`) | BSD 3-Clause | https://scipy.org/ |
| pandas | Shot tables, summaries and CSV files | BSD 3-Clause | https://pandas.pydata.org/ |
| Matplotlib | Charts in the app and in the HTML report | Matplotlib License (PSF-based) | https://matplotlib.org/ |
| PyInstaller | Packages the Windows download (not used when running from source) | GPL 2 with bootloader exception | https://pyinstaller.org/ |

When you run from source, `pip` installs these packages and their license files into your Python environment.

The Windows download contains copies of these libraries. Its `THIRD_PARTY_LICENSES.txt` holds the full license
text of every bundled package. Qt and FFmpeg are kept as separate library files (not merged into the program), so
you can replace them with your own builds as the LGPL allows. Their source code is available from the project
pages above.
