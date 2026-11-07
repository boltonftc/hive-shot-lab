# Hive Shot Lab

**Measure your FTC BIOBUZZ pollen shots from a slow-motion video.**
Made by FIRST Tech Challenge team **36563 Prime Symmetry** (Bolton Robotics FTC).

Film your robot shooting from the side, open the video in Hive Shot Lab, click the turret once, and press **Process**.
For every ball it finds, you get:

- **Launch speed** (ft/s) and **launch angle**
- The **whole flight path**, drawn over the video and in a side-view chart
- Whether it reaches the hive cell **IN**, **LOW**, **HIGH** or **SHORT**, judged against a virtual cell opening,
  so you don't need a real field element
- **Bursts**: balls fired close together are numbered #1, #2, #3, … so you can see whether later balls slow down
- **Comparisons** between robot setups (flywheel rpm, flywheel mass, hood angle) across many videos
- A shots table (CSV) and a one-page **HTML report** with the method, settings, tables and charts

## Get it

### Windows: download

1. Download `HiveShotLab-…-windows.zip` from [Releases](https://github.com/boltonftc/hive-shot-lab/releases).
2. Unzip it anywhere and run `HiveShotLab.exe` in the `HiveShotLab` folder.
3. The program isn't code-signed, so Windows may say *"Windows protected your PC"*. Click **More info → Run anyway**.

### Any computer: run from source

You need [Python](https://www.python.org/downloads/) 3.10 or newer.

**Windows:** download or clone this repository and double-click `run_hive_shot_lab.bat`.
The first run installs everything into a `.venv` folder, which needs internet and takes a few minutes.
You can drag a video onto the `.bat` file to open it straight away.

**Mac / Linux / by hand:**

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
hive-shot-lab                      # or: python -m hive_shot_lab [VIDEO]
```

## Filming: this decides how good your numbers are

The app measures what it sees, so set the camera up carefully once and leave it there.

| Do | Why |
|---|---|
| **Slow motion, 120 or 240 fps**, 1080p, HDR off | At 240 fps a 20 ft/s ball moves about 1 in between frames, so every flight gives 100+ points to fit |
| Phone **on a tripod**, looking **straight at the side** of the shot (square to the plane the ball flies in) | The app measures in that one plane. Turning the camera makes near things look bigger than far things |
| **One AprilTag standing upright in the plane the ball flies in** (on the robot's centerline), out of the ball's way | The tag sets the scale (pixels per inch) and which way is up. A tag 9 in closer to the camera than the ball path, at 10 ft away, makes every number about 7 % wrong |
| Use a **tag36h11** AprilTag and **measure its black square** edge to edge | The DECODE field tags are 6.5 in (165.1 mm), which is the default. Glue a printed tag flat on cardboard |
| **Plain background behind the flight path**, ideally a green screen, and **nothing else yellow** in view | The ball is found by its yellow colour. Yellow robot parts, shirts or game pieces lying around can confuse it |
| Bright, steady light; lock focus and exposure (long-press on iPhone) | A fast shutter keeps the ball round instead of a streak. Some LED lights flicker at high frame rates |
| Get the turret, the whole flight up to the hive cell area, and the floor in frame | The floor line drawn by the app is a free check that the scale and level are right |
| Copy the **original** video to the computer (USB cable, AirDrop, or "All Photos Data" export) | A video shared or exported "for playback" may be saved at 30 fps. The app usually detects this from gravity and fixes the timing, but the original is better |
| Name videos after the robot setup, like `robot1_2571rpm_152g.mov` (optional `_30deg` for the hood) | Batch process fills in robot, rpm, flywheel mass and hood angle from the name |
| Don't move the camera between videos you want to compare | Each video needs its own turret click, but the same setup keeps comparisons fair |

## Using the app

1. **📂 Open video.** The app finds the AprilTag and draws a 1-ft grid, the floor line and the virtual hive cell.
2. **Check the tag size** (step 2 on the left) matches your printed tag.
3. **🎯 Pick turret** and click the center of the turret in the video. Tick *"Shots fly toward the right"* if they do.
   Enter the turret height above the floor.
4. **Check the target**: the virtual hive cell opening (distance from the turret and height). The defaults are our
   measurement of the BIOBUZZ hive cell for a robot against the wall; change them to match your own.
5. **▶ Process** (F5). It runs in the background; you can keep looking at the video.
6. Step through the video (← → keys, Shift for 10 frames, Space to play). Each shot's path is drawn, and the result
   flashes when the ball reaches the cell. White dotted lines are tracks the app decided were *not* shots.
7. Look at the **📈 Charts**, **📋 Shots** and **🏆 Summary** tabs. **💾 Export CSV** and **📄 Report** save the results.

Changed only the target, ball size or burst gap? Press **⚖ Re-judge**; it's instant. Changed the tag, turret,
search area, colour or tracking settings? Press **Process** again.

**Comparing many videos (batches).** Press **🗂 Batch process** and pick your videos. Click each video in the batch list
and pick its turret (a ✔ appears), then press **Batch process** again. The charts, shots table and summary then cover every
video, coloured by robot setup, and you can tick data sets on and off and filter by rpm and flywheel mass.

**Advanced settings** (bottom of the left panel) hold every other number: tag id, ball size and colour range,
cell shape, the area to search for flying balls, and tracking limits. Hover over the video to read the
position in inches and the colour (H/S/V) under the mouse. That is how you tune the yellow range.

### What counts as a shot

A track counts as a shot only if it flies like a thrown ball:

- it falls at about 1 g
- it fits a smooth flight path with air drag within 1.5 in
- it starts within 18 in of the turret
- it rises at least 6 in above the turret

Someone carrying a ball, a ball rolling on the floor, or a bounce is ignored.

### Results

| Result | Meaning |
|---|---|
| **IN** | The ball's center crosses the opening at least one ball radius inside both edges |
| **LOW / HIGH** | It crosses the opening line below / above that |
| **SHORT** | It never reaches the opening |

## How it works

1. **Scale and level.** One AprilTag of known size gives pixels per inch from its corners (found to a fraction of a
   pixel), and its edges show which way is up. The turret click is the origin: x = inches toward the cell,
   y = inches above the floor.
2. **Find the ball.** In each frame, yellow pixels (an HSV colour range) are found inside the search area, and only
   ball-sized blobs are kept. While nothing is flying, only some frames are checked, so long reloads are fast.
3. **Track.** Each blob is joined to the ball whose predicted position is nearest. When two balls of a burst overlap in
   the picture, both tracks are kept and picked up again once they separate. The time a ball sits still in the turret
   and anything after the flight ends (a bounce, or another yellow object) are cut off.
4. **Physics.** Each path is fitted with a projectile **with air drag**. A plain parabola reads the launch speed about
   2 % low. The fit gives launch speed and angle, the highest point, and the whole path, including past the last
   sighting.
5. **Gravity check.** Every real flight should fall at about 1 g. If they all fall 4× or 8× too slowly, the video was
   saved at playback speed, and the app corrects its frame rate.
6. **Judge and group.** Where the path crosses the cell opening decides the result. Launches closer together than the
   burst gap form a burst.

**Accuracy.** Tested on a generated video with known answers: launch speed within 0.3 % and launch angle within
0.3°, with the camera tilted 1.5°.

**Limits.** It only sees one side, so left/right misses aren't measured. A ball hidden behind the robot or the tag is
lost for those frames. Everything depends on the tag standing in the flight plane.

## Troubleshooting

| Problem | Try |
|---|---|
| "No AprilTag (tag36h11) found" | Make sure the whole tag is in view, sharp, and well lit. It must be a tag36h11 tag. Pick its id under Advanced → AprilTag if several are in view |
| No shots, or "track N: not a shot" on real shots | Look for yellow things in view (robot parts too). Narrow the search area under Advanced ("x from" / "x to"). Check the turret click and "Shots fly toward the right" |
| Speeds look 4× or 8× off | The video lost its slow-motion frame rate. Copy the original, or set *Frame rate override* under Advanced |
| The grid looks tilted | The tag isn't standing straight. Straighten it, or use *Extra level adjustment* under Advanced |
| Ball not found at all | Hover over the ball to read its H/S/V colour, and adjust the yellow range under Advanced → Ball. Turn on *Yellow mask* to check |

## Credits and license

Copyright © 2026 Bolton Robotics FTC. Licensed under the [Apache License 2.0](LICENSE). Free to use with
attribution, see [NOTICE](NOTICE).

**Special Thanks To** the open-source projects that make this possible: OpenCV, AprilTag (APRIL Robotics Lab,
University of Michigan), FFmpeg, Qt for Python (PySide6), NumPy, SciPy, pandas, Matplotlib and PyInstaller.
Details and licenses are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

*FIRST*® and *FIRST*® Tech Challenge are registered trademarks of *FIRST*®, which is not overseeing, involved with,
or responsible for this software.
