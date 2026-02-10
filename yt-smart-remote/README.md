# YT Smart Remote

Control YouTube touchlessly using hand gestures. Designed for a comfortable sitting experience.

## Gestures
- **Play/Pause**: Perform the sequence: **Open Palm** -> **Fist** -> **Open Palm**.
- **Skip Forward (+10s)**: Show **Victory Sign** (two fingers) and **Swipe Right**.
- **Skip Backward (-10s)**: Show **Victory Sign** (two fingers) and **Swipe Left**.
- **Volume Up/Down**: (Optional) Point Index Finger UP/DOWN. *Note: Currently disabled for silent use.*

## Features
- **Sequence Protection**: The Play/Pause command requires a sequence to avoid accidental triggers.
- **Victory Swipe**: Robust skip detection using horizontal hand movement tracking while holding the Victory sign.
- **Premium Overlay**: Real-time feedback in a sleek header with a debug `STATE` indicator.
- **Optimized for YT**: Uses standard YouTube keyboard shortcuts (`j`, `l`, `space`).

## Requirements
- OpenCV
- MediaPipe
- PyAutoGUI

## Usage
```powershell
python yt-smart-remote/yt_remote.py
```
