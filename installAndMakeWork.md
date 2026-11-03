## Installation

You need Python 3.12 or older, git, and a webcam.

### Linux / macOS

```bash
git clone https://github.com/Mchemet/HandTracker-python.git
cd HandTracker-python
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe opencv-python
python hand_tracker.py
```

### Windows (PowerShell)

```powershell
git clone https://github.com/Mchemet/HandTracker-python.git
cd HandTracker-python
python -m venv .venv
.venv\Scripts\activate
pip install mediapipe opencv-python
python hand_tracker.py
```

### Without git

1. Click **Code → Download ZIP** on this page and unzip it.
2. Open a terminal inside the unzipped folder.
3. Run the `python -m venv .venv` line and everything after it from the commands above.

The first run downloads the model file (`gesture_recognizer.task`), so you need an internet connection once. Press `q` or `ESC` in the video window to quit.

### Running it again later

```bash
cd HandTracker-python
source .venv/bin/activate        
python hand_tracker.py
```

### Common problems

- **`No matching distribution found for mediapipe`:** your Python version is too new. Install Python 3.12 and recreate the environment.
- **`No module named 'cv2'`:** the virtual environment isn't active. Run the activate command again.
- **Black window or webcam error:** change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `hand_tracker.py`.
- **Linux, Qt/xcb error:** run `QT_QPA_PLATFORM=xcb python hand_tracker.py`.