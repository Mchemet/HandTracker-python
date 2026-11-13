# HandTracker-python
Coding a hand tracking program in python, using cv2 and mediapipe

This was made on an archlinux-based environment and this program runs from an arch-based terminal, meaning options like "leaving" the window when the programs is tracking your hand i.e counts the risen fingers, observs each member of your hand and tracks both hands etc... requires you to press either on `q`, `ESC` or `CTRL + C`

Most of this code is composed of cv2-based commands that come from its library, the commands of mediapipe and also mathematic functions. 

![Hand tracker demo](demo.gif)

As the owner of the repo, I apologize for the imperfections.I built this on an archlinux machine; meaning I haven't done it on windows or Mac. `yet.`

Future modifications to improve the project will be done, eventually.

# Installation

You need Python 3.12 or older, git, and a webcam.

# Linux / MacOS

```bash
git clone https://github.com/Mchemet/HandTracker-python.git
cd HandTracker-python
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe opencv-python
python hand_tracker.py
```
# Windows (Powershell)

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

The first run downloads the model file (`gesture_recognizer.task`), so you need an internet connection once. Press `q` for `ESC` in the video window to quit.

# Running it again later

```bash
cd HandTracker-python
source .venv/bin/activate        
python hand_tracker.py
```

# Common issues

note :this program was made on an archlinux machine. any errors from MacOS or Windows should be reported to the owner of this repo.