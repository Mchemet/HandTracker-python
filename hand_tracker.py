import math
import os #checks if the model file exists (hand_landmarker.task)
import time #FPS counter
import urllib.request #downloads the model file if it doesn't exist
import cv2 #webcam, drawing and gives us the window
import mediapipe as mp #loads the MediaPipe library and gives it the short name mp, so we can write mp.Image(..) instead of mediapipe.Image(...)
from mediapipe.tasks import python as mp_python #(model file loading)
from mediapipe.tasks.python import vision # HandLandmarker lives here

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

    # ^
    # the pip (python package installer) contains the code only, so the trained neural network has to be downloaded separately
    # in other words, the tasks api has to work with a neural network that needs to be downloaded.

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task (one time, ~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm edge
]

# ^
#1-precise coordinates since the landmark model is trained to always output the 21 points in the same fixed order, so index 0 is 
# always the wrist, 4 is always the thumb tip, 8 is always the index tip etc...

FINGERS = {"Index" : (8, 6), "Middle": (12, 10), "Ring": (16, 14), "Pinky": (20, 18)}

#^
#python tuples with our own declarations

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),#video mode: mediapipe tracks hands across frames,It needs a timestamp that
    #always increases
    #other modes: IMAGE (single pictures) and LIVE_STREAM
    #(async callback, more complex).
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,                          # max hands to look for
    min_hand_detection_confidence=0.5,    # palm detector threshold
    min_hand_presence_confidence=0.5,     # "is a hand still there?" threshold
    min_tracking_confidence=0.5,          # below this, it re-runs detection
)
landmarker = vision.HandLandmarker.create_from_options(options)

def dist(a, b):
    #Return the distance between two landmarks. (in normalized units)
    return math.hypot(a.x - b.x, a.y - b.y)

def count_fingers(lm): #lm short for landmarks
    
    #1)- count raised fingers from the 21 landmarks `lm`.
    #2)- a finger is "up" when its tip is farther from the wrist than its
    #middle joint (pip). Using distance instead of "tip.y < pip.y" means it
    #still works if your hand is rotated or sideways.

    wrist = lm[0]
    up = []

    for name, (tip, pip) in FINGERS.items():
        if dist(lm[tip], wrist) > dist(lm[pip], wrist):
            up.append(name)

    if dist(lm[4], lm[17]) > dist(lm[3], lm[17]):
        up.append("Thumb")
    return up

def draw_hand(frame, lm, color):
    #Draw skeleton lines then joint dots on `frame` (modified in place).
    h, w, _ = frame.shape

    #LMs (landmarks) are NORMALIZED (0.0-1.0 relative to image width/height).
    #multiply by the frame size to get pixel coordinates.
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for i, (x, y) in enumerate(pts):
        #(optional) the size of fingers' dots
        r = 7 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), r, color, 1)
    return pts

#main loop :

def main():
    cap = cv2.VideoCapture(0)             
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    start = time.time()
    prev = start

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        #mirror the image so it behaves like a mirror (left hand on the left)
        #mediapipe's left/right labels assume this mirrored "selfie" view
        frame = cv2.flip(frame, 1)

        #OpenCV uses BGR color order, mediapipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        #timestamp in milliseconds, strictly increasing (required in video mode).
        timestamp_ms = int((time.time() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        #`result` holds parallel lists, one entry per detected hand:
        #result.hand_landmarks -> 21 normalized (x, y, z) points
        #result.handedness     -> "Left"/"Right" + confidence score
        #(z = depth relative to the wrist; smaller = closer to the camera.)
        for lm, handed in zip(result.hand_landmarks, result.handedness):
            label = handed[0].category_name            #"left" or "right"
            score = handed[0].score
            color = (0, 200, 0) if label == "Right" else (255, 120, 0)

            pts = draw_hand(frame, lm, color)
            fingers = count_fingers(lm)

            #pinch distance: thumb tip (4) to index tip (8), in pixels.
            pinch_px = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1])
            cv2.line(frame, pts[4], pts[8], (0, 0, 255), 2)

        
            x, y = pts[0]
            cv2.putText(frame, f"{label} ({score:.2f})  fingers: {len(fingers)}",
                        (x - 60, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"pinch: {pinch_px:.0f}px",
                        (x - 60, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):     #q or ESC (ctrl+c if running from terminal)
            break

   
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()