import sys
import time
import math
import cv2
import mediapipe as mp

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "cat-dance.mp4"
DISPLAY_SCALE = 1 / 2.5
DETECT_EVERY = 2
MOTION_THRESHOLD = 0.02
GRACE_PERIOD = 0.5

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
if not cam.isOpened():
    sys.exit("Could not open the camera. Grant camera access to your terminal in "
             "System Settings > Privacy & Security > Camera, then restart it.")

for _ in range(15):
    if cam.read()[0]:
        break
    time.sleep(0.2)
else:
    sys.exit("Camera opened but returned no frames. Check camera permission for "
             "your terminal in System Settings > Privacy & Security > Camera.")

video = cv2.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    sys.exit(f"Could not open video: {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS)
if not fps or fps != fps or fps <= 0 or fps > 240:
    fps = 30.0
frame_interval = 1.0 / fps

hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
WRIST = mp.solutions.hands.HandLandmark.WRIST

ok_first, current_frame = video.read()
if not ok_first:
    sys.exit(f"Could not decode any frames from {VIDEO_PATH}")

prev_wrists = []
landmark_list = []
last_motion = 0.0
last_advance = 0.0
tick = 0
window_name = "cat-dance"

try:
    while True:
        ok, frame = cam.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        now = time.time()
        tick += 1

        if tick % DETECT_EVERY == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(rgb)
            landmark_list = hand_result.multi_hand_landmarks or []

            wrists = [(lm.landmark[WRIST].x, lm.landmark[WRIST].y) for lm in landmark_list]

            moved = 0.0
            if wrists and len(wrists) == len(prev_wrists):
                for x, y in wrists:
                    moved = max(moved, min(math.hypot(x - px, y - py) for px, py in prev_wrists))
            prev_wrists = wrists

            if moved > MOTION_THRESHOLD:
                last_motion = now

        for landmarks in landmark_list:
            mp_draw.draw_landmarks(
                frame,
                landmarks,
                HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

        playing = now - last_motion < GRACE_PERIOD

        if playing and now - last_advance >= frame_interval:
            ok_v, video_frame = video.read()
            if not ok_v:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok_v, video_frame = video.read()
                if not ok_v:
                    sys.exit(f"Could not decode any frames from {VIDEO_PATH}")
            current_frame = video_frame
            last_advance = now

        cv2.imshow(
            window_name,
            cv2.resize(current_frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE),
        )

        #status = "DANCING" if playing else "move your hands" ⁡⁢⁣⁣#DEBAG⁡
        color = (0, 255, 0) if playing else (0, 0, 255)
        #cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2) ⁡⁢⁣⁣#DEBAG⁡
        cv2.imshow("camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
except KeyboardInterrupt:
    pass
finally:
    cam.release()
    video.release()
    hands.close()
    cv2.destroyAllWindows()