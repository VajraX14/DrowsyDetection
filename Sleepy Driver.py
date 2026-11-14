# -*- coding: utf-8 -*-
"""Webcam-based driver drowsiness detection using eye aspect ratio (EAR)."""

from pathlib import Path
import sys
import time

import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist

# macOS has a built-in audio player (afplay). Keep playsound for other OSes.
if sys.platform != "darwin":
    import playsound


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FACE_CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
PREDICTOR_PATH = MODEL_DIR / "shape_predictor_68_face_landmarks.dat"
ALARM_PATH = BASE_DIR / "alarm.wav"

EYE_THRESHOLD = 0.30
FRAMES_LIMIT = 50
CAMERA_INDEX = 0


def sound_alarm(path: Path) -> None:
    """Play the alarm using the operating system's available audio player."""
    if sys.platform == "darwin":
        import subprocess

        subprocess.run(["afplay", str(path)], check=False)
    else:
        playsound.playsound(str(path))


def eye_aspect_ratio(eye) -> float:
    """Calculate the eye aspect ratio from six eye landmark coordinates."""
    vertical_a = dist.euclidean(eye[1], eye[5])
    vertical_b = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])

    if horizontal == 0:
        return 0.0

    return (vertical_a + vertical_b) / (2.0 * horizontal)


def main() -> None:
    if not FACE_CASCADE_PATH.is_file():
        raise FileNotFoundError(
            "Missing OpenCV Haar cascade. Download "
            "'haarcascade_frontalface_default.xml' into the 'models' folder. "
            f"Expected location: {FACE_CASCADE_PATH}"
        )
    if not PREDICTOR_PATH.is_file():
        raise FileNotFoundError(
            "Missing dlib landmark model. Download and extract "
            "'shape_predictor_68_face_landmarks.dat' into the 'models' folder. "
            f"Expected location: {PREDICTOR_PATH}"
        )
    if not ALARM_PATH.is_file():
        raise FileNotFoundError(f"Alarm audio file not found: {ALARM_PATH}")

    print("[INFO] Loading face detector and facial landmark predictor...")

    detector = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if detector.empty():
        raise RuntimeError("OpenCV could not load the Haar cascade classifier.")

    predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    (left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("[INFO] Starting webcam...")
    video_stream = VideoStream(src=CAMERA_INDEX).start()
    time.sleep(1.0)

    closed_eye_frames = 0

    try:
        while True:
            frame = video_stream.read()
            if frame is None:
                print("[ERROR] Could not read a frame from the webcam.")
                break

            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for (x, y, width, height) in faces:
                face_rect = dlib.rectangle(
                    int(x), int(y), int(x + width), int(y + height)
                )
                landmarks = predictor(gray, face_rect)
                landmarks = face_utils.shape_to_np(landmarks)

                left_eye = landmarks[left_start:left_end]
                right_eye = landmarks[right_start:right_end]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                left_hull = cv2.convexHull(left_eye)
                right_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)

                if ear < EYE_THRESHOLD:
                    closed_eye_frames += 1
                    if closed_eye_frames >= FRAMES_LIMIT:
                        cv2.putText(
                            frame,
                            "DROWSINESS ALERT!",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        sound_alarm(ALARM_PATH)
                        closed_eye_frames = 0
                else:
                    closed_eye_frames = 0

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        video_stream.stop()
        print("[INFO] Drowsiness detection stopped.")


if __name__ == "__main__":
    main()
