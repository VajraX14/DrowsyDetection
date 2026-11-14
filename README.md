# Drowsy Detection

A real-time driver drowsiness monitoring prototype built with Python, OpenCV and dlib. The application uses a webcam to locate a driver's face, tracks facial landmarks around both eyes, and calculates the **Eye Aspect Ratio (EAR)**. When the eyes remain closed (EAR below a configurable threshold) for a sustained number of frames, the program plays an audio alarm.

> **Safety notice:** This is an experimental computer-vision project, not a certified driver-monitoring or vehicle safety system. It can produce false alarms or fail to detect drowsiness. Never rely on it as a substitute for adequate rest, safe driving practices, or a production-grade driver assistance system.

## How it works

1. Capture frames from the default webcam.
2. Resize each frame and convert it to grayscale.
3. Detect face regions using OpenCV's Haar cascade classifier.
4. Use dlib's 68-point facial landmark predictor to locate the eye landmarks.
5. Calculate the EAR for the left and right eyes, then average the two values.
6. Count consecutive frames for which the average EAR is below the configured threshold.
7. Play `alarm.wav` and display a drowsiness warning when the frame-count limit is reached.

### Eye Aspect Ratio

The Eye Aspect Ratio estimates how open an eye is by comparing vertical eye-landmark distances with the horizontal eye width:

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

A relatively low EAR indicates that the eye is closed or nearly closed. This implementation uses the average EAR of both eyes.

## Repository contents

| File | Description |
|---|---|
| `Sleepy Driver.py` | Main webcam-based drowsiness detection script |
| `alarm.wav` | Audio alarm played when the drowsiness threshold is reached |

The script also requires two external model files that are **not currently included** in this repository:

- `haarcascade_frontalface_default.xml` — OpenCV Haar cascade for face detection.
- `shape_predictor_68_face_landmarks.dat` — dlib's 68 facial landmark predictor.

## Requirements

- Python 3
- A working webcam
- Speakers or headphones for the alarm
- The Python packages:
  - OpenCV (`opencv-python`)
  - dlib
  - imutils
  - SciPy
  - playsound

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/VajraX14/DrowsyDetection.git
cd DrowsyDetection
```

### 2. Create and activate a virtual environment (recommended)

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python dlib imutils scipy playsound
```

**Note:** dlib may require additional build tools depending on your operating system and Python version. Installation can be more involved than the other packages.

### 4. Download the required model files

Download the dlib 68-point facial landmark predictor from the [dlib model files](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) page and extract the `.dat` file.

The Haar cascade is distributed with OpenCV. You can locate it from your Python environment:

```python
import cv2
print(cv2.data.haarcascades)
```

The required file is `haarcascade_frontalface_default.xml`. Alternatively, obtain it from the [OpenCV repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

Place both model files in a convenient local directory. The current script contains absolute Windows paths, so **update the paths in `Sleepy Driver.py`** to match your machine before running it. In particular, update the paths used to initialise the face detector, landmark predictor, and alarm file. The alarm file is already present in this repository.

## Run

From the repository directory, with your virtual environment active:

```bash
python "Sleepy Driver.py"
```

The script opens a webcam window and displays the calculated EAR. Press **q** while the video window is active to stop the application.

## Configuration

The main detection parameters are defined near the beginning of `Sleepy Driver.py`:

| Parameter | Current value | Purpose |
|---|---:|---|
| `eye_threshold` | `0.3` | EAR below which an eye is treated as closed |
| `frames_limit` | `50` | Number of consecutive low-EAR frames before the alarm is triggered |
| `VideoStream(src=0)` | `0` | Webcam device index |

These are prototype defaults, not universal values. Results depend on camera position, lighting, face orientation, frame rate, and individual facial characteristics. Changing the threshold or frame limit may affect false-positive and false-negative rates.

## Limitations

- The current implementation uses a fixed EAR threshold and consecutive-frame counter.
- Face and eye landmark detection may be unreliable in poor lighting, with occlusions, or when the driver's face is turned away.
- The alarm is played synchronously, which may interrupt frame processing while the sound is playing.
- Camera index `0` assumes the default webcam is the intended input.
- The script currently uses absolute Windows file paths and needs path updates for other machines.
- The project has not been validated as a safety-critical system.

## Potential improvements

- Replace machine-specific paths with configurable, cross-platform paths.
- Add a `requirements.txt` and configuration file.
- Separate alarm playback from the video-processing loop.
- Add calibration or temporal smoothing to make detection more robust.
- Handle missing camera, model files, and face-detection failures gracefully.
- Add tests and document evaluation against representative lighting, camera, and user conditions.

## Contributing

Contributions and suggestions are welcome. For substantial changes, open an issue first to discuss the proposed improvement.

## License

No license file is currently included in this repository. Unless a license is added, the code should be treated as copyrighted by its owner; do not assume it is available for unrestricted reuse.
