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

| File or directory | Description |
|---|---|
| `Sleepy Driver.py` | Main webcam-based drowsiness detection script |
| `alarm.wav` | Audio alarm played when drowsiness is detected |
| `environment.yml` | Conda environment for the Python dependencies |
| `models/` | Local directory for the dlib facial landmark model (download required) |

The OpenCV Haar cascade is loaded from the installed OpenCV package. The dlib landmark model is downloaded separately and is not stored in this repository.

## Requirements

- Python 3.10 (recommended for the supplied Conda environment)
- A webcam
- Speakers or headphones
- Conda / Miniforge (recommended, particularly on Apple Silicon)

## Setup

### macOS (Apple Silicon / M1, M2, M3, M4)

The project supports macOS. The script uses paths relative to its own location, obtains the Haar cascade from the installed OpenCV package, and uses macOS's built-in `afplay` command for the alarm.

#### 1. Install Miniforge

Download and install the **Apple Silicon** installer from [conda-forge / Miniforge](https://conda-forge.org/download/). After installation, close and reopen Terminal.

#### 2. Clone the repository

```bash
git clone https://github.com/VajraX14/DrowsyDetection.git
cd DrowsyDetection
```

#### 3. Create the environment

```bash
conda env create -f environment.yml
conda activate drowsy-detection
```

#### 4. Download the dlib facial landmark model

The program needs the 68-point landmark file. Run these commands from the repository directory:

```bash
mkdir -p models
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk models/shape_predictor_68_face_landmarks.dat.bz2
```

The extracted file should be located at:

```text
DrowsyDetection/
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── alarm.wav
└── Sleepy Driver.py
```

#### 5. Run the application

```bash
python "Sleepy Driver.py"
```

When macOS asks for camera access, allow access for the application running Python (for example, Terminal). You can review this under **System Settings → Privacy & Security → Camera**.

The webcam window displays the detected eye contours and current EAR. Press **q** while the webcam window is active to quit.

### Windows / Linux

Install Python 3.10 and the dependencies listed in `environment.yml` using Conda. On Windows and Linux, the script uses the `playsound` package for alarm playback, so install it in the environment:

```bash
python -m pip install playsound==1.2.2
```

Then download the dlib landmark model as described above and run:

```bash
python "Sleepy Driver.py"
```

## Configuration

The main parameters are defined near the top of `Sleepy Driver.py`:

| Parameter | Default | Purpose |
|---|---:|---|
| `EYE_THRESHOLD` | `0.30` | EAR below which an eye is treated as closed |
| `FRAMES_LIMIT` | `50` | Consecutive low-EAR frames before the alarm is triggered |
| `CAMERA_INDEX` | `0` | Webcam device index |

These are prototype defaults, not universal values. Results depend on camera position, lighting, face orientation, frame rate, and individual facial characteristics.

## Limitations

- The implementation uses a fixed EAR threshold and consecutive-frame counter.
- Face and eye landmark detection may be unreliable in poor lighting, with occlusions, or when the driver's face is turned away.
- Alarm playback is synchronous and can pause frame processing while the sound plays.
- Camera index `0` assumes the default webcam is the intended input.
- The project has not been validated as a safety-critical system.
- The 68-point dlib model is trained using the iBUG 300-W dataset; its dataset terms exclude commercial use without obtaining appropriate permission. Review the [dlib model documentation](https://github.com/davisking/dlib-models) before any commercial use.

## Potential improvements

- Add calibration and temporal smoothing to improve robustness.
- Run alarm playback asynchronously so video processing is not blocked.
- Add camera selection and better handling of camera/model loading errors.
- Add automated tests and evaluate performance across lighting, camera, and user conditions.
- Consider a more robust face and eye tracking approach.

## Contributing

Contributions and suggestions are welcome. For substantial changes, open an issue first to discuss the proposed improvement.

## License

No license file is currently included in this repository. Unless a license is added, the code should be treated as copyrighted by its owner; do not assume it is available for unrestricted reuse.
