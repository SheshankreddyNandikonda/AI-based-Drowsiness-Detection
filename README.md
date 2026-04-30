
# Drowsiness Detection using MediaPipe and LSTM

![GitHub](https://img.shields.io/badge/Language-Python-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

**Drowsiness Detection** is an advanced computer vision project that detects driver drowsiness in real-time using facial landmarks and deep learning. The system leverages Google's **MediaPipe** library for efficient facial feature extraction and an **LSTM-based neural network** for accurate drowsiness classification.

This system is particularly useful for:
- 🚗 **Driver Safety**: Real-time alerts to prevent accidents caused by driver drowsiness
- 👁️ **Facial Analysis**: Extraction of multiple facial features including eye aspect ratio, mouth opening, and pupil circularity
- 🤖 **AI-Powered Detection**: Advanced LSTM model for temporal pattern recognition

---

## ✨ Features

- **Real-time Detection**: Processes live video feed from webcam with minimal latency
- **Multiple Facial Features**:
  - **Eye Aspect Ratio (EAR)**: Detects eye closure patterns
  - **Mouth Aspect Ratio (MAR)**: Monitors yawning behavior
  - **Pupil Circularity**: Analyzes pupil shape consistency
  - **Mouth-to-Eye Ratio**: Combined feature for enhanced accuracy
  
- **Smart Calibration**: User-friendly calibration process to normalize features for individual variations
- **Deep Learning Model**: LSTM network trained to recognize drowsy state patterns over temporal sequences
- **Visual Feedback**: Real-time overlay of facial landmarks and detection results on video feed

---

## 🔧 How It Works

### Architecture

1. **Face Detection & Landmark Extraction** (MediaPipe FaceMesh)
   - Detects 468 facial landmarks in real-time
   - Provides normalized 3D coordinates for robust feature calculation

2. **Feature Extraction**
   - Computes 4 key facial features from landmark positions
   - Features are normalized based on calibration baseline
   - Features are accumulated over a sliding window of frames

3. **Temporal Analysis** (LSTM Model)
   - Processes sequential feature data (20-frame sequences)
   - Recognizes drowsiness patterns in temporal dynamics
   - Uses ensemble predictions across overlapping windows for robustness

4. **Drowsiness Classification**
   - Outputs binary prediction: Alert or Drowsy
   - Triggers audio/visual alerts when drowsiness is detected

### Facial Features Explained

**Eye Aspect Ratio (EAR)**:
- Ratio of eye height to eye width
- Decreases significantly when eyes close or during drooping

**Mouth Aspect Ratio (MAR)**:
- Ratio of mouth height to mouth width
- Increases during yawning (common drowsiness indicator)

**Pupil Circularity**:
- Measures how circular the pupil region appears
- Irregular patterns indicate potential fatigue or drowsiness

**Mouth-to-Eye Ratio (MOE)**:
- Combined feature: MAR / EAR
- Provides holistic drowsiness indicator

---

## 📦 Requirements

- Python 3.7 or higher
- OpenCV (cv2)
- MediaPipe
- PyTorch
- NumPy

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Drowsiness-Detection-Mediapipe.git
cd Drowsiness-Detection-Mediapipe-main
```

### 2. Install Dependencies
```bash
pip install opencv-python mediapipe torch numpy
```

Or using requirements.txt (if available):
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Running Inference

```bash
python inference.py
```

### Step-by-Step Guide

1. **Launch the Script**
   ```bash
   python inference.py
   ```

2. **Calibration Phase**
   - The system will show a "Calibration" message
   - Keep your head still in a neutral position
   - Calibration collects 25 frames for baseline features
   - Press 'q' to skip calibration (not recommended for first run)

3. **Detection Phase**
   - The system begins monitoring your facial features in real-time
   - **Green text**: Alert state (eyes open, attentive)
   - **Red text**: Drowsy state (potential eye closure or yawning detected)
   - Visual overlay shows:
     - Facial mesh landmarks
     - Current drowsiness classification
     - Real-time feature values

4. **Exit**
   - Press 'q' to quit the application

---

## 📁 Project Structure

```
Drowsiness-Detection-Mediapipe-main/
│
├── inference.py           # Main inference script with real-time detection
├── models/
│   └── clf_lstm_jit6.pth  # Pre-trained LSTM model (JIT format)
├── README.md              # Project documentation
├── LICENSE                # License file
└── .git/                  # Git repository

```

### Key Components

**inference.py** contains:
- `distance()`: Calculate Euclidean distance between facial landmarks
- `eye_aspect_ratio()`: Compute EAR for drowsiness detection
- `mouth_feature()`: Extract mouth-related features
- `pupil_circularity()`: Calculate pupil shape consistency
- `eye_feature()`: Average EAR across both eyes
- `run_face_mp()`: Main MediaPipe face detection pipeline
- `calibrate()`: User calibration to establish baseline
- `get_classification()`: LSTM model inference for drowsiness prediction
- `infer()`: Main inference loop for real-time detection

---

## 🎯 Model Details

### LSTM Model Architecture
- **Input**: Sequential facial features (6 overlapping windows of 5-frame sequences)
- **Hidden Units**: Configured for temporal pattern recognition
- **Output**: Binary classification (Alert/Drowsy)
- **Threshold**: 0.5 probability with ensemble voting (≥5 out of 6 windows for drowsy classification)

### Model File
- **Path**: `models/clf_lstm_jit6.pth`
- **Format**: PyTorch JIT (Just-In-Time compiled for efficiency)
- **Status**: Pre-trained and ready for inference

---

## 📊 Performance Considerations

- **Real-time Processing**: Optimized for ~30 FPS on standard hardware
- **Calibration Importance**: Accurate calibration significantly improves detection accuracy
- **Lighting Conditions**: Works best in well-lit environments
- **Face Position**: Frontal face view recommended for optimal results

---

## ⚙️ Configuration & Customization

### Adjustment Parameters in inference.py

```python
# Calibration frame count (default: 25)
calib_frame_count = 25

# Feature smoothing decay (default: 0.9)
decay = 0.9

# Threshold for drowsiness ensemble (default: ≥5 out of 6)
# Modify the condition in get_classification() function
```

---

## 🔍 Troubleshooting

### Issue: Face not detected
- **Solution**: Ensure adequate lighting and frontal face orientation
- Check that your face is clearly visible to the camera
- Try recalibration

### Issue: Poor accuracy after calibration
- **Solution**: Recalibrate with eyes fully open and alert
- Ensure consistent lighting during calibration and inference

### Issue: Low FPS / Lagging
- **Solution**: Check camera resolution and reduce if necessary
- Ensure no other heavy processes are running

---

## 🤝 Contributing

Contributions are welcome! Please feel free to fork the repository, create feature branches, and submit pull requests.

---

## 👥 Authors

- **Nandikonda Sheshank Reddy**
- **Chegonda Vamshi**
- **Nitta Anil**
- **Bomma Richitha**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google MediaPipe**: For the robust facial landmark detection framework
- **PyTorch**: For the deep learning framework
- **OpenCV**: For computer vision utilities

---

## 📧 Contact & Support

For questions, issues, or suggestions, please open an issue on GitHub or contact the project maintainers.

---

**Disclaimer**: This project is for educational and research purposes. While it can provide alerts for drowsiness detection, it should not be the sole solution for preventing accidents. Always practice safe driving habits and take regular breaks during long drives.




