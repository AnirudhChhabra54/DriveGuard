# DriveGuard: AI-Based Driver Drowsiness Detection System

## Overview
DriveGuard is an intelligent web application designed to prevent accidents caused by driver fatigue. Running entirely in the browser without remote backend dependencies, it utilizes advanced computer vision via TensorFlow.js and MediaPipe to analyze a driver's state of alertness in real-time. 

## Key Features
- **Real-Time Inference:** Operates at 30 FPS with latency below 40ms, directly in the browser.
- **Privacy First:** No images or video feeds are sent to any server. Everything is processed locally on the client.
- **Hybrid Alertness Detection:** Combines a custom CNN binary classification with geometric facial tracking (EAR & PERCLOS).
- **Responsive Dashboard:** Sleek Glassmorphism UI built to provide immediate visual and audio feedback to the driver.

## System Architecture
The application bridges the gap between deep learning models and client-side execution for zero-latency inference.

![System Architecture](imgg/arch.png)
*(See `imgg/dfd.png`, `imgg/state.png`, and `imgg/use_case.png` for complete architectural workflows).*

## Deep Learning & Model Accuracy
We utilized the Kaggle **Driver Drowsiness Dataset (DDD)**, applying extensive data augmentation (rotation, zooming, brightness scaling) to train a robust Convolutional Neural Network (CNN).

### CNN Model Performance
- **Training Accuracy:** ~99.97%
- **Testing (Validation) Accuracy:** ~99.86%
- **Quantization:** Reduced to an INT8 quantized footprint of `1.17 MB` for rapid browser loading.

![CNN Training Metrics](training_plots.png)

![Confusion Matrix](confusion_matrix.png)

## Core Algorithms

### Eye Aspect Ratio (EAR)
EAR calculates the vertical distance between the eyelids against the horizontal width. Using MediaPipe's 468 3D facial landmarks:

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
```
If `EAR < 0.25`, the eyes are classified as closed.

### PERCLOS (Percentage of Eye Closure)
A rolling window evaluates the density of eye closures to prevent false alarms from standard blinking. If PERCLOS exceeds the safety threshold (70%), the system triggers a **FATIGUED** state.

## Web Application Dashboard
The frontend synthesizes TensorFlow.js execution, MediaPipe Face Mesh, and standard WebRTC camera interfacing. 

![Dashboard UI Mockup](ui_dashboard_mockup.png)

## Local Development
To run this project locally:
1. Clone the repository.
2. Navigate to the `webapp` directory.
3. Serve the directory using any static HTTP server (e.g., `python3 -m http.server 8000` or `npx serve`).
4. Access via `http://localhost:8000` (Camera permissions required).
