# Web Application Implementation Details
**AI-Based Driver Drowsiness Detection and Emergency Alert System**

---

## 1. Introduction
This document outlines the complete implementation of the driver drowsiness detection **Web Application**. The system runs fully in the browser without any remote backend capabilities. By utilizing pure computer vision (TensorFlow.js and MediaPipe), the web app actively processes live webcam feeds to evaluate a driver's state of alertness.

---

## 2. Dataset and Preprocessing
To train the neural network, we utilized the Kaggle **Driver Drowsiness Dataset (DDD)** which comprises approximately 41,000 images categorized into two classes:
- **Open Eye** (Driver is awake)
- **Closed Eye** (Driver is drowsy)

### Sample Images from our Processed Dataset:

**Open Eye Sample:**
![Processed Open Eye](file:///Users/anirudhchhabra/DriverDrow/dataset_processed/validation/open/h0294.png)

**Closed Eye Sample:**
![Processed Closed Eye](file:///Users/anirudhchhabra/DriverDrow/dataset_processed/validation/closed/A0001.png)

### Custom Preprocessing Pipeline
We built a custom Python script to prepare the image data for the CNN.
1. **Resizing:** All varied eye extractions were rigidly resized to `112x112 pixels`.
2. **Normalization:** Pixel values were scaled by `1/255` to sit bound between `0` and `1`.
3. **Data Augmentation:** Applied heavy rotation, zoom, horizontal flipping, and brightness variation to build robustness against dynamic lighting and varying head poses.
4. **Data Split:** 80% Training, 15% Validation, 5% Testing.

---

## 3. Training the Custom CNN
We deployed a Custom Convolutional Neural Network (CNN) architecture using TensorFlow and Keras. The model processes the `112x112` eye inputs and outputs a binary categorical determination (Open vs Closed).

### Hyperparameters:
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 16
- **Epochs:** 17 (Capped manually via Early Stopping)
- **Quantization:** Weights were converted to a strictly INT8 quantized `.tflite` (and subsequently `.json` / `.bin` structure for the WebApp) to shrink the spatial size down to merely `1.17 MB`.

### Training Diagnostics:
The network achieved a Training Accuracy of `~99.97%` and Testing Accuracy of `~99.86%`.
![CNN Training Loss and Accuracy Plot](file:///Users/anirudhchhabra/DriverDrow/training_plots.png)

---

## 4. Core Algorithms and Formulas Used

Our prediction pipeline inherently blends geometric facial tracking algorithms combined alongside our neural network. 

### 4.1 Eye Aspect Ratio (EAR)
The **Eye Aspect Ratio** (EAR) calculates the vertical distance between the eyelids divided by the horizontal distance across the eye. If the scalar value drops, it strictly indicates an eye closure.
Using the standardized 6 facial landmarks per eye ($p_1, \dots, p_6$ where $p_1$ and $p_4$ are the horizontal corners):

$$ EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||} $$

The constants dictate that an open eye generally sits at $EAR > 0.25$, whereas a closing eye yields $EAR < 0.21$.

### 4.2 PERCLOS (Percentage of Eye Closure)
PERCLOS actively polls the past $N$ frames to calculate the absolute percentage of time the eyes were shut. It acts as our smoothening temporal filter to prevent a standard "blink" from triggering a false alarm.

$$ PERCLOS = \frac{\text{Number of frames where EAR} < \text{Threshold}}{\text{Total Window Frame Size (e.g., 60 frames)}} \times 100 $$

If $PERCLOS > 20\%$, the alert system triggers.

---

## 5. WebApp Frontend Implementation

To orchestrate this logic seamlessly without installing native executables, we constructed a lightweight HTML/JS architecture that operates in-browser:

### The Real-Time Pipeline
1. **Camera Interfacing:** Interfaced with `navigator.mediaDevices.getUserMedia` to fetch a continuous 30 FPS video canvas.
2. **MediaPipe Face Mesh:** A CDN-loaded pre-compiled WebAssembly model actively extracts exactly 468 3D facial landmarks from the canvas buffer.
3. **Bounding Box Logic:** Derived the $X, Y$ scalar coordinates isolating the Left (landmarks: `33, 160, 158, 133, 153, 144`) and Right (landmarks: `362, 385, 387, 263, 373, 380`) eyes to paint dynamic red boundary boxes.
4. **TensorFlow.js Execution:** We implemented standard `tf.browser.fromPixels()` logic to crop these bounding box regions. They were resized to `112x112`, normalized, and passed synchronously to `window.tf_model.predict()`. 
5. **UI Rendering:** Variables are parsed and painted directly over the video utilizing TailwindCSS "Glassmorphism" layout mechanics.
6. **Geolocation Alerts:** Triggering phase-2 protocols polls the HTML5 `navigator.geolocation` payload to compile simulated SMS prompts.

By synthesizing these technologies, the WebApp executes inference latency well below 40ms, making it fully immune to network lag while ensuring absolute localized driver privacy.
