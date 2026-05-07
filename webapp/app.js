const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

window.appMetrics = JSON.parse(localStorage.getItem('appMetrics')) || {
    totalDriveTimeSeconds: 0,
    timesFellAsleep: 0,
    alertsIgnored: 0
};

// Auto save and increment drive time every second
setInterval(() => {
    if (mediaPipeCamera) { // Only increment if camera is running
        window.appMetrics.totalDriveTimeSeconds++;
        localStorage.setItem('appMetrics', JSON.stringify(window.appMetrics));
    }
}, 1000);

let tfModel = null;
let isPredicting = false;

// UI Elements
const valEar = document.getElementById('val-ear');
const valFrames = document.getElementById('val-frames');
const valPerclos = document.getElementById('val-perclos');
const valCnn = document.getElementById('val-cnn');
const valFps = document.getElementById('val-fps');
const statusText = document.getElementById('status-text');
const statusIcon = document.getElementById('status-icon');
const btnStart = document.getElementById('btn-start');

// Drowsiness Variables
const EAR_THRESHOLD = 0.25;
const PERCLOS_WINDOW = 30; // frames
const CONSECUTIVE_FRAMES_WARNING = 60; // ~2 seconds at 30 fps

let eyeHistory = [];
let consecutiveClosedCount = 0;
let lastTime = performance.now();
let frames = 0;

// MediaPipe Landmark Indices for Eye Aspect Ratio calculation
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];
const LEFT_EYE = [33, 160, 158, 133, 153, 144];

async function loadTensorFlowModel() {
    try {
        console.log("Loading local TFJS Model...");
        // Serve tfjs_model folder contents
        tfModel = await tf.loadLayersModel('models/tfjs_model/model.json');
        console.log("Model loaded successfully!");
    } catch (e) {
        console.error("TF.js Model Load Error:", e);
        if (valCnn) {
            valCnn.innerText = "MODEL ERROR";
            valCnn.classList.replace("text-green-400", "text-red-500");
        }
    }
}

// Euclidean distance helper
function distance(p1, p2, width, height) {
    const x1 = p1.x * width;
    const y1 = p1.y * height;
    const x2 = p2.x * width;
    const y2 = p2.y * height;
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
}

function calculateEAR(landmarks, eyeIndices, width, height) {
    // vertical
    const cv1 = distance(landmarks[eyeIndices[1]], landmarks[eyeIndices[5]], width, height);
    const cv2 = distance(landmarks[eyeIndices[2]], landmarks[eyeIndices[4]], width, height);
    // horizontal
    const ch = distance(landmarks[eyeIndices[0]], landmarks[eyeIndices[3]], width, height);

    return (cv1 + cv2) / (2.0 * ch);
}

function updateHUD(status, alertLevel) {
    const banner = document.getElementById('status-banner');
    const statusText = document.getElementById('status-text');
    const iconContainer = document.getElementById('status-icon');

    // Reset alarm tracking if status goes back to Normal
    if (status === "Normal") {
        resetAlarmTracking();
    }

    if (alertLevel === 'CRITICAL') {

        statusIcon.className = "w-12 h-12 rounded-full bg-red-600 flex items-center justify-center pulse-red";
        statusIcon.innerHTML = `<svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`;
        playAlarm();
    } else if (alertLevel === 'DROWSY') {
        statusText.innerText = "DROWSY - WARNING";
        statusText.className = "text-2xl font-black tracking-widest text-orange-400";
        statusIcon.className = "w-12 h-12 rounded-full bg-orange-500 flex items-center justify-center shadow-[0_0_15px_rgba(249,115,22,0.5)]";
        statusIcon.innerHTML = `<svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
        playAlarm();
    } else {
        statusText.innerText = "AWAKE & ACTIVE";
        statusText.className = "text-2xl font-black tracking-widest text-green-400";
        statusIcon.className = "w-12 h-12 rounded-full bg-green-500 flex items-center justify-center shadow-[0_0_15px_rgba(34,197,94,0.5)]";
        statusIcon.innerHTML = `<svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>`;
    }
}

// HTML5 Simple Synth Alarm
let audioCtx;
let lastBeepTime = 0;
let alarmStartTime = 0;
let alertIgnoredLogged = false;

function playAlarm() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();

    // Track how long the alarm is playing to log ignored alerts
    if (alarmStartTime === 0) {
        alarmStartTime = performance.now();
        alertIgnoredLogged = false;

        // Log a 'fell asleep' event as soon as the alarm triggers (which means they hit DROWSY/SLEEP)
        window.appMetrics.timesFellAsleep++;
        localStorage.setItem('appMetrics', JSON.stringify(window.appMetrics));
    } else if (!alertIgnoredLogged && (performance.now() - alarmStartTime) > 10000) {
        // If alarm plays continuously for > 10 seconds, count as an ignored alert
        window.appMetrics.alertsIgnored++;
        localStorage.setItem('appMetrics', JSON.stringify(window.appMetrics));
        alertIgnoredLogged = true;
    }

    // throttle beeps
    if (Date.now() - lastBeepTime < 1000) return;
    lastBeepTime = Date.now();

    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    osc.type = 'square';
    osc.frequency.setValueAtTime(800, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.1);

    gainNode.gain.setValueAtTime(1, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);

    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.5);
}

// Custom handler if driver wakes up to reset alarm timer
function resetAlarmTracking() {
    alarmStartTime = 0;
    alertIgnoredLogged = false;
}


async function cropAndPredict(imageElement, landmarks) {
    if (!tfModel || isPredicting) return false;
    isPredicting = true;

    try {
        // In a full implementation, you'd extract the eye bounding box here using canvas drawImage
        // Because MediaPipe provides ultra-accurate landmarks for EAR natively in JS, 
        // we'll run a lightweight placeholder prediction visualization for the neural net block 
        // to save main-thread framerates vs transferring buffers to WebGL continuously.

        // Simulating the CNN inference for Demo visualization based on geometric pre-computation 
        // (If EAR < Threshold, tell CNN context it is likely closed to augment).
        return new Promise((resolve) => {
            setTimeout(() => {
                isPredicting = false;
                resolve(false); // Return false = OPEN mock
            }, 10);
        });
    } catch (e) {
        console.error(e);
        isPredicting = false;
        return false;
    }
}

function onResults(results) {
    // FPS Calc
    frames++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
        valFps.innerText = (frames * 1000 / (now - lastTime)).toFixed(1);
        frames = 0;
        lastTime = now;
    }

    // Reset Canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];

        // Draw Face Mesh
        drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION, { color: '#ffffff30', lineWidth: 1 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, { color: '#ef4444', lineWidth: 2 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, { color: '#ef4444', lineWidth: 2 });
        drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, { color: '#60a5fa', lineWidth: 2 });

        // Calculations
        const width = canvasElement.width;
        const height = canvasElement.height;

        let rightEAR = calculateEAR(landmarks, RIGHT_EYE, width, height);
        let leftEAR = calculateEAR(landmarks, LEFT_EYE, width, height);
        let avgEAR = (leftEAR + rightEAR) / 2.0;

        valEar.innerText = avgEAR.toFixed(3);

        // Logic
        let isClosed = avgEAR < EAR_THRESHOLD;

        eyeHistory.push(isClosed);
        if (eyeHistory.length > PERCLOS_WINDOW) eyeHistory.shift();

        if (isClosed) {
            consecutiveClosedCount++;
        } else {
            consecutiveClosedCount = 0;
        }

        valFrames.innerText = consecutiveClosedCount;

        // PERCLOS
        let closedFramesCount = eyeHistory.filter(Boolean).length;
        let perclosPercent = (closedFramesCount / eyeHistory.length) * 100.0;
        valPerclos.innerText = perclosPercent.toFixed(1) + "%";

        if (perclosPercent > 70) {
            valPerclos.className = "metric-value text-red-500 font-black";
        } else {
            valPerclos.className = "metric-value text-orange-400";
        }

        // CNN Integration Check Placeholder (Asynchronous block)
        if (isClosed) {
            if (valCnn) {
                valCnn.innerText = "CLOSED";
                valCnn.className = "metric-value text-red-500 font-bold";
            }
        } else {
            if (valCnn) {
                valCnn.innerText = "OPEN";
                valCnn.className = "metric-value text-green-400";
            }
        }

        // State Machine
        if (consecutiveClosedCount >= CONSECUTIVE_FRAMES_WARNING) {
            updateHUD("Dormant", "CRITICAL");
        } else if (perclosPercent >= 70 && eyeHistory.length >= PERCLOS_WINDOW / 2) {
            updateHUD("Fatigued", "DROWSY");
        } else {
            updateHUD("Active", "AWAKE");
        }

    } else {
        // No Face
        statusText.innerText = "NO FACE DETECTED";
        statusText.className = "text-2xl font-black tracking-widest text-slate-500";
        statusIcon.className = "w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center";
        statusIcon.innerHTML = `<svg class="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg>`;
    }
    canvasCtx.restore();
}

const faceMesh = new FaceMesh({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }
});
faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});
faceMesh.onResults(onResults);

let mediaPipeCamera = null;

document.addEventListener("DOMContentLoaded", () => {
    const btnStart = document.getElementById('btn-start');

    if (btnStart) {
        btnStart.addEventListener('click', () => {
            btnStart.innerText = "Initializing Camera...";
            btnStart.disabled = true;

            // Match video element aspect ratio dynamically
            const container = document.getElementById('video-container');
            canvasElement.width = container.clientWidth;
            canvasElement.height = container.clientHeight;

            loadTensorFlowModel().then(() => {
                mediaPipeCamera = new Camera(videoElement, {
                    onFrame: async () => {
                        await faceMesh.send({ image: videoElement });
                    },
                    width: 1280,
                    height: 720
                });
                mediaPipeCamera.start().then(() => {
                    btnStart.style.display = "none";
                    window.cameraActive = true;
                    document.getElementById('status-banner').style.display = 'flex';
                    document.getElementById('analytics-panel').style.display = 'block';
                    document.getElementById('controls-panel').style.display = 'block';
                });
            });
        });
    }

    const btnTestAlert = document.getElementById('btn-test-alert');
    if (btnTestAlert) {
        btnTestAlert.addEventListener('click', () => {
            // Simulate finding GPS
            alert("Emergency SOS Triggered! \n\nAcquiring GPS Signal... \nLocation Data Generated. \nSending SMS to Emergency Contacts: 'EMERGENCY: Driver is unresponsive. Location: https://maps.google.com/?q=37.7749,-122.4194'");
        });
    }
});
