// Dashboard Controller & 3D Analytics using Three.js

let scene, camera, renderer, globe;
let isDashboardActive = false;

// Machine Learning insights generator (derived from Real tracked data)
function analyzeDrivingPatterns() {
    // Generate AI Tip text based on ratio of asleeps/accidents/time
    let style = "Safe";
    let score = 95;
    let tip = "Your alertness looks good. Hydrate regularly.";
    let windowHint = "Midnight - 3:00 AM"; // Default historical assumption 

    const elapsedHrs = (window.appMetrics.totalDriveTimeSeconds / 3600);
    const asleepRatio = elapsedHrs > 0 ? (window.appMetrics.timesFellAsleep / elapsedHrs) : 0;

    if (window.appMetrics.alertsIgnored > 0) {
        style = "Aggressive / Risky";
        score = Math.max(0, 100 - (window.appMetrics.alertsIgnored * 15) - (window.appMetrics.timesFellAsleep * 5));
        tip = "You have ignored critical safety alerts. Please pull over as soon as possible.";
    } else if (asleepRatio > 2) {
        style = "Drowsy";
        score = Math.max(0, 100 - (window.appMetrics.timesFellAsleep * 5));
        tip = "You are frequently falling asleep! Stop driving and take a break.";
    } else if (elapsedHrs > 2) {
        style = "Fatigued";
        score -= 5;
        tip = "You have been driving a long time continuously. Consider resting.";
    }

    // Update DOM elements
    document.getElementById('dash-score').innerText = Math.floor(score);
    document.getElementById('dash-style').innerText = style;
    document.getElementById('dash-tip').innerText = tip;
    document.getElementById('dash-window').innerText = windowHint;

    // Format Weekly Driving Time (or "Total Driving" for this MVP)
    const hrs = Math.floor(window.appMetrics.totalDriveTimeSeconds / 3600);
    const mins = Math.floor((window.appMetrics.totalDriveTimeSeconds % 3600) / 60);
    document.getElementById('dash-weekly').innerText = `${hrs}h ${mins}m`;

    // Assign real metrics
    document.getElementById('dash-accidents').innerText = window.appMetrics.alertsIgnored;
    document.getElementById('dash-asleep').innerText = window.appMetrics.timesFellAsleep;
}

function init3D() {
    const canvas = document.getElementById('three-canvas');
    if (!canvas) return;

    // Setup scene
    scene = new THREE.Scene();

    // Setup camera
    const aspect = canvas.clientWidth / canvas.clientHeight;
    camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    camera.position.z = 5;

    // Setup renderer
    renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Create a wireframe sphere (representing the glowing AI core/score)
    const geometry = new THREE.SphereGeometry(2, 32, 32);
    const material = new THREE.MeshBasicMaterial({
        color: 0x00ffff,
        wireframe: true,
        transparent: true,
        opacity: 0.3
    });

    globe = new THREE.Mesh(geometry, material);
    scene.add(globe);

    // Add inner glowing solid core
    const innerGeo = new THREE.SphereGeometry(1.9, 32, 32);
    const innerMat = new THREE.MeshBasicMaterial({ color: 0x0f172a }); // Dark slate matching bg
    const innerGlobe = new THREE.Mesh(innerGeo, innerMat);
    scene.add(innerGlobe);

    // Animation Loop
    function animate() {
        if (!isDashboardActive) return; // Save resources when hidden

        requestAnimationFrame(animate);

        globe.rotation.y += 0.005;
        globe.rotation.x += 0.002;

        renderer.render(scene, camera);
    }

    // Handle Window Resize
    window.addEventListener('resize', () => {
        if (isDashboardActive) {
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        }
    });

    animate();
}

// Hook called by index.html Router when navigating to/from Dashboard
window.setDashboardActive = function (isActive) {
    isDashboardActive = isActive;

    // Force a resize calculation when showing
    if (isActive) {
        const canvas = document.getElementById('three-canvas');
        if (canvas) {
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }
    }
};

// Initialize 3D when script loads
init3D();

// Initialize when DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
    // Generate new patterns every minute while dashboard is open
    setInterval(() => {
        if (isDashboardActive) analyzeDrivingPatterns();
    }, 60000);
});
