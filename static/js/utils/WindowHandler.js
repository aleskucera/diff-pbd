import { CAMERA_CONFIG } from "../config.js";

export function setupWindowHandlers(app) {
  // Add window resize handler
  window.addEventListener("resize", () => handleWindowResize(app));

  // Add error handler
  window.addEventListener("error", handleError);

  // Add keyboard controls (optional)
  window.addEventListener("keydown", (event) => handleKeyboard(event, app));

  // Cleanup function
  return () => {
    window.removeEventListener("resize", () => handleWindowResize(app));
    window.removeEventListener("error", handleError);
    window.removeEventListener("keydown", (event) =>
      handleKeyboard(event, app),
    );
  };
}

function handleWindowResize(app) {
  const { camera, renderer } = app;

  // Update camera
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  // Update renderer
  renderer.setSize(window.innerWidth, window.innerHeight);

  // Optional: handle DPI scaling
  handleDPIScaling(app);
}

function handleDPIScaling(app) {
  const { renderer } = app;
  const pixelRatio = window.devicePixelRatio;
  const width = window.innerWidth;
  const height = window.innerHeight;

  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(width, height, false);
}

function handleError(event) {
  console.error("Application Error:", {
    message: event.message,
    source: event.filename,
    lineNumber: event.lineno,
    columnNumber: event.colno,
    error: event.error,
  });
}

function handleKeyboard(event, app) {
  // Example keyboard controls
  switch (event.key.toLowerCase()) {
    case "r": // Reset camera
      resetCamera(app);
      break;
    case "h": // Toggle help overlay
      toggleHelpOverlay();
      break;
    // Add more keyboard shortcuts as needed
  }
}

function resetCamera(app) {
  const { camera, controls } = app;

  // Reset camera to initial position
  camera.position.set(...CAMERA_CONFIG.position);
  camera.up.set(...CAMERA_CONFIG.up);

  // Update controls target
  controls.target.set(0, 0, 0);
  controls.update();
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch((err) => {
      console.error(`Error attempting to enable fullscreen: ${err.message}`);
    });
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }
}

// Optional: Help overlay for keyboard controls
function toggleHelpOverlay() {
  let helpOverlay = document.getElementById("help-overlay");

  if (!helpOverlay) {
    helpOverlay = createHelpOverlay();
    document.body.appendChild(helpOverlay);
  }

  helpOverlay.style.display =
    helpOverlay.style.display === "none" ? "block" : "none";
}

function createHelpOverlay() {
  const overlay = document.createElement("div");
  overlay.id = "help-overlay";
  overlay.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 20px;
        border-radius: 5px;
        z-index: 1000;
    `;

  overlay.innerHTML = `
        <h3>Keyboard Controls</h3>
        <ul>
            <li>R - Reset camera</li>
            <li>F - Toggle fullscreen</li>
            <li>H - Toggle this help menu</li>
        </ul>
        <p>Click anywhere to close</p>
    `;

  overlay.addEventListener("click", () => {
    overlay.style.display = "none";
  });

  return overlay;
}

// Optional: Add performance monitoring
function setupPerformanceMonitoring() {
  if (process.env.NODE_ENV === "development") {
    const stats = new Stats();
    document.body.appendChild(stats.dom);
    return stats;
  }
  return null;
}

// Optional: Add responsive design handling
function handleResponsiveDesign(app) {
  const breakpoints = {
    mobile: 480,
    tablet: 768,
    desktop: 1024,
  };

  const width = window.innerWidth;

  if (width <= breakpoints.mobile) {
    adjustForMobile(app);
  } else if (width <= breakpoints.tablet) {
    adjustForTablet(app);
  } else {
    adjustForDesktop(app);
  }
}

function adjustForMobile(app) {
  const { camera, controls } = app;
  // Adjust camera and controls for mobile view
  controls.enableZoom = true;
  controls.enableRotate = true;
  controls.enablePan = false;
}

function adjustForTablet(app) {
  const { controls } = app;
  // Adjust controls for tablet view
  controls.enableZoom = true;
  controls.enableRotate = true;
  controls.enablePan = true;
}

function adjustForDesktop(app) {
  const { controls } = app;
  // Enable all controls for desktop view
  controls.enableZoom = true;
  controls.enableRotate = true;
  controls.enablePan = true;
}

// Export additional utilities if needed
export const WindowUtils = {
  toggleFullscreen,
  resetCamera,
  toggleHelpOverlay,
};
