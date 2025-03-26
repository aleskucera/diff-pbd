export class AnimationController {
  constructor(app) {
    this.app = app;
    this.states = [];
    this.isPlaying = false;
    this.playbackSpeed = 1;
    this.isRecording = false;
    this.capturer = null;
    this.startTime = null;
    this.frameCount = 0;
    this.recordingFormat = "webm"; // Default recording format
    this.currentStateIndex = 0;

    // Add progress display
    this.progressElement = this.createProgressElement();
  }

  loadAnimation(states) {
    this.states = states;
    this.currentStateIndex = 0;
    this.animationStartTime = performance.now(); // Use performance.now() for more precise timing
  }

  // Check if we already have a frame at this time
  hasFrame(time) {
    return this.states.some((state) => Math.abs(state.time - time) < 0.0001);
  }

  // Add a new frame to the animation timeline
  addFrame(stateData) {
    this.states.push({
      time: stateData.time,
      bodies: stateData.bodies || [],
    });

    // Keep states sorted by time
    this.states.sort((a, b) => a.time - b.time);
  }

  play() {
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.animationStartTime = performance.now();
      this.animate();
    }
  }

  pause() {
    this.isPlaying = false;
  }

  setSpeed(speed) {
    this.playbackSpeed = speed;
  }

  setRecordingFormat(format) {
    this.recordingFormat = format;
  }

  stepForward() {
    if (this.currentStateIndex < this.states.length - 1) {
      this.currentStateIndex++;
      this.updateScene();
    }
  }

  stepBackward() {
    if (this.currentStateIndex > 0) {
      this.currentStateIndex--;
      this.updateScene();
    }
  }

  createProgressElement() {
    const progress = document.createElement("div");
    progress.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 10px;
      border-radius: 4px;
      font-family: monospace;
      display: none;
    `;
    document.body.appendChild(progress);
    return progress;
  }

  startRecording() {
    if (this.isRecording) return;

    // Reset animation to start
    this.currentStateIndex = 0;
    this.animationStartTime = performance.now();

    const options = {
      framerate: 30,
      verbose: true,
      motionBlurFrames: 1,
    };

    // Set format-specific options
    if (this.recordingFormat === "webm") {
      options.format = "webm";
      options.quality = 200;
    } else if (this.recordingFormat === "jpg") {
      options.format = "jpg";
      options.quality = 500;
    }

    // Initialize CCapture
    this.capturer = new CCapture(options);

    this.isRecording = true;
    this.frameCount = 0;
    this.startTime = performance.now();
    this.progressElement.style.display = "block";

    // Start recording and play animation if not already playing
    this.capturer.start();
    if (!this.isPlaying) {
      this.play();
    }
  }

  stopRecording() {
    if (!this.isRecording) return;

    this.isRecording = false;

    this.capturer.stop();
    this.capturer.save();
    this.progressElement.style.display = "none";

    // Reset recording state
    this.startTime = null;
    this.frameCount = 0;
  }

  animate() {
    if (!this.isPlaying) return;

    const currentTime = performance.now();
    const elapsedSeconds = (currentTime - this.animationStartTime) / 1000;
    const adjustedTime = elapsedSeconds * this.playbackSpeed;

    // Calculate total animation duration
    const totalDuration = this.states[this.states.length - 1].time;

    // Calculate the current time in the animation loop
    const loopTime = adjustedTime % totalDuration;

    // Find the appropriate state index
    let newStateIndex = 0;
    while (
      newStateIndex < this.states.length - 1 &&
      this.states[newStateIndex + 1].time <= loopTime
    ) {
      newStateIndex++;
    }

    // Update state if changed
    if (newStateIndex !== this.currentStateIndex) {
      this.currentStateIndex = newStateIndex;
      this.updateScene();
    }

    if (this.isRecording && this.capturer) {
      this.frameCount++;
      const elapsed = currentTime - this.startTime;
      const duration = totalDuration * 1000; // Convert to milliseconds

      // Update progress display
      const progress = Math.min(((elapsed / duration) * 100).toFixed(1), 100);
      this.progressElement.textContent = `Recording: ${progress}%`;

      this.capturer.capture(this.app.renderer.domElement);

      // Stop recording if we've completed one loop
      if (elapsed >= duration) {
        this.stopRecording();
      }
    }
    requestAnimationFrame(() => this.animate());
  }

  updateScene() {
    if (!this.app.bodies || !this.states[this.currentStateIndex]) return;

    const state = this.states[this.currentStateIndex];

    // Update all body states
    state.bodies.forEach((bodyState) => {
      const body = this.app.bodies.get(bodyState.name);
      if (body) {
        // Update the body state - this handles both batched and non-batched formats
        body.updateState(bodyState);
      }
    });

    // Update the body state window display
    if (this.app.bodyStateWindow) {
      this.app.bodyStateWindow.update();
    }
  }
}
