export function createPlaybackControls(animationController) {
  const container = document.createElement("div");
  Object.assign(container.style, {
    position: "absolute",
    bottom: "20px",
    left: "50%",
    transform: "translateX(-50%)",
    display: "flex",
    gap: "10px",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    padding: "10px",
    borderRadius: "5px",
  });

  // Play/Pause button
  const playButton = createButton("Play", () => {
    if (animationController.isPlaying) {
      animationController.pause();
      playButton.textContent = "Play";
    } else {
      animationController.play();
      playButton.textContent = "Pause";
    }
  });

  // Step backward button
  const stepBackButton = createButton("←", () => {
    animationController.pause();
    animationController.stepBackward();
    playButton.textContent = "Play";
  });

  // Step forward button
  const stepForwardButton = createButton("→", () => {
    animationController.pause();
    animationController.stepForward();
    playButton.textContent = "Play";
  });

  // Speed control
  const speedSelect = document.createElement("select");
  [0.1, 0.25, 0.5, 1, 2].forEach((speed) => {
    const option = document.createElement("option");
    option.value = speed;
    option.text = `${speed}x`;
    if (speed === 1) option.selected = true;
    speedSelect.appendChild(option);
  });
  speedSelect.addEventListener("change", (e) => {
    console.debug("Speed changed to", e.target.value);
    animationController.setSpeed(parseFloat(e.target.value));
  });

  // Frame counter
  const frameCounter = document.createElement("span");
  frameCounter.style.color = "white";

  // Update frame counter
  setInterval(() => {
    frameCounter.textContent = `Time: ${animationController.states[animationController.currentStateIndex].time.toFixed(2)} / ${animationController.states[animationController.states.length - 1].time.toFixed(2)}`;
  }, 100);

  [
    stepBackButton,
    playButton,
    stepForwardButton,
    speedSelect,
    frameCounter,
  ].forEach((element) => container.appendChild(element));

  document.body.appendChild(container);

  // Keyboard controls
  document.addEventListener("keydown", (event) => {
    switch (event.key.toLowerCase()) {
      case "n":
        animationController.stepForward();
        break;
      case "b":
        animationController.stepBackward();
        break;
      case " ":
        playButton.click();
        break;
    }
  });
}

function createButton(text, onClick) {
  const button = document.createElement("button");
  Object.assign(button.style, {
    padding: "5px 10px",
    backgroundColor: "#444",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
  });
  button.textContent = text;
  button.addEventListener("click", onClick);
  return button;
}
