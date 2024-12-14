let showWireframe = true;
let showPoints = true;
let appState = null;

export function initUIControls(appInstance) {
  appState = appInstance; // Store the app instance
  createControlPanel();
  createKeyboardControls();
}

function createControlPanel() {
  const container = createControlContainer();

  // Create and add wireframe toggle button
  const wireframeButton = createButton("Toggle Wireframe/Mesh", () =>
    toggleWireframe(),
  );
  container.appendChild(wireframeButton);

  // Create and add points toggle button
  const pointsButton = createButton(
    "Toggle Points",
    () => togglePoints(),
    "10px", // marginLeft
  );
  container.appendChild(pointsButton);

  document.body.appendChild(container);
}

function createControlContainer() {
  const container = document.createElement("div");
  Object.assign(container.style, {
    position: "absolute",
    top: "20px",
    left: "20px",
    zIndex: "100",
    display: "flex",
    gap: "10px",
  });
  return container;
}

function createButton(text, onClick, marginLeft = "0px") {
  const button = document.createElement("button");
  Object.assign(button.style, {
    padding: "8px 16px",
    backgroundColor: "#444",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    marginLeft: marginLeft,
  });

  button.textContent = text;
  button.addEventListener("click", onClick);

  // Hover effects
  button.addEventListener("mouseover", () => {
    button.style.backgroundColor = "#555";
  });
  button.addEventListener("mouseout", () => {
    button.style.backgroundColor = "#444";
  });

  return button;
}

function toggleWireframe() {
  showWireframe = !showWireframe;
  Object.values(appState.objects).forEach((obj) => {
    obj.children.forEach((child) => {
      if (child.isWireframe || child.isMesh) {
        child.visible = child.isWireframe ? showWireframe : !showWireframe;
      }
    });
  });
}

function togglePoints() {
  showPoints = !showPoints;
  Object.values(appState.objects).forEach((obj) => {
    obj.children.forEach((child) => {
      if (child.isPoints) {
        child.visible = showPoints;
      }
    });
  });
}

function createKeyboardControls() {
  window.addEventListener("keydown", (event) => {
    switch (event.key.toLowerCase()) {
      case "w":
        toggleWireframe();
        break;
      case "p":
        togglePoints();
        break;
    }
  });
}

export function getControlState() {
  return {
    showWireframe,
    showPoints,
  };
}
