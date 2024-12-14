import { GUI } from "three/addons/libs/lil-gui.module.min.js";

let appState = null;
let gui = null;

export function initUIControls(appInstance) {
  appState = appInstance;
  createDatGUI();
  // updateVisualizationMode(visualizationMode);
}

function createDatGUI() {
  // Create GUI instance
  gui = new GUI();

  // Create an object to hold our controls
  const controls = {
    bodyVisualizationMode: appState.bodyVisualizationMode,
    showContactPoints: appState.contactPointsVisible,
    showContactNormals: appState.contactNormalsVisible,
  };

  // Add controls to GUI
  const displayFolder = gui.addFolder("Display Options");

  // Add dropdown for visualization mode
  displayFolder
    .add(controls, "bodyVisualizationMode", ["mesh", "wireframe", "points"])
    .name("Body Visualization Mode")
    .onChange((value) => {
      updateVisualizationMode(value);
    });

  // Add checkbox for contact points visibility
  displayFolder
    .add(controls, "showContactPoints")
    .name("Show Contact Points")
    .onChange((value) => {
      updateContactPointsVisibility(value);
    });

  // Add checkbox for contact normals visibility
  displayFolder
    .add(controls, "showContactNormals")
    .name("Show Contact Normals")
    .onChange((value) => {
      updateContactNormalsVisibility(value);
    });

  displayFolder.open();
}

function updateVisualizationMode(mode) {
  Object.values(appState.bodies).forEach((obj) => {
    obj.children.forEach((child) => {
      if (child.isMesh) {
        child.visible = mode === "mesh";
      }
      if (child.isWireframe) {
        child.visible = mode === "wireframe";
      }
      if (child.isPoints) {
        child.visible = mode === "points";
      }
    });
  });
}

function updateContactPointsVisibility(show) {
  Object.values(appState.contactPoints).forEach((obj) => {
    obj.visible = show;
  });
  appState.contactPointsVisible = show;
}

function updateContactNormalsVisibility(show) {
  Object.values(appState.contactNormals).forEach((obj) => {
    obj.visible = show;
  });
  appState.contactNormalsVisible = show;
}

export function getControlState() {
  return {
    visualizationMode,
  };
}

export function destroyGUI() {
  if (gui) {
    gui.destroy();
    gui = null;
  }
}
