import { initScene } from "./components/Scene.js";
import { setupWebSocket } from "./utils/WebSocket.js";
import { initUIControls } from "./ui/Controls.js";
import { setupWindowHandlers } from "./utils/WindowHandler.js";
import { SelectionWindow } from "./ui/SelectionWindow.js";
import { AnimationController } from "./components/AnimationController.js";
import { InteractionController } from "./components/InteractionController.js";
import { APP_CONFIG } from "./config.js";

let app = {
  scene: null,
  camera: null,
  controls: null,
  renderer: null,
  selectionWindow: null,
  animationController: null,
  interactionController: null,

  selectedObjects: new Set(),

  // Body objects
  bodies: {},
  bodyVisualizationMode: APP_CONFIG.bodyVisualizationMode,
  selectedBodies: new Set(),

  // Contact points and normals
  contactPoints: {},
  contactPointsVisible: APP_CONFIG.contactPointsVisible,
  selectedContactPoints: new Set(),

  contactNormals: {},
  contactNormalsVisible: APP_CONFIG.contactNormalsVisible,
};

function init() {
  const sceneSetup = initScene();
  Object.assign(app, sceneSetup);

  setupWebSocket(app);
  setupWindowHandlers(app);

  initUIControls(app);

  app.selectionWindow = new SelectionWindow(app);
  app.animationController = new AnimationController(app);
  app.interactionController = new InteractionController(app);

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  app.renderer.render(app.scene, app.camera);
}

init();
