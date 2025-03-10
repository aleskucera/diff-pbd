import { initScene } from "./components/Scene.js";
import { setupWebSocket } from "./utils/WebSocket.js";
import { UIControls } from "./ui/Controls.js";
import { setupWindowHandlers } from "./utils/WindowHandler.js";
import { BodyStateWindow } from "./ui/BodyStateWindow.js";
import { AnimationController } from "./components/AnimationController.js";
import { InteractionController } from "./components/InteractionController.js";
import { APP_CONFIG } from "./config.js";

let app = {
  scene: null,
  camera: null,
  controls: null,
  renderer: null,
  uiControls: null,
  bodyStateWindow: null,
  animationController: null,
  interactionController: null,

  selectedObjects: new Set(),

  // Ground
  ground: null,
  terrain: null,

  // Body objects
  bodies: new Map(),
  selectedBodies: new Set(),

  // Controls
  axesVisible: APP_CONFIG.axesVisible,
  bodyVisualizationMode: APP_CONFIG.bodyVisualizationMode,
  contactPointsVisible: APP_CONFIG.contactPointsVisible,

  bodyVectorVisible: {
    linearVelocity: APP_CONFIG.bodyVectorVisible.linearVelocity,
    angularVelocity: APP_CONFIG.bodyVectorVisible.angularVelocity,
    linearForce: APP_CONFIG.bodyVectorVisible.linearForce,
    torque: APP_CONFIG.bodyVectorVisible.torque,
  },

  terrainVisualizationModes: {
    surface: APP_CONFIG.terrainVisualizationModes.surface,
    wireframe: APP_CONFIG.terrainVisualizationModes.wireframe,
  },
  terrainNormalsVisible: APP_CONFIG.terrainNormalsVisible,
  terrainColorMap: APP_CONFIG.terrainColorMap,
};

function init() {
  const sceneSetup = initScene();
  Object.assign(app, sceneSetup);

  setupWebSocket(app);
  setupWindowHandlers(app);

  app.uiControls = new UIControls(app);

  app.bodyStateWindow = new BodyStateWindow(app);
  app.animationController = new AnimationController(app);
  // app.interactionController = new InteractionController(app);

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  app.renderer.render(app.scene, app.camera);
}

init();
