import { initScene } from "./components/Scene.js";
import { setupWebSocket } from "./utils/WebSocket.js";
import { initUIControls } from "./ui/Controls.js";
import { setupWindowHandlers } from "./utils/WindowHandler.js";
import { AnimationController } from "./components/AnimationController.js";
import { APP_CONFIG } from "./config.js";

let app = {
  scene: null,
  camera: null,
  controls: null,
  renderer: null,
  animationController: null,

  // Body objects
  bodies: {},
  bodyVisualizationMode: APP_CONFIG.bodyVisualizationMode,

  // Contact points and normals
  contactPoints: {},
  contactPointsVisible: APP_CONFIG.contactPointsVisible,
  contactNormals: {},
  contactNormalsVisible: APP_CONFIG.contactNormalsVisible,
};

function init() {
  const sceneSetup = initScene();
  Object.assign(app, sceneSetup);

  setupWebSocket(app);
  setupWindowHandlers(app);

  initUIControls(app);

  app.animationController = new AnimationController(app);

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  app.renderer.render(app.scene, app.camera);
}

init();
