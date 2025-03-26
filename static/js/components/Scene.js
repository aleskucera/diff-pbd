import * as THREE from "three";
import { GridHelper } from "three";
import { setupControls } from "./InteractionControls.js";
import {
  SCENE_CONFIG,
  RENDERER_CONFIG,
  CAMERA_CONFIG,
  LIGHTING_CONFIG,
  GROUND_CONFIG,
} from "../config.js";

export function initScene() {
  THREE.Object3D.DEFAULT_UP.set(...SCENE_CONFIG.defaultUp);
  const scene = new THREE.Scene();
  const renderer = createRenderer();
  const camera = createCamera();
  const controls = setupControls(camera, renderer);

  setupLighting(scene);

  return { renderer, scene, camera, controls };
}

function createRenderer() {
  const renderer = new THREE.WebGLRenderer({
    antialias: RENDERER_CONFIG.antialias,
    preserveDrawingBuffer: RENDERER_CONFIG.preserveDrawingBuffer,
  });

  renderer.setPixelRatio(RENDERER_CONFIG.pixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(
    RENDERER_CONFIG.clearColor,
    RENDERER_CONFIG.clearAlpha,
  );

  document.body.appendChild(renderer.domElement);
  return renderer;
}

function createCamera() {
  const camera = new THREE.PerspectiveCamera(
    CAMERA_CONFIG.fov,
    window.innerWidth / window.innerHeight,
    CAMERA_CONFIG.near,
    CAMERA_CONFIG.far,
  );

  camera.position.set(...CAMERA_CONFIG.position);
  camera.up.set(...CAMERA_CONFIG.up);

  return camera;
}

function setupLighting(scene) {
  // Ambient light
  const ambientLight = new THREE.AmbientLight(
    LIGHTING_CONFIG.ambient.color,
    LIGHTING_CONFIG.ambient.intensity,
  );
  scene.add(ambientLight);

  // Directional light
  const directionalLight = new THREE.DirectionalLight(
    LIGHTING_CONFIG.directional.color,
    LIGHTING_CONFIG.directional.intensity,
  );
  directionalLight.position.set(...LIGHTING_CONFIG.directional.position);
  scene.add(directionalLight);
}
