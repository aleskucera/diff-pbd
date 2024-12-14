import * as THREE from "three";
import { GridHelper } from "three";
import { setupControls } from "./Controls.js";
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
  setupGround(scene);

  return { renderer, scene, camera, controls };
}

function createRenderer() {
  const renderer = new THREE.WebGLRenderer({
    antialias: RENDERER_CONFIG.antialias,
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

function setupGround(scene) {
  // Create grid helper
  const grid = new GridHelper(
    GROUND_CONFIG.size,
    GROUND_CONFIG.divisions,
    GROUND_CONFIG.gridColor,
    GROUND_CONFIG.gridColor,
  );

  // Rotate grid to match Z-up coordinate system
  grid.rotation.set(-Math.PI / 2, 0, 0);

  // Create ground plane
  const groundGeometry = new THREE.PlaneGeometry(
    GROUND_CONFIG.size,
    GROUND_CONFIG.size,
  );

  const groundMaterial = new THREE.MeshPhongMaterial({
    color: GROUND_CONFIG.mainColor,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.5,
  });

  const ground = new THREE.Mesh(groundGeometry, groundMaterial);

  // Rotate and position the ground to match the grid
  ground.rotation.set(...GROUND_CONFIG.rotation);
  ground.position.set(...GROUND_CONFIG.position);

  // Add both grid and ground to scene
  scene.add(grid);
  scene.add(ground);

  return { grid, ground };
}
