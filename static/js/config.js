export const APP_CONFIG = {
  bodyVisualizationMode: "mesh",
  contactPointsVisible: false,
  contactNormalsVisible: false,
};

export const CONTROLS_CONFIG = {
  minDistance: 1,
  maxDistance: 500,
  enableDamping: true,
  dampingFactor: 0.05,
  screenSpacePanning: true,
  maxPolarAngle: Math.PI,
  enablePan: true,
  panSpeed: 2.0,
  rotateSpeed: 1.5,
  zoomSpeed: 1.2,
};

export const SCENE_CONFIG = {
  defaultUp: [0, 0, 1],
};

export const RENDERER_CONFIG = {
  antialias: true,
  pixelRatio: window.devicePixelRatio,
  clearColor: 0x000000,
  clearAlpha: 0.0,
};

export const CAMERA_CONFIG = {
  fov: 40,
  near: 0.1,
  far: 500,
  position: [-15, -15, 10],
  up: [0, 0, 1],
};

export const LIGHTING_CONFIG = {
  ambient: {
    color: 0xffffff,
    intensity: 0.9,
  },
  directional: {
    color: 0xffffff,
    intensity: 0.8,
    position: [10, 10, 10],
  },
};

export const GROUND_CONFIG = {
  size: 1000, // Size of the ground plane
  divisions: 100, // Number of grid divisions
  mainColor: 0x444444, // Main ground color
  gridColor: 0x888888, // Grid line color
  position: [0, 0, 0], // Position of the ground plane
  rotation: [0, 0, 0], // Rotation of the ground plane
};

export const BODY_CONFIG = {
  geometry: {
    box: {
      widthSegments: 4,
      heightSegments: 4,
      depthSegments: 4,
    },
    sphere: {
      widthSegments: 16,
      heightSegments: 16,
    },
    cylinder: {
      radialSegments: 32,
      heightSegments: 4,
    },
  },
  wireframe: {
    color: 0x4080ff,
  },
  mesh: {
    color: 0xc1c1c1,
    roughness: 0.8,
    metalness: 0.9,
    opacity: 1.0,
    envMapIntensity: 0.5,
  },
  points: {
    size: 0.2,
    opacity: 0.7,
    texture: "static/textures/points/ball1.png",
    transparent: false,
  },
  maps: {
    env: "static/textures/cube/SwedishRoyalCastle/",
  },
};

export const CONTACT_CONFIG = {
  points: {
    size: 0.5,
    opacity: 1.0,
    texture: "static/textures/contacts/red-cross2.png",
    transparent: false,
  },
  normals: {
    color: 0xff0000,
  },
};

export const SELECTION_CONFIG = {
  BODIES: {
    key: "ctrl",
    set: "selectedBodies",
    objects: "bodies",
  },
  CONTACT_POINTS: {
    key: "Alt",
    set: "selectedContactPoints",
    objects: "contactPoints",
  },
};
