import * as THREE from "three";
import { CONTACT_CONFIG } from "../config.js";

export function createContactPoints(pointsArray) {
  if (pointsArray.length === 0) return null;

  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(pointsArray.flat());

  const sprite = new THREE.TextureLoader().load(CONTACT_CONFIG.points.texture);
  sprite.colorSpace = THREE.SRGBColorSpace;

  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );

  const material = new THREE.PointsMaterial({
    size: CONTACT_CONFIG.points.size,
    sizeAttenuation: true,
    opacity: CONTACT_CONFIG.points.opacity,
    map: sprite,
    alphaTest: 0.5,
    transparent: CONTACT_CONFIG.points.transparent,
  });

  const points = new THREE.Points(geometry, material);
  points.isPoints = true;
  return points;
}

export function createContactNormals(normals) {
  const group = new THREE.Group();
  normals.forEach((normal) => {
    const start = new THREE.Vector3().fromArray(normal.start);
    const end = new THREE.Vector3().fromArray(normal.end);
    const arrow = createArrow(start, end, CONTACT_CONFIG.normals.color);
    group.add(arrow);
  });
  return group;
}

export function createPoints(pointsArray, params) {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(pointsArray.flat());

  const sprite = new THREE.TextureLoader().load(params.texture);
  sprite.colorSpace = THREE.SRGBColorSpace;

  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );

  const material = new THREE.PointsMaterial({
    size: params.size,
    sizeAttenuation: true,
    opacity: params.opacity,
    map: sprite,
    alphaTest: 0.5,
    transparent: params.transparent,
  });

  const points = new THREE.Points(geometry, material);
  points.isPoints = true;
  points.visible = params.visible;

  return points;
}

export function createArrow(start, end, color) {
  const dir = end.clone().sub(start);
  const length = dir.length();
  const arrow = new THREE.ArrowHelper(dir.normalize(), start, length, color);
  return arrow;
}
