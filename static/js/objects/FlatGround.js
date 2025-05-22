import * as THREE from "three";
import { GridHelper } from "three";
import { GROUND_CONFIG } from "../config.js";

export class FlatGround {
  constructor() {
    this.group = new THREE.Group();
    this.group.name = "FlatGround";

    // Create grid helper
    this.grid = new GridHelper(
      GROUND_CONFIG.size,
      GROUND_CONFIG.divisions,
      GROUND_CONFIG.gridColor,
      GROUND_CONFIG.gridColor,
    );

    // Rotate grid to match Z-up coordinate system
    this.grid.rotation.set(-Math.PI / 2, 0, 0);

    // Create ground plane
    const groundGeometry = new THREE.PlaneGeometry(
      GROUND_CONFIG.size,
      GROUND_CONFIG.size,
    );

    const groundMaterial = new THREE.MeshPhongMaterial({
      color: GROUND_CONFIG.mainColor,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 1.0,
    });

    this.ground = new THREE.Mesh(groundGeometry, groundMaterial);

    // Rotate and position the ground to match the grid
    this.ground.rotation.set(...GROUND_CONFIG.rotation);
    this.ground.position.set(...GROUND_CONFIG.position);

    // Add both grid and ground to group
    this.group.add(this.grid);
    this.group.add(this.ground);
  }

  // Show/hide grid
  setGridVisible(visible) {
    this.grid.visible = visible;
  }

  // Show/hide ground plane
  setGroundVisible(visible) {
    this.ground.visible = visible;
  }

  // Implement common interface method
  getObject3D() {
    return this.group;
  }
}
