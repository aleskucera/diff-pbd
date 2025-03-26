import * as THREE from "three";
import { APP_CONFIG, BODY_CONFIG, BODY_VECTOR_CONFIG } from "../config.js";
import {
  createGeometry,
  createMesh,
  createPoints,
  createWireframe,
  createContactPoints,
  createArrow,
} from "./utils.js";

export class Body {
  constructor(bodyData, app) {
    this.app = app;
    this.name = bodyData.name;
    this.batchCount = app.batchCount;
    this.visualizations = new Map();

    // Arrays to store batch-specific data
    this.batchGroups = [];
    this.positions = Array(this.batchCount)
      .fill()
      .map(() => new THREE.Vector3());
    this.quaternions = Array(this.batchCount)
      .fill()
      .map(() => new THREE.Quaternion());
    this.rotations = Array(this.batchCount)
      .fill()
      .map(() => new THREE.Euler());
    this.linearVelocities = Array(this.batchCount)
      .fill()
      .map(() => new THREE.Vector3());
    this.angularVelocities = Array(this.batchCount)
      .fill()
      .map(() => new THREE.Vector3());
    this.linearForces = Array(this.batchCount)
      .fill()
      .map(() => new THREE.Vector3());
    this.torques = Array(this.batchCount)
      .fill()
      .map(() => new THREE.Vector3());
    this.activeContacts = Array(this.batchCount)
      .fill()
      .map(() => new Set());

    this.initializeGroup();
    this.createBatchGroups(bodyData);
    this.updateState(bodyData);
  }

  initializeGroup() {
    this.group = new THREE.Group();
    this.group.name = this.name;
  }

  createBatchGroups(bodyData) {
    // Create a separate group for each batch
    for (let i = 0; i < this.batchCount; i++) {
      const batchGroup = new THREE.Group();
      batchGroup.name = `${this.name}_batch_${i}`;
      this.group.add(batchGroup);
      this.batchGroups.push(batchGroup);

      // Create visualizations for this batch
      this.createVisualRepresentations(batchGroup, bodyData);

      // Create batch-specific axes
      const axes = new THREE.AxesHelper(1);
      axes.visible = APP_CONFIG.axesVisible;
      batchGroup.add(axes);

      // Create batch-specific contact points
      if (bodyData.bodyPoints?.length) {
        this.initializeContactPoints(batchGroup, bodyData.bodyPoints, i);
      }

      // Create batch-specific vectors
      this.initializeBodyVectors(batchGroup, BODY_VECTOR_CONFIG, i);
    }
  }

  // Modified to support a specific batch group
  createVisualRepresentations(batchGroup, bodyData) {
    this.geometry = createGeometry(bodyData.shape, BODY_CONFIG.geometry);

    const representations = {
      mesh: { object: createMesh(this.geometry, BODY_CONFIG.mesh) },
      wireframe: {
        object: createWireframe(this.geometry, BODY_CONFIG.wireframe),
      },
      points: { object: createPoints(bodyData.bodyPoints, BODY_CONFIG.points) },
    };

    for (const [name, repr] of Object.entries(representations)) {
      if (repr.object) {
        repr.object.visible = name === APP_CONFIG.bodyVisualizationMode;
        batchGroup.add(repr.object);
        if (!this.visualizations.has(name)) {
          this.visualizations.set(name, []);
        }
        this.visualizations.get(name).push(repr.object);
      }
    }
  }

  // Modified to handle batch-specific contact points
  initializeContactPoints(batchGroup, bodyPoints, batchIndex) {
    if (!bodyPoints?.length) return;

    const contactPoints = createContactPoints(
      bodyPoints,
      BODY_CONFIG.contactPoints,
    );
    if (!contactPoints) return;

    const pointCount = bodyPoints.length;
    const contactPointSizes = new Float32Array(pointCount);
    contactPointSizes.fill(0);

    contactPoints.geometry.setAttribute(
      "size",
      new THREE.Float32BufferAttribute(contactPointSizes, 1),
    );

    batchGroup.add(contactPoints);
    contactPoints.visible = APP_CONFIG.contactPointsVisible;

    // Store batch-specific contact points
    if (!this.contactPoints) this.contactPoints = [];
    this.contactPoints[batchIndex] = contactPoints;

    if (!this.contactPointSizes) this.contactPointSizes = [];
    this.contactPointSizes[batchIndex] = contactPointSizes;
  }

  // Modified to handle batch-specific vectors
  initializeBodyVectors(batchGroup, vectorConfigs, batchIndex) {
    if (!this.bodyVectors)
      this.bodyVectors = Array(this.batchCount)
        .fill()
        .map(() => new Map());

    for (const [name, config] of Object.entries(vectorConfigs)) {
      const vector = createArrow(
        new THREE.Vector3(),
        new THREE.Vector3(0, 1, 0),
        config,
      );
      vector.visible = APP_CONFIG.bodyVectorVisible[name];
      vector.userData = { scale: config.scale };

      batchGroup.add(vector);
      this.bodyVectors[batchIndex].set(name, vector);
    }
  }

  setTransform(transform, batchIndex = 0) {
    if (!transform || transform.length < 7) return;

    const [x, y, z, qw, qx, qy, qz] = transform;

    // Update stored properties
    this.positions[batchIndex].set(x, y, z);
    this.quaternions[batchIndex].set(qx, qy, qz, qw);
    this.rotations[batchIndex].setFromQuaternion(this.quaternions[batchIndex]);

    // Update scene object
    if (this.batchGroups[batchIndex]) {
      // Store original position without offset
      const originalPosition = { x, y, z };
      this.batchGroups[batchIndex].userData.originalPosition = originalPosition;

      // Set base position and rotation
      this.batchGroups[batchIndex].quaternion.set(qx, qy, qz, qw);

      // Apply position with batch offset if available
      if (this.app.batchManager) {
        const offset = this.app.batchManager.getBatchOffset(batchIndex);
        this.batchGroups[batchIndex].position.set(
          x + offset.x,
          y + offset.y,
          z + offset.z,
        );
      } else {
        this.batchGroups[batchIndex].position.set(x, y, z);
      }
    }
  }

  // Set velocity for a specific batch
  setVelocity(velocity, batchIndex = 0) {
    if (!velocity || velocity.length < 6) return;

    const [vx, vy, vz, wx, wy, wz] = velocity;

    this.linearVelocities[batchIndex].set(vx, vy, vz);
    this.angularVelocities[batchIndex].set(wx, wy, wz);

    this.updateBodyVector(
      "linearVelocity",
      this.linearVelocities[batchIndex],
      batchIndex,
    );
    this.updateBodyVector(
      "angularVelocity",
      this.angularVelocities[batchIndex],
      batchIndex,
    );
  }

  // Set force for a specific batch
  setForce(force, batchIndex = 0) {
    if (!force || force.length < 6) return;

    const [fx, fy, fz, tx, ty, tz] = force;

    this.linearForces[batchIndex].set(fx, fy, fz);
    this.torques[batchIndex].set(tx, ty, tz);

    this.updateBodyVector(
      "linearForce",
      this.linearForces[batchIndex],
      batchIndex,
    );
    this.updateBodyVector("torque", this.torques[batchIndex], batchIndex);
  }

  // Update a body vector for a specific batch
  updateBodyVector(type, vector, batchIndex = 0) {
    if (!this.bodyVectors || !this.bodyVectors[batchIndex]) return;

    const arrow = this.bodyVectors[batchIndex].get(type);
    if (!arrow) return;

    const scale = arrow.userData.scale;
    const length = vector.length() * scale;

    const normalizedVector = vector.clone().normalize();

    arrow.setDirection(normalizedVector);
    arrow.setLength(length, length * 0.2, length * 0.1);
  }

  // Update contact points for a specific batch
  updateContactPointsVisibility(contactIndices, batchIndex = 0) {
    if (
      !this.contactPoints ||
      !this.contactPoints[batchIndex] ||
      !this.contactPointSizes ||
      !this.contactPointSizes[batchIndex]
    )
      return;

    // Reset all sizes to 0
    this.contactPointSizes[batchIndex].fill(0);

    // Set size for active contacts
    contactIndices.forEach((index) => {
      if (index < this.contactPointSizes[batchIndex].length) {
        this.contactPointSizes[batchIndex][index] =
          BODY_CONFIG.contactPoints.size;
        this.activeContacts[batchIndex].add(index);
      }
    });

    // Update the buffer
    const sizeAttribute =
      this.contactPoints[batchIndex].geometry.getAttribute("size");
    sizeAttribute.array = this.contactPointSizes[batchIndex];
    sizeAttribute.needsUpdate = true;
  }

  // Apply batch offset to a specific batch group
  applyBatchOffset(batchIndex) {
    if (
      !this.app.batchManager ||
      batchIndex < 0 ||
      batchIndex >= this.batchCount
    )
      return;

    // Get the offset for this batch
    const offset = this.app.batchManager.getBatchOffset(batchIndex);

    // Apply offset to the batch group's position
    if (this.batchGroups && this.batchGroups[batchIndex]) {
      const group = this.batchGroups[batchIndex];

      // Store original position without offset if not already stored
      if (!group.userData.originalPosition) {
        group.userData.originalPosition = {
          x: this.positions[batchIndex].x,
          y: this.positions[batchIndex].y,
          z: this.positions[batchIndex].z,
        };
      }

      const originalPos = group.userData.originalPosition;

      // Apply offset to the group's position
      group.position.set(
        originalPos.x + offset.x,
        originalPos.y + offset.y,
        originalPos.z + offset.z,
      );
    }
  }

  // Main update method - processes state with batch data
  updateState(bodyState) {
    // Check for batched transforms
    if (bodyState.transform && Array.isArray(bodyState.transform)) {
      for (
        let i = 0;
        i < Math.min(this.batchCount, bodyState.transform.length);
        i++
      ) {
        this.setTransform(bodyState.transform[i], i);
      }
    } else if (bodyState.transform) {
      this.setTransform(bodyState.transform);
    }

    // Check for batched velocities
    if (bodyState.velocity && Array.isArray(bodyState.velocity)) {
      for (
        let i = 0;
        i < Math.min(this.batchCount, bodyState.velocity.length);
        i++
      ) {
        this.setVelocity(bodyState.velocity[i], i);
      }
    } else if (bodyState.velocity) {
      this.setVelocity(bodyState.velocity);
    }

    // Check for batched forces
    if (bodyState.force && Array.isArray(bodyState.force)) {
      for (
        let i = 0;
        i < Math.min(this.batchCount, bodyState.force.length);
        i++
      ) {
        this.setForce(bodyState.force[i], i);
      }
    } else if (bodyState.force) {
      this.setForce(bodyState.force);
    }

    // Check for batched contacts
    if (bodyState.contacts && Array.isArray(bodyState.contacts)) {
      for (
        let i = 0;
        i < Math.min(this.batchCount, bodyState.contacts.length);
        i++
      ) {
        this.updateContactPointsVisibility(bodyState.contacts[i], i);
      }
    }
    // Legacy contact support
    else if (bodyState.contacts) {
      this.updateContactPointsVisibility(bodyState.contacts, 0);
    }
  }

  // Display control methods - apply to all batches
  updateVisualizationMode(mode) {
    for (const [type, objects] of this.visualizations.entries()) {
      for (const object of objects) {
        object.visible = type === mode;
      }
    }
  }

  toggleContactPoints(visible) {
    if (!this.contactPoints) return;

    for (let i = 0; i < this.batchCount; i++) {
      if (this.contactPoints[i]) {
        this.contactPoints[i].visible = visible;
      }
    }
  }

  toggleAxes(visible) {
    if (!this.batchGroups) return;

    for (const group of this.batchGroups) {
      const axes = group.children.find(
        (child) => child instanceof THREE.AxesHelper,
      );
      if (axes) axes.visible = visible;
    }
  }

  toggleBodyVector(type, visible) {
    if (!this.bodyVectors) return;

    for (let i = 0; i < this.batchCount; i++) {
      const vector = this.bodyVectors[i]?.get(type);
      if (vector) vector.visible = visible;
    }
  }

  getObject3D() {
    return this.group;
  }
}
