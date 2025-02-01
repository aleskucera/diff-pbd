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
  constructor(bodyData) {
    this.name = bodyData.name;
    this.visualizations = new Map();
    this.activeContacts = new Set();

    this.position = new THREE.Vector3();
    this.quaternion = new THREE.Quaternion();
    this.rotation = new THREE.Euler();
    this.linearVelocity = new THREE.Vector3();
    this.angularVelocity = new THREE.Vector3();
    this.linearForce = new THREE.Vector3();
    this.torque = new THREE.Vector3();

    this.initializeGroup();
    this.createVisualRepresentations(bodyData);
    this.initializeContactPoints(bodyData.collision_points);
    this.initializeBodyVectors(BODY_VECTOR_CONFIG);
    this.initializeAxes();

    this.updateState(bodyData);
  }

  initializeGroup() {
    this.group = new THREE.Group();
    this.group.name = this.name;
  }

  initializeAxes() {
    const axes = new THREE.AxesHelper(1);
    axes.visible = APP_CONFIG.axesVisible;
    this.group.add(axes);
    this.axes = axes;
  }

  createVisualRepresentations(bodyData) {
    this.geometry = createGeometry(bodyData.shape, BODY_CONFIG.geometry);

    const representations = {
      mesh: {
        object: createMesh(this.geometry, BODY_CONFIG.mesh),
      },
      wireframe: {
        object: createWireframe(this.geometry, BODY_CONFIG.wireframe),
      },
      points: {
        object: createPoints(bodyData.collision_points, BODY_CONFIG.points),
      },
    };

    for (const [name, repr] of Object.entries(representations)) {
      if (repr.object) {
        repr.object.visible = repr.defaultVisible;
        this.group.add(repr.object);
        this.visualizations.set(name, repr.object);
      }
    }

    this.updateVisualizationMode(APP_CONFIG.bodyVisualizationMode);
  }

  initializeContactPoints(collisionPoints) {
    if (!collisionPoints?.length) return;

    this.contactPoints = createContactPoints(
      collisionPoints,
      BODY_CONFIG.contactPoints,
    );
    if (!this.contactPoints) return;

    const pointCount = collisionPoints.length;
    this.contactPointSizes = new Float32Array(pointCount);
    this.contactPointSizes.fill(0);

    this.contactPoints.geometry.setAttribute(
      "size",
      new THREE.Float32BufferAttribute(this.contactPointSizes, 1),
    );

    this.group.add(this.contactPoints);

    this.contactPoints.visible = APP_CONFIG.contactPointsVisible;
  }

  initializeBodyVectors(vectorConfigs) {
    this.bodyVectors = new Map();

    for (const [name, config] of Object.entries(vectorConfigs)) {
      const vector = createArrow(
        new THREE.Vector3(),
        new THREE.Vector3(0, 1, 0),
        config,
      );
      vector.visible = APP_CONFIG.bodyVectorVisible[name];
      vector.userData = { scale: config.scale };

      this.group.add(vector);
      this.bodyVectors.set(name, vector);
    }
  }

  updateBodyVector(type, vector) {
    const arrow = this.bodyVectors.get(type);
    if (!arrow) return;

    const scale = arrow.userData.scale;
    const length = vector.length() * scale;

    let normalizedVector;
    if (type === "linearVelocity" || type === "linearForce") {
      const rotatedVector = vector
        .clone()
        .applyQuaternion(this.quaternion.clone().invert());
      normalizedVector = rotatedVector.clone().normalize();
    } else {
      normalizedVector = vector.clone().normalize();
    }

    arrow.setDirection(normalizedVector);
    arrow.setLength(length, length * 0.2, length * 0.1);
  }

  updateState(bodyState) {
    // Update position and rotation
    const [x, y, z, qw, qx, qy, qz] = bodyState.q;
    this.position = new THREE.Vector3(x, y, z);
    this.quaternion = new THREE.Quaternion(qx, qy, qz, qw);
    this.rotation = new THREE.Euler().setFromQuaternion(this.quaternion);

    this.setTransform(bodyState.q);

    // Update linear and angular velocity
    const [wx, wy, wz, vx, vy, vz] = bodyState.qd;
    this.linearVelocity = new THREE.Vector3(vx, vy, vz);
    this.angularVelocity = new THREE.Vector3(wx, wy, wz);

    this.updateBodyVector("linearVelocity", this.linearVelocity);
    this.updateBodyVector("angularVelocity", this.angularVelocity);

    // Update the linear force and torque
    const [tx, ty, tz, fx, fy, fz] = bodyState.f;
    this.linearForce = new THREE.Vector3(fx, fy, fz);
    this.torque = new THREE.Vector3(tx, ty, tz);

    this.updateBodyVector("linearForce", this.linearForce);
    this.updateBodyVector("torque", this.torque);

    // Update contact points
    if (bodyState.contacts) {
      this.updateContactPointsVisibility(bodyState.contacts);
    }
  }

  updateContactPointsVisibility(contactIndices) {
    if (!this.contactPoints || !this.contactPointSizes) return;

    // Reset all sizes to 0
    this.contactPointSizes.fill(0);

    // Set size for active contacts
    contactIndices.forEach((index) => {
      if (index < this.contactPointSizes.length) {
        this.contactPointSizes[index] = BODY_CONFIG.contactPoints.size;
      }
    });

    // Update the buffer attribute
    const sizeAttribute = this.contactPoints.geometry.getAttribute("size");
    sizeAttribute.array = this.contactPointSizes;
    sizeAttribute.needsUpdate = true;
  }

  /**
   * Updates the transform of the body
   * @param {Array<number>} transform - [x, y, z, qw, qx, qy, qz]
   */
  setTransform(transform) {
    const [x, y, z, qw, qx, qy, qz] = transform;
    this.group.position.set(x, y, z);
    this.group.quaternion.set(qx, qy, qz, qw);
  }

  updateVisualizationMode(mode) {
    if (!this.visualizations || this.visualizations.size === 0) return;
    if (!this.visualizations.has(mode)) {
      throw new Error(`Invalid visualization mode: ${mode}`);
    }

    for (const [type, object] of this.visualizations) {
      object.visible = type === mode;
    }
  }

  getObject3D() {
    return this.group;
  }

  toggleContactPoints(visible) {
    if (!this.contactPoints) return;
    this.contactPoints.visible = visible;
  }

  toggleAxes(visible) {
    this.axes.visible = visible;
  }

  toggleBodyVector(type, visible) {
    const vector = this.bodyVectors.get(type);
    if (vector) vector.visible = visible;
  }

  togglePointVectors(type, visible) {
    const vectorGroup = this.pointVectors.get(type);
    if (vectorGroup) vectorGroup.visible = visible;
  }
}
