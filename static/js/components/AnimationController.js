import {
  createContactPoints,
  createContactNormals,
} from "../objects/ContactFactory.js";

export class AnimationController {
  constructor(app) {
    this.app = app;
    this.states = [];
    this.isPlaying = false;
    this.playbackSpeed = 1;
    this.lastUpdateTime = 0;
    this.currentStateIndex = 0;
  }

  loadAnimation(states) {
    this.states = states;
    this.currentStateIndex = 0;

    Object.values(this.app.contactPoints).forEach((obj) =>
      this.app.scene.remove(obj),
    );
    Object.values(this.app.contactNormals).forEach((obj) =>
      this.app.scene.remove(obj),
    );
    console.log("Loaded animation states:", states.length);
  }

  play() {
    this.isPlaying = true;
    this.animate();
  }

  pause() {
    this.isPlaying = false;
  }

  stepForward() {
    if (this.currentStateIndex < this.states.length - 1) {
      this.currentStateIndex++;
      this.updateScene();
    } else {
      this.currentStateIndex = 0;
      this.updateScene();
    }
  }

  stepBackward() {
    if (this.currentStateIndex > 0) {
      this.currentStateIndex--;
      this.updateScene();
    } else {
      this.currentStateIndex = this.states.length - 1;
      this.updateScene();
    }
  }

  setSpeed(speed) {
    console.debug("Speed changed to", speed);
    this.playbackSpeed = speed;
  }

  animate() {
    if (!this.isPlaying) return;

    const currentTime = Date.now();
    const currentState = this.states[this.currentStateIndex];
    const nextState = this.states[this.currentStateIndex + 1];

    if (nextState) {
      const stateTimeDiff =
        (nextState.time - currentState.time) / this.playbackSpeed;
      const elapsedTime = (currentTime - this.lastUpdateTime) / 1000;

      if (elapsedTime >= stateTimeDiff) {
        this.currentStateIndex++;
        this.updateScene();
        this.lastUpdateTime = currentTime;
      }
    } else {
      this.currentStateIndex = 0;
    }

    requestAnimationFrame(() => this.animate());
  }

  updateScene() {
    if (!this.app.bodies || !this.states[this.currentStateIndex]) return;

    const state = this.states[this.currentStateIndex];

    state.bodies.forEach((body) => {
      const object = this.app.bodies[body.name];
      if (object) {
        object.position.set(body.q[0], body.q[1], body.q[2]);
        object.quaternion.set(body.q[4], body.q[5], body.q[6], body.q[3]);
      }
    });

    Object.values(this.app.contactPoints).forEach((obj) =>
      this.app.scene.remove(obj),
    );
    Object.values(this.app.contactNormals).forEach((obj) =>
      this.app.scene.remove(obj),
    );

    state.bodies.forEach((body) => {
      if (body.contact_points.length === 0) return;
      const contactPoints = createContactPoints(body.contact_points);
      contactPoints.visible = this.app.contactPointsVisible;
      const contactNormals = createContactNormals(body.contact_normals);
      contactNormals.visible = this.app.contactNormalsVisible;
      this.app.scene.add(contactPoints);
      this.app.contactPoints[body.name] = contactPoints;
      this.app.scene.add(contactNormals);
      this.app.contactNormals[body.name] = contactNormals;
    });

    this.app.selectionWindow.update();
  }
}
