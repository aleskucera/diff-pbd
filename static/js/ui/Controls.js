import { GUI } from "three/addons/libs/lil-gui.module.min.js";

export class UIControls {
  constructor(app) {
    this.app = app;
    this.gui = this.createDatGUI();
    this.setupKeyboardControls();
  }

  createDatGUI() {
    this.gui = new GUI();

    const controls = {
      bodyVisualizationMode: this.app.bodyVisualizationMode,
      showContactPoints: this.app.contactPointsVisible,
      showContactNormals: this.app.contactNormalsVisible,
      showAxes: this.app.axesVisible,
      showLinearVelocity: this.app.bodyVectorVisible.linearVelocity,
      showAngularVelocity: this.app.bodyVectorVisible.angularVelocity,
      showLinearForce: this.app.bodyVectorVisible.linearForce,
      showTorque: this.app.bodyVectorVisible.torque,
    };

    this.displayFolder = this.gui.addFolder("Display Options");

    this.displayFolder
      .add(controls, "bodyVisualizationMode", ["mesh", "wireframe", "points"])
      .name("Body Visualization Mode (B)")
      .onChange((value) => {
        this.updateVisualizationMode(value);
      });

    this.displayFolder
      .add(controls, "showAxes")
      .name("Show Axes (A)")
      .onChange((value) => {
        this.updateAxesVisibility(value);
      });

    this.displayFolder
      .add(controls, "showContactPoints")
      .name("Show Contact Points (C)")
      .onChange((value) => {
        this.updateContactPointsVisibility(value);
      });

    // Combined vector controls
    const vectorControls = [
      {
        property: "showLinearVelocity",
        name: "Show Linear Velocity (V)",
        type: "linearVelocity",
      },
      {
        property: "showAngularVelocity",
        name: "Show Angular Velocity (W)",
        type: "angularVelocity",
      },
      {
        property: "showLinearForce",
        name: "Show Linear Force (F)",
        type: "linearForce",
      },
      { property: "showTorque", name: "Show Torque (T)", type: "torque" },
    ];

    vectorControls.forEach((control) => {
      this.displayFolder
        .add(controls, control.property)
        .name(control.name)
        .onChange((value) => {
          this.updateVectorVisibility(control.type, value);
        });
    });

    this.displayFolder.open();
    return this.gui;
  }

  setupKeyboardControls() {
    window.addEventListener("keydown", (event) => {
      switch (event.key.toLowerCase()) {
        case "b":
          const modes = ["mesh", "wireframe", "points"];
          const currentIndex = modes.indexOf(this.app.bodyVisualizationMode);
          const nextIndex = (currentIndex + 1) % modes.length;
          this.updateVisualizationMode(modes[nextIndex]);
          const controller = this.findController("bodyVisualizationMode");
          if (controller) controller.setValue(modes[nextIndex]);
          break;
        case "a":
          this.toggleControl("showAxes");
          break;
        case "c":
          this.toggleControl("showContactPoints");
          break;
        case "v":
          this.toggleControl("showLinearVelocity");
          break;
        case "w":
          this.toggleControl("showAngularVelocity");
          break;
        case "f":
          this.toggleControl("showLinearForce");
          break;
        case "t":
          this.toggleControl("showTorque");
          break;
      }
    });
  }

  findController(property) {
    for (const controller of this.displayFolder.controllers) {
      if (controller.property === property) {
        return controller;
      }
    }
    return null;
  }

  toggleControl(property) {
    const controller = this.findController(property);
    if (controller) {
      controller.setValue(!controller.getValue());
    }
  }

  updateVisualizationMode(mode) {
    this.app.bodies.forEach((body) => {
      body.updateVisualizationMode(mode);
    });
    this.app.bodyVisualizationMode = mode;
  }

  updateAxesVisibility(show) {
    this.app.bodies.forEach((body) => {
      body.toggleAxes(show);
    });
    this.app.axesVisible = show;
  }

  updateContactPointsVisibility(show) {
    this.app.bodies.forEach((body) => {
      body.toggleContactPoints(show);
    });
    this.app.contactPointsVisible = show;
  }

  // Combined vector visibility update function
  updateVectorVisibility(vectorType, show) {
    this.app.bodies.forEach((body) => {
      body.toggleBodyVector(vectorType, show);
    });
    this.app.bodyVectorVisible[vectorType] = show;
  }
}
