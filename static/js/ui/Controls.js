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

    this.bodyFolder = this.gui.addFolder("Body Options");

    this.bodyFolder
      .add(controls, "bodyVisualizationMode", ["mesh", "wireframe", "points"])
      .name("Body Visualization Mode (B)")
      .onChange((value) => {
        this.updateVisualizationMode(value);
      });

    this.bodyFolder
      .add(controls, "showAxes")
      .name("Show Axes (A)")
      .onChange((value) => {
        this.updateAxesVisibility(value);
      });

    this.bodyFolder
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
      this.bodyFolder
        .add(controls, control.property)
        .name(control.name)
        .onChange((value) => {
          this.updateVectorVisibility(control.type, value);
        });
    });

    this.bodyFolder.open();

    // Add terrain controls
    this.terrainFolder = this.gui.addFolder("Terrain Options");

    // Create terrain controls object
    const terrainControls = {
      showSurface: this.app.terrainVisualizationModes?.surface ?? true,
      showWireframe: this.app.terrainVisualizationModes?.wireframe ?? true,
      showNormals: this.app.terrainNormalsVisible ?? false,
      colorMap: this.app.terrain?.currentColorMap || "viridis",
    };

    // Add terrain visualization toggles
    this.terrainFolder
      .add(terrainControls, "showSurface")
      .name("Show Surface")
      .onChange((value) => {
        this.updateTerrainVisualization("surface", value);
      });

    this.terrainFolder
      .add(terrainControls, "showWireframe")
      .name("Show Wireframe")
      .onChange((value) => {
        this.updateTerrainVisualization("wireframe", value);
      });

    this.terrainFolder
      .add(terrainControls, "showNormals")
      .name("Show Normals")
      .onChange((value) => {
        this.updateTerrainNormals(value);
      });

    // Add colormap dropdown
    // Select a reasonable subset of the available colormaps
    const colorMapOptions = [
      "viridis",
      "viridis_r",
      "plasma",
      "plasma_r",
      "inferno",
      "inferno_r",
      "magma",
      "magma_r",
      "jet",
      "jet_r",
      "rainbow",
      "rainbow_r",
      "terrain",
      "terrain_r",
      "coolwarm",
      "coolwarm_r",
      "Spectral",
      "Spectral_r",
      "YlGnBu",
      "YlGnBu_r",
      "RdYlBu",
      "RdYlBu_r",
      "Greys",
      "Greys_r",
    ];

    this.terrainFolder
      .add(terrainControls, "colorMap", colorMapOptions)
      .name("Color Map")
      .onChange((value) => {
        this.updateTerrainColorMap(value);
      });

    this.terrainFolder.open();
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
    for (const controller of this.bodyFolder.controllers) {
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

  updateTerrainVisualization(type, visible) {
    if (this.app.terrain) {
      this.app.terrain.toggleVisualization(type, visible);

      // Update app state
      if (!this.app.terrainVisualizationModes) {
        this.app.terrainVisualizationModes = {};
      }
      this.app.terrainVisualizationModes[type] = visible;
    }
  }

  updateTerrainNormals(show) {
    if (this.app.terrain) {
      this.app.terrain.toggleNormalVectors(show);
      this.app.terrainNormalsVisible = show;
    }
  }

  updateTerrainColorMap(colorMap) {
    if (this.app.terrain) {
      this.app.terrain.setColorMap(colorMap);
    }
  }
}
