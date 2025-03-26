import { Body } from "../objects/Body.js";
import { FlatGround } from "../objects/FlatGround.js";
import { Terrain } from "../objects/Terrain.js";
import { createPlaybackControls } from "../ui/PlaybackControls.js";
import { BatchManager } from "../components/BatchManager.js";

export function setupWebSocket(app) {
  const socket = io();

  socket.on("connect", () => {
    console.log("Connected to server");
    socket.emit("get_model");
    socket.emit("get_states");
  });

  socket.on("model", (model) => handleModel(app, model));
  socket.on("states", (states) => handleStates(app, states));
  socket.on("disconnect", () => handleDisconnect());

  // Error handling
  socket.on("error", (error) => handleError(error));

  return socket;
}

function handleModel(app, modelData) {
  try {
    console.debug("Received model:", modelData);
  } catch (error) {
    console.error("Failed to parse model:", error);
    return;
  }

  // Initialize batch manager
  app.batchManager = new BatchManager(app);
  app.batchManager.initialize(modelData);

  // Clear existing objects
  for (const body of app.bodies.values()) {
    app.scene.remove(body);
  }

  if (app.terrain) {
    app.scene.remove(app.terrain.getObject3D());
  }

  if (app.ground) {
    app.scene.remove(app.ground.getObject3D());
  }

  // Process bodies
  app.bodies = new Map();
  if (Array.isArray(modelData.bodies)) {
    modelData.bodies.forEach((bodyData) => {
      createBody(app, bodyData);
    });
  }

  // Load terrain or ground
  if (modelData.terrain) {
    console.debug("Using terrain data");
    app.terrain = new Terrain(modelData.terrain, app);
    app.scene.add(app.terrain.getObject3D());
  } else {
    console.debug("No terrain data provided, defaulting to flat ground");
    app.ground = new FlatGround();
    app.scene.add(app.ground.getObject3D());
  }

  app.bodyStateWindow.update();
}

function createBody(app, bodyData) {
  // Create body with the batch count
  const batchCount = app.batchManager ? app.batchManager.getBatchCount() : 1;
  const body = new Body(bodyData, app);

  app.bodies.set(bodyData.name, body);
  app.scene.add(body.getObject3D());
}

function handleStates(app, statesData) {
  console.debug("Received states:", statesData);
  app.animationController.loadAnimation(statesData);
  createPlaybackControls(app.animationController);
}

function handleDisconnect() {
  console.log("Disconnected from server");
}

function handleError(error) {
  console.error("WebSocket Error:", error);
}
