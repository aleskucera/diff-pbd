import { createBody } from "../objects/BodyFactory.js";
import { Body } from "../objects/Body.js";
import { createPlaybackControls } from "../ui/PlaybackControls.js";

export function setupWebSocket(app) {
  const socket = io();

  socket.on("connect", () => {
    console.log("Connected to server");
    socket.emit("get_model");
    socket.emit("get_states");
  });

  socket.on("model", (model) => handleModel(model, app));
  socket.on("states", (states) => handleStates(states, app));
  socket.on("disconnect", () => handleDisconnect());

  // Error handling
  socket.on("error", (error) => handleError(error));

  return socket;
}

function handleModel(model, app) {
  try {
    console.debug("Received model:", model);
  } catch (error) {
    console.error("Failed to parse model:", error);
    return;
  }

  // Clear existing objects
  for (const body of app.bodies.values()) {
    app.scene.remove(body);
  }

  app.bodies = new Map();

  if (model.robot) {
    const body = new Body(model.robot, app.bodyVisualizationMode);
    app.bodies.set(model.robot.name, body);
    app.scene.add(body.getObject3D());
  }

  model.bodies.forEach((bodyData) => {
    const body = new Body(bodyData, app.bodyVisualizationMode);
    if (body) {
      app.bodies.set(bodyData.name, body);
      app.scene.add(body.getObject3D());
    }
  });
  app.bodyStateWindow.update();
}

function handleStates(states, app) {
  try {
    console.debug("Received states:", states);
  } catch (error) {
    console.error("Failed to parse states:", error);
    return;
  }
  app.animationController.loadAnimation(states);
  createPlaybackControls(app.animationController);
}

function handleDisconnect() {
  console.log("Disconnected from server");
}

function handleError(error) {
  console.error("WebSocket Error:", error);
}

// Helper function to clean up WebSocket connection
export function cleanupWebSocket(socket) {
  if (socket) {
    socket.disconnect();
  }
}
