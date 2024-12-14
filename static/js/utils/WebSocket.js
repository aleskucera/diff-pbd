import { createBody } from "../objects/BodyFactory.js";
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
  console.debug("Received model config:", model);

  // Clear existing objects
  Object.values(app.bodies).forEach((obj) => app.scene.remove(obj));
  app.bodies = {};

  model.bodies.forEach((body) => {
    const object = createBody(body);
    if (object) {
      app.bodies[body.name] = object;
      app.scene.add(object);
    }
  });
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
