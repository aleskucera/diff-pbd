export class BodyStateWindow {
  constructor(app) {
    this.app = app;
    this.window = null;
    this.initWindow();
  }

  initWindow() {
    // Create window container
    this.window = document.createElement("div");
    Object.assign(this.window.style, {
      position: "fixed",
      top: "10px",
      left: "10px",
      backgroundColor: "rgba(0, 0, 0, 0.7)",
      color: "white",
      padding: "20px",
      borderRadius: "5px",
      fontFamily: "Arial, sans-serif",
      fontSize: "20px",
      zIndex: "1000",
      maxHeight: "700px",
      overflowY: "auto",
      minWidth: "200px",
    });

    // Create header
    const header = document.createElement("div");
    Object.assign(header.style, {
      borderBottom: "1px solid white",
      paddingBottom: "5px",
      marginBottom: "5px",
      fontWeight: "bold",
      fontSize: "24px",
    });
    header.textContent = "Body State";
    this.window.appendChild(header);

    // Create content container
    this.content = document.createElement("div");
    this.window.appendChild(this.content);

    document.body.appendChild(this.window);
  }

  update() {
    // Clear previous content
    this.content.innerHTML = "";

    if (!this.app.bodies || this.app.bodies.size === 0) {
      const noSelection = document.createElement("div");
      noSelection.textContent = "No bodies in the scene";
      noSelection.style.fontStyle = "italic";
      noSelection.style.color = "#999";
      this.content.appendChild(noSelection);
      return;
    }

    // Create list of selected objects
    const list = document.createElement("ul");
    Object.assign(list.style, {
      margin: "0",
      padding: "0 0 0 20px",
    });

    this.app.bodies.forEach((body, name) => {
      const item = document.createElement("li");
      Object.assign(item.style, {
        marginBottom: "5px",
      });

      // Create object info
      const info = this.createObjectInfo(body);
      item.appendChild(info);

      list.appendChild(item);
    });

    this.content.appendChild(list);
  }

  createObjectInfo(obj) {
    const container = document.createElement("div");

    // Object name/type
    const name = document.createElement("div");
    name.textContent = obj.name || "Unnamed Object";
    name.style.fontWeight = "bold";
    container.appendChild(name);

    // Object properties
    const props = document.createElement("div");
    props.style.fontSize = "14px";
    props.style.color = "#ccc";

    // Position
    const pos = obj.position;
    props.innerHTML += `Position: (${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})<br>`;

    // Rotation
    const rot = obj.rotation;
    props.innerHTML += `Rotation: (${rot.x.toFixed(2)}, ${rot.y.toFixed(2)}, ${rot.z.toFixed(2)})<br>`;

    // Linear velocity
    const linearVelocity = obj.linearVelocity;
    props.innerHTML += `Linear Velocity: (${linearVelocity.x.toFixed(2)}, ${linearVelocity.y.toFixed(2)}, ${linearVelocity.z.toFixed(2)})<br>`;

    // Angular velocity
    const angularVelocity = obj.angularVelocity;
    props.innerHTML += `Angular Velocity: (${angularVelocity.x.toFixed(2)}, ${angularVelocity.y.toFixed(2)}, ${angularVelocity.z.toFixed(2)})<br>`;

    container.appendChild(props);
    return container;
  }

  show() {
    this.window.style.display = "block";
  }

  hide() {
    this.window.style.display = "none";
  }

  dispose() {
    if (this.window && this.window.parentNode) {
      this.window.parentNode.removeChild(this.window);
    }
  }
}
