export class SelectionWindow {
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
      padding: "10px",
      borderRadius: "5px",
      fontFamily: "Arial, sans-serif",
      fontSize: "14px",
      zIndex: "1000",
      maxHeight: "300px",
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
    });
    header.textContent = "Selected Objects";
    this.window.appendChild(header);

    // Create content container
    this.content = document.createElement("div");
    this.window.appendChild(this.content);

    document.body.appendChild(this.window);
  }

  update() {
    // Clear previous content
    this.content.innerHTML = "";

    if (!this.app.selectedBodies || this.app.selectedBodies.size === 0) {
      const noSelection = document.createElement("div");
      noSelection.textContent = "No objects selected";
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

    this.app.selectedBodies.forEach((obj) => {
      const item = document.createElement("li");
      Object.assign(item.style, {
        marginBottom: "5px",
      });

      // Create object info
      const info = this.createObjectInfo(obj);
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
    props.style.fontSize = "12px";
    props.style.color = "#ccc";

    // Position
    const pos = obj.position;
    props.innerHTML += `Position: (${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})<br>`;

    // Scale
    const scale = obj.scale;
    props.innerHTML += `Scale: (${scale.x.toFixed(2)}, ${scale.y.toFixed(2)}, ${scale.z.toFixed(2)})<br>`;

    // Type
    props.innerHTML += `Type: ${obj.type}<br>`;

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
