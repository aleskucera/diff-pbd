import * as THREE from "three";

export class BodyStateWindow {
  constructor(app) {
    this.app = app;
    this.selectedBody = null;
    this.selectedBatch = 0; // Default to first batch
    this.window = null;
    this.batchSelector = null;
    this.content = null;
    this.scalarContent = null;

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
      fontSize: "16px",
      zIndex: "1000",
      maxHeight: "700px",
      overflowY: "auto",
      minWidth: "300px",
    });

    // Create header with title and batch selector
    const header = document.createElement("div");
    Object.assign(header.style, {
      borderBottom: "1px solid white",
      paddingBottom: "10px",
      marginBottom: "10px",
      fontWeight: "bold",
      fontSize: "20px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    });

    const title = document.createElement("span");
    title.textContent = "Body State";
    header.appendChild(title);

    // Add batch selector if we have multiple batches
    if (this.app.batchManager && this.app.batchManager.getBatchCount) {
      const batchCount = this.app.batchManager.getBatchCount();

      if (batchCount > 1) {
        const selectorContainer = document.createElement("div");
        selectorContainer.style.display = "flex";
        selectorContainer.style.alignItems = "center";

        const label = document.createElement("span");
        label.textContent = "Batch: ";
        label.style.marginRight = "5px";
        selectorContainer.appendChild(label);

        const batchSelector = document.createElement("select");
        batchSelector.style.backgroundColor = "rgba(50, 50, 50, 0.8)";
        batchSelector.style.color = "white";
        batchSelector.style.border = "1px solid white";
        batchSelector.style.padding = "3px 5px";
        batchSelector.style.borderRadius = "3px";

        for (let i = 0; i < batchCount; i++) {
          const option = document.createElement("option");
          option.value = i;
          option.textContent = `${i + 1}`;
          batchSelector.appendChild(option);
        }

        batchSelector.addEventListener("change", (e) => {
          this.selectedBatch = parseInt(e.target.value);
          this.update();
        });

        selectorContainer.appendChild(batchSelector);
        header.appendChild(selectorContainer);
        this.batchSelector = batchSelector;
      }
    }

    this.window.appendChild(header);

    // Create body content container
    this.content = document.createElement("div");
    this.window.appendChild(this.content);

    // Create scalar values container if batch manager has scalar values
    if (
      this.app.batchManager &&
      this.app.batchManager.scalarNames &&
      this.app.batchManager.scalarNames.length > 0
    ) {
      this.scalarContent = document.createElement("div");
      this.scalarContent.style.marginTop = "15px";
      this.scalarContent.style.borderTop = "1px solid rgba(255, 255, 255, 0.3)";
      this.scalarContent.style.paddingTop = "10px";

      const scalarHeader = document.createElement("div");
      scalarHeader.textContent = "Scalar Values";
      scalarHeader.style.fontWeight = "bold";
      scalarHeader.style.marginBottom = "8px";

      this.scalarContent.appendChild(scalarHeader);
      this.window.appendChild(this.scalarContent);
    }

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

    // Create body list section
    const bodiesTitle = document.createElement("div");
    bodiesTitle.textContent = "Bodies:";
    bodiesTitle.style.fontWeight = "bold";
    bodiesTitle.style.marginBottom = "8px";
    this.content.appendChild(bodiesTitle);

    // Create clickable body list
    const list = document.createElement("ul");
    Object.assign(list.style, {
      margin: "0 0 15px 0",
      padding: "0 0 0 20px",
    });

    this.app.bodies.forEach((body, name) => {
      const item = document.createElement("li");
      Object.assign(item.style, {
        marginBottom: "5px",
        cursor: "pointer",
        padding: "3px 5px",
        borderRadius: "3px",
        transition: "background-color 0.2s",
      });

      // Highlight selected body
      if (this.selectedBody === name) {
        item.style.backgroundColor = "rgba(76, 175, 80, 0.3)";
        item.style.fontWeight = "bold";
      } else {
        item.style.backgroundColor = "transparent";
      }

      item.textContent = name;

      // Hover effect
      item.addEventListener("mouseover", () => {
        if (this.selectedBody !== name) {
          item.style.backgroundColor = "rgba(255, 255, 255, 0.1)";
        }
      });

      item.addEventListener("mouseout", () => {
        if (this.selectedBody !== name) {
          item.style.backgroundColor = "transparent";
        }
      });

      // Selection with toggle behavior
      item.addEventListener("click", () => {
        // Toggle selection - clicking the selected body will unselect it
        if (this.selectedBody === name) {
          this.selectedBody = null;
        } else {
          this.selectedBody = name;
        }
        this.update();
      });

      list.appendChild(item);
    });

    this.content.appendChild(list);

    // Show selected body details
    if (this.selectedBody) {
      const body = this.app.bodies.get(this.selectedBody);
      if (body) {
        const details = this.createBodyDetails(body);
        this.content.appendChild(details);
      }
    }

    // Update scalar values
    this.updateScalarValues();
  }

  createBodyDetails(body) {
    const container = document.createElement("div");
    container.style.marginTop = "10px";
    container.style.padding = "12px";
    container.style.backgroundColor = "rgba(50, 50, 50, 0.5)";
    container.style.borderRadius = "5px";
    container.style.position = "relative"; // For positioning the close button

    // Add close button
    const closeButton = document.createElement("div");
    closeButton.innerHTML = "✕"; // Unicode X symbol
    Object.assign(closeButton.style, {
      position: "absolute",
      top: "8px",
      right: "10px",
      cursor: "pointer",
      fontSize: "14px",
      color: "#aaa",
      fontWeight: "bold",
      width: "20px",
      height: "20px",
      textAlign: "center",
      lineHeight: "20px",
      borderRadius: "50%",
    });

    // Close button hover effect
    closeButton.addEventListener("mouseover", () => {
      closeButton.style.backgroundColor = "rgba(255, 255, 255, 0.2)";
      closeButton.style.color = "white";
    });

    closeButton.addEventListener("mouseout", () => {
      closeButton.style.backgroundColor = "transparent";
      closeButton.style.color = "#aaa";
    });

    // Close button click handler
    closeButton.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent propagation to container
      this.selectedBody = null;
      this.update();
    });

    container.appendChild(closeButton);

    // Get batch-specific data based on selected batch
    const batchIndex = this.selectedBatch;

    // Handle both multi-batch and single-batch formats
    let position,
      rotation,
      linearVelocity,
      angularVelocity,
      linearForce,
      torque;

    // Check if body has batch-specific properties
    if (body.positions && body.positions[batchIndex]) {
      position = body.positions[batchIndex];
      rotation = body.rotations[batchIndex];
      linearVelocity = body.linearVelocities[batchIndex];
      angularVelocity = body.angularVelocities[batchIndex];
      linearForce = body.linearForces[batchIndex];
      torque = body.torques[batchIndex];
    } else {
      // Fallback to single-batch properties
      position = body.position;
      rotation = body.rotation;
      linearVelocity = body.linearVelocity;
      angularVelocity = body.angularVelocity;
      linearForce = body.linearForce;
      torque = body.torque;
    }

    // Ensure we have valid objects
    position = position || new THREE.Vector3();
    rotation = rotation || new THREE.Euler();
    linearVelocity = linearVelocity || new THREE.Vector3();
    angularVelocity = angularVelocity || new THREE.Vector3();
    linearForce = linearForce || new THREE.Vector3();
    torque = torque || new THREE.Vector3();

    // Body details header
    const detailsHeader = document.createElement("div");
    detailsHeader.textContent = `${body.name} Details${this.app.batchManager && this.app.batchManager.getBatchCount() > 1 ? ` (Batch ${batchIndex + 1})` : ""}`;
    detailsHeader.style.fontWeight = "bold";
    detailsHeader.style.marginBottom = "10px";
    container.appendChild(detailsHeader);

    // Create table for properties
    const table = document.createElement("table");
    table.style.width = "100%";
    table.style.fontSize = "14px";
    table.style.borderCollapse = "collapse";

    // Helper function to add a row with vector data
    const addVectorRow = (label, vector, precision = 3) => {
      const row = table.insertRow();

      const labelCell = row.insertCell(0);
      labelCell.textContent = label;
      labelCell.style.padding = "4px 10px 4px 0";
      labelCell.style.fontWeight = "bold";
      labelCell.style.color = "#ddd";

      const valueCell = row.insertCell(1);
      valueCell.textContent = `(${vector.x.toFixed(precision)}, ${vector.y.toFixed(precision)}, ${vector.z.toFixed(precision)})`;
      valueCell.style.padding = "4px 0";
      valueCell.style.fontFamily = "monospace";
    };

    // Add all properties
    addVectorRow("Position", position);
    addVectorRow("Rotation", rotation);
    addVectorRow("Linear Velocity", linearVelocity);
    addVectorRow("Angular Velocity", angularVelocity);
    addVectorRow("Force", linearForce);
    addVectorRow("Torque", torque);

    // Note: Contact points information is now removed as requested

    container.appendChild(table);
    return container;
  }

  updateScalarValues() {
    // Skip if we don't have scalar content or batch manager
    if (
      !this.scalarContent ||
      !this.app.batchManager ||
      !this.app.batchManager.scalarNames ||
      this.app.batchManager.scalarNames.length === 0
    ) {
      return;
    }

    // Clear previous scalar values, keeping the header
    while (this.scalarContent.children.length > 1) {
      this.scalarContent.removeChild(this.scalarContent.lastChild);
    }

    // Get batch-specific scalar values
    const batchIndex = this.selectedBatch;

    // Create table for scalar values
    const table = document.createElement("table");
    table.style.width = "100%";
    table.style.fontSize = "14px";
    table.style.borderCollapse = "collapse";

    // Add each scalar value as a row
    this.app.batchManager.scalarNames.forEach((name) => {
      const value = this.app.batchManager.getScalarValue(name, batchIndex);

      const row = table.insertRow();

      const labelCell = row.insertCell(0);
      labelCell.textContent = name;
      labelCell.style.padding = "4px 10px 4px 0";
      labelCell.style.fontWeight = "bold";
      labelCell.style.color = "#ddd";

      const valueCell = row.insertCell(1);
      valueCell.textContent = value.toFixed(3);
      valueCell.style.padding = "4px 0";
      valueCell.style.fontFamily = "monospace";
    });

    this.scalarContent.appendChild(table);
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
