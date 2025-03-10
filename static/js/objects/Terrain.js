import * as THREE from "three";
import { TERRAIN_CONFIG, APP_CONFIG } from "../config.js";

export class Terrain {
  constructor(terrainData) {
    this.terrainData = terrainData;
    this.group = new THREE.Group();
    this.group.name = "TerrainGround";

    this.visualizations = new Map();
    this.currentColorMap = APP_CONFIG.terrainColorMap || "viridis";

    this.createVisualRepresentations();
  }

  createVisualRepresentations() {
    // Create geometry from the provided data
    const geometry = this.createGeometryFromHeightData();

    // Create surface mesh with color gradient
    const surfaceMaterial = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: TERRAIN_CONFIG.flatShading || false,
      shininess: TERRAIN_CONFIG.shininess || 10,
    });

    const surfaceMesh = new THREE.Mesh(geometry, surfaceMaterial);
    surfaceMesh.receiveShadow = true;
    surfaceMesh.castShadow = true;

    // Create wireframe
    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      opacity: 0.2,
      transparent: true,
    });

    const wireframeMesh = new THREE.Mesh(geometry, wireframeMaterial);

    // Add visualizations to map
    this.visualizations.set("surface", {
      object: surfaceMesh,
    });

    this.visualizations.set("wireframe", {
      object: wireframeMesh,
    });

    // Add meshes to group
    for (const [name, viz] of this.visualizations.entries()) {
      viz.object.visible = APP_CONFIG.terrainVisualizationModes[name] || true;
      this.group.add(viz.object);
    }

    // Prepare for normal vector visualization (optional)
    if (TERRAIN_CONFIG.showNormals) {
      this.createNormalVectors();
    }
  }

  createGeometryFromHeightData() {
    const { dimensions, bounds, heightData, normals } = this.terrainData;
    const { size_x, size_y, resolution_x, resolution_y } = dimensions;

    console.log(
      `Creating terrain geometry: ${size_x}x${size_y} m, resolution: ${resolution_x}x${resolution_y}`,
    );

    // Create a plane geometry with the right number of segments
    const geometry = new THREE.PlaneGeometry(
      size_x,
      size_y,
      resolution_x - 1,
      resolution_y - 1,
    );

    // Center the geometry based on bounds
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    geometry.translate(centerX, centerY, 0);

    // Get attributes for direct manipulation
    const position = geometry.attributes.position;
    const normalAttribute = geometry.attributes.normal;

    // Create color buffer
    const colorAttribute = new THREE.BufferAttribute(
      new Float32Array(position.count * 3),
      3,
    );

    // Apply height data to geometry
    // NOTE: THREE.js PlaneGeometry vertices are arranged in rows from bottom to top (Y increases)
    const startTime = performance.now();
    for (let i = 0; i < position.count; i++) {
      // Convert vertex index to grid coordinates
      const col = i % resolution_x;
      const invertedRow = Math.floor(i / resolution_x);
      const row = resolution_y - invertedRow - 1; // Invert row index

      // Calculate index in the flattened height data array
      const dataIndex = row * resolution_x + col;

      // Set Z coordinate (height)
      if (dataIndex < heightData.length) {
        position.setZ(i, heightData[dataIndex]);

        // Set normal if available
        if (normals && dataIndex * 3 + 2 < normals.length) {
          normalAttribute.setXYZ(
            i,
            normals[dataIndex * 3],
            normals[dataIndex * 3 + 1],
            normals[dataIndex * 3 + 2],
          );
        }

        // Calculate color based on height
        const normalizedHeight =
          (heightData[dataIndex] - bounds.minZ) /
          (bounds.maxZ - bounds.minZ || 1);

        const color = this.getColorFromSelectedMap(normalizedHeight);
        colorAttribute.setXYZ(i, color.r, color.g, color.b);
      }
    }
    const endTime = performance.now();
    console.log(`Terrain generation completed in ${endTime - startTime}ms`);

    // Add colors to the geometry
    geometry.setAttribute("color", colorAttribute);

    // Make sure changes are applied
    position.needsUpdate = true;
    normalAttribute.needsUpdate = true;

    return geometry;
  }

  getColorFromSelectedMap(value) {
    try {
      let cmapName = this.currentColorMap;
      let reversed = false;

      // Check if this is a reversed colormap request
      if (cmapName.endsWith("_r")) {
        cmapName = cmapName.substring(0, cmapName.length - 2);
        reversed = true;
      }

      // Use evaluate_cmap if available
      if (typeof evaluate_cmap === "function") {
        try {
          const [r, g, b] = evaluate_cmap(value, cmapName, reversed);
          return new THREE.Color(r / 255, g / 255, b / 255);
        } catch (e) {
          console.warn(`Error using evaluate_cmap for ${cmapName}:`, e);
        }
      }

      // Try direct function access as fallback
      const directFunction = window[this.currentColorMap];
      if (typeof directFunction === "function") {
        const [r, g, b] = directFunction(value);
        return new THREE.Color(r / 255, g / 255, b / 255);
      }
    } catch (e) {
      console.error(`Error using colormap ${this.currentColorMap}:`, e);
    }
    switch (this.currentColorMap) {
      case "grayscale":
        return new THREE.Color(value, value, value);
      case "heatmap":
        // Simple heatmap: blue->cyan->green->yellow->red
        if (value < 0.25) {
          return new THREE.Color(0, value * 4, 1);
        } else if (value < 0.5) {
          return new THREE.Color(0, 1, 1 - (value - 0.25) * 4);
        } else if (value < 0.75) {
          return new THREE.Color((value - 0.5) * 4, 1, 0);
        } else {
          return new THREE.Color(1, 1 - (value - 0.75) * 4, 0);
        }
      case "terrain":
        // Terrain color map - blues for low areas, greens for middle, browns/whites for high
        if (value < 0.2) {
          return new THREE.Color(0.0, 0.2, 0.5 + value); // Deep to shallow water
        } else if (value < 0.4) {
          const t = (value - 0.2) * 5; // 0-1 within this range
          return new THREE.Color(0.2 * t, 0.5 + 0.2 * t, 0.7 - 0.2 * t); // Shore transition
        } else if (value < 0.75) {
          const t = (value - 0.4) / 0.35; // 0-1 within this range
          return new THREE.Color(0.2 + 0.3 * t, 0.7 - 0.2 * t, 0.5 - 0.4 * t); // Green to brown
        } else {
          const t = (value - 0.75) * 4; // 0-1 within this range
          return new THREE.Color(0.5 + 0.5 * t, 0.5 + 0.5 * t, 0.1 + 0.9 * t); // Brown to white (snow)
        }
      default:
        // Default blue to red gradient
        return new THREE.Color(value, 0.2, 1 - value);
    }
  }

  createNormalVectors() {
    const { dimensions, heightData, normals } = this.terrainData;
    const { size_x, size_y, resolution_x, resolution_y } = dimensions;

    if (!normals) {
      console.warn("Cannot visualize normals - normal data not provided");
      return;
    }

    this.normalVectors = new THREE.Group();
    this.normalVectors.visible = APP_CONFIG.terrainNormalsVisible || false;

    // Create a helper arrow for each normal
    const normalLength = TERRAIN_CONFIG.normalLength || 0.5;
    const skipFactor = Math.max(1, Math.floor(resolution_x / 20)); // Adaptive skip factor based on resolution

    console.log(`Creating normal visualization with skip factor ${skipFactor}`);

    // Sample normals at regular intervals
    for (let row = 0; row < resolution_y; row += skipFactor) {
      for (let col = 0; col < resolution_x; col += skipFactor) {
        const dataIndex = row * resolution_x + col;

        if (
          dataIndex < heightData.length &&
          dataIndex * 3 + 2 < normals.length
        ) {
          // Calculate real-world coordinates
          const x =
            this.terrainData.bounds.minX + col * (size_x / (resolution_x - 1));
          const y =
            this.terrainData.bounds.minY + row * (size_y / (resolution_y - 1));
          const z = heightData[dataIndex];

          // Get normal data
          const nx = normals[dataIndex * 3];
          const ny = normals[dataIndex * 3 + 1];
          const nz = normals[dataIndex * 3 + 2];

          const origin = new THREE.Vector3(x, y, z);
          const direction = new THREE.Vector3(nx, ny, nz);

          const arrowHelper = new THREE.ArrowHelper(
            direction.normalize(),
            origin,
            normalLength,
            0xff0000,
          );

          this.normalVectors.add(arrowHelper);
        }
      }
    }

    this.group.add(this.normalVectors);
  }

  // Toggle methods for visualizations - similar to Body.js
  toggleVisualization(type, visible) {
    const visualization = this.visualizations.get(type);
    if (visualization) {
      visualization.object.visible = visible;
    }
  }

  toggleNormalVectors(visible) {
    if (this.normalVectors) {
      this.normalVectors.visible = visible;
    } else if (visible && TERRAIN_CONFIG.showNormals) {
      this.createNormalVectors();
      if (this.normalVectors) this.normalVectors.visible = visible;
    }
  }

  // Change the colormap and update the terrain
  setColorMap(colorMapName) {
    this.currentColorMap = colorMapName;
    // Recreate the terrain with the new colormap
    this.updateTerrain();
  }

  // Update terrain colors with current colormap
  updateTerrain() {
    // Get the current surface mesh geometry
    const surface = this.visualizations.get("surface")?.object;
    if (!surface) return;

    const geometry = surface.geometry;
    const position = geometry.attributes.position;
    const colorAttribute = geometry.attributes.color;

    // Update all colors based on the new colormap
    for (let i = 0; i < position.count; i++) {
      // Convert vertex index to grid coordinates
      const { dimensions, bounds, heightData } = this.terrainData;
      const { resolution_x, resolution_y } = dimensions;

      const col = i % resolution_x;
      const invertedRow = Math.floor(i / resolution_x);
      const row = resolution_y - invertedRow - 1; // Invert row index

      // Calculate index in the flattened height data array
      const dataIndex = row * resolution_x + col;

      // Update color based on height
      if (dataIndex < heightData.length) {
        const normalizedHeight =
          (heightData[dataIndex] - bounds.minZ) /
          (bounds.maxZ - bounds.minZ || 1);

        const color = this.getColorFromSelectedMap(normalizedHeight);
        colorAttribute.setXYZ(i, color.r, color.g, color.b);
      }
    }

    // Update the buffer
    colorAttribute.needsUpdate = true;
  }

  // Get terrain height at specific world coordinates (x,y)
  getHeightAt(x, y) {
    if (!this.terrainData || !this.terrainData.heightData) {
      return 0;
    }

    const { dimensions, bounds, heightData } = this.terrainData;
    const { size_x, size_y, resolution_x, resolution_y } = dimensions;

    // Convert world coordinates to grid indices
    const gridX = Math.floor(((x - bounds.minX) / size_x) * (resolution_x - 1));
    const gridY = Math.floor(((y - bounds.minY) / size_y) * (resolution_y - 1));

    // Check bounds
    if (
      gridX < 0 ||
      gridX >= resolution_x - 1 ||
      gridY < 0 ||
      gridY >= resolution_y - 1
    ) {
      return 0;
    }

    // Get the heights at the four corners of the grid cell
    const idx00 = gridY * resolution_x + gridX;
    const idx10 = idx00 + 1;
    const idx01 = idx00 + resolution_x;
    const idx11 = idx01 + 1;

    const h00 = heightData[idx00];
    const h10 = heightData[idx10];
    const h01 = heightData[idx01];
    const h11 = heightData[idx11];

    // Calculate the fractional position within the grid cell
    const fx = ((x - bounds.minX) / size_x) * (resolution_x - 1) - gridX;
    const fy = ((y - bounds.minY) / size_y) * (resolution_y - 1) - gridY;

    // Bilinear interpolation
    const h0 = h00 * (1 - fx) + h10 * fx;
    const h1 = h01 * (1 - fx) + h11 * fx;
    const height = h0 * (1 - fy) + h1 * fy;

    return height;
  }

  // Get normal at specific world coordinates (x,y)
  getNormalAt(x, y) {
    if (!this.terrainData || !this.terrainData.normals) {
      return new THREE.Vector3(0, 0, 1); // Default upward normal
    }

    const { dimensions, bounds, normals } = this.terrainData;
    const { size_x, size_y, resolution_x, resolution_y } = dimensions;

    // Convert world coordinates to grid indices
    const gridX = Math.floor(((x - bounds.minX) / size_x) * (resolution_x - 1));
    const gridY = Math.floor(((y - bounds.minY) / size_y) * (resolution_y - 1));

    // Check bounds
    if (
      gridX < 0 ||
      gridX >= resolution_x - 1 ||
      gridY < 0 ||
      gridY >= resolution_y - 1
    ) {
      return new THREE.Vector3(0, 0, 1);
    }

    // Get indices for the four corners
    const idx00 = (gridY * resolution_x + gridX) * 3;
    const idx10 = idx00 + 3;
    const idx01 = idx00 + resolution_x * 3;
    const idx11 = idx01 + 3;

    // Get normals at corners
    const n00 = new THREE.Vector3(
      normals[idx00],
      normals[idx00 + 1],
      normals[idx00 + 2],
    );
    const n10 = new THREE.Vector3(
      normals[idx10],
      normals[idx10 + 1],
      normals[idx10 + 2],
    );
    const n01 = new THREE.Vector3(
      normals[idx01],
      normals[idx01 + 1],
      normals[idx01 + 2],
    );
    const n11 = new THREE.Vector3(
      normals[idx11],
      normals[idx11 + 1],
      normals[idx11 + 2],
    );

    // Calculate the fractional position within the grid cell
    const fx = ((x - bounds.minX) / size_x) * (resolution_x - 1) - gridX;
    const fy = ((y - bounds.minY) / size_y) * (resolution_y - 1) - gridY;

    // Bilinear interpolation for normal
    const n0x = n00.x * (1 - fx) + n10.x * fx;
    const n0y = n00.y * (1 - fx) + n10.y * fx;
    const n0z = n00.z * (1 - fx) + n10.z * fx;

    const n1x = n01.x * (1 - fx) + n11.x * fx;
    const n1y = n01.y * (1 - fx) + n11.y * fx;
    const n1z = n01.z * (1 - fx) + n11.z * fx;

    const nx = n0x * (1 - fy) + n1x * fy;
    const ny = n0y * (1 - fy) + n1y * fy;
    const nz = n0z * (1 - fy) + n1z * fy;

    // Create and normalize the interpolated normal
    const normal = new THREE.Vector3(nx, ny, nz).normalize();
    return normal;
  }

  // Helper method to check if a point is in bounds
  isPointInBounds(x, y) {
    if (!this.terrainData || !this.terrainData.bounds) {
      return false;
    }

    const { bounds } = this.terrainData;
    return (
      x >= bounds.minX &&
      x <= bounds.maxX &&
      y >= bounds.minY &&
      y <= bounds.maxY
    );
  }

  // Update terrain data (e.g., if simulation modifies heights)
  updateTerrainData(newTerrainData) {
    this.terrainData = newTerrainData;

    // Remove old visualizations
    for (const viz of this.visualizations.values()) {
      this.group.remove(viz.object);
    }

    if (this.normalVectors) {
      this.group.remove(this.normalVectors);
      this.normalVectors = null;
    }

    this.visualizations.clear();

    // Create new visualizations
    this.createVisualRepresentations();
  }

  // Get THREE.js group containing all visualizations
  getObject3D() {
    return this.group;
  }
}
