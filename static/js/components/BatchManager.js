import * as THREE from "three";

export class BatchManager {
  constructor(app) {
    this.app = app;
    this.batchCount = 1; // Default to single batch

    // Batch offset configuration
    this.batchOffsets = []; // Array of {x, y, z} offsets for each batch
    this.defaultOffsetX = 45; // Default X offset between batches
  }

  initialize(modelData) {
    if (!modelData) return;

    // Set batch count from model data
    if (modelData.simBatches !== undefined) {
      this.batchCount = Math.max(1, parseInt(modelData.simBatches));
      console.log(`Initializing with ${this.batchCount} simulation batches`);
      this.app.batchCount = this.batchCount;
    }

    // Initialize batch offsets
    this.batchOffsets = [];
    for (let i = 0; i < this.batchCount; i++) {
      this.batchOffsets.push({
        x: i * this.defaultOffsetX, // First batch (i=0) has no offset
        y: 0,
        z: 0,
      });
    }
  }

  getBatchCount() {
    return this.batchCount;
  }

  // Get offset for a specific batch
  getBatchOffset(batchIndex) {
    if (batchIndex >= 0 && batchIndex < this.batchOffsets.length) {
      return this.batchOffsets[batchIndex];
    }
    return { x: 0, y: 0, z: 0 };
  }

  // Set offset for a specific batch
  setBatchOffset(batchIndex, offset) {
    if (batchIndex >= 0 && batchIndex < this.batchOffsets.length) {
      this.batchOffsets[batchIndex] = { ...offset };

      // Update all bodies to apply the new offset
      this.app.bodies.forEach((body) => {
        if (body.applyBatchOffset) {
          body.applyBatchOffset(batchIndex);
        }
      });

      // Update terrain if needed
      if (this.app.terrain && this.app.terrain.updateBatchOffsets) {
        this.app.terrain.updateBatchOffsets();
      }
    }
  }
}
