import * as THREE from "three";
import { BODY_CONFIG } from "../config.js";

export function createBody(bodyData) {
  const geometry = createGeometry(bodyData.shape);

  if (!geometry) return null;

  // Create visualizations only if collision points exist
  const objects = createObjects(
    geometry,
    bodyData.collision_points || [], // Pass empty array if no collision points
  );

  // Create a group to hold all visualizations
  const bodyGroup = new THREE.Group();
  bodyGroup.add(objects.mesh);
  bodyGroup.add(objects.wireframe);
  if (objects.points) bodyGroup.add(objects.points);

  // Set initial transform
  setTransform(bodyGroup, bodyData.q);

  return bodyGroup;
}

function createGeometry(shape) {
  let geometry;

  switch (shape.type) {
    case 0: // Box
      geometry = new THREE.BoxGeometry(
        shape.hx * 2,
        shape.hy * 2,
        shape.hz * 2,
        BODY_CONFIG.geometry.box.widthSegments,
        BODY_CONFIG.geometry.box.heightSegments,
        BODY_CONFIG.geometry.box.depthSegments,
      );
      break;
    case 1: // Sphere
      geometry = new THREE.SphereGeometry(
        shape.radius,
        BODY_CONFIG.geometry.sphere.widthSegments,
        BODY_CONFIG.geometry.sphere.heightSegments,
      );
      break;
    case 2: // Cylinder
      geometry = new THREE.CylinderGeometry(
        shape.radius,
        shape.radius,
        shape.height,
        BODY_CONFIG.geometry.cylinder.radialSegments,
        BODY_CONFIG.geometry.cylinder.heightSegments,
      );
      break;
    default:
      console.error("Unknown body shape type:", shape.type);
      return null;
  }

  geometry.rotateX(Math.PI / 2);
  return geometry;
}

function createObjects(geometry, pointsArray) {
  // Create wireframe
  const wireframeMaterial = new THREE.LineBasicMaterial({
    color: BODY_CONFIG.wireframe.color,
  });
  const wireframe = new THREE.LineSegments(
    new THREE.WireframeGeometry(geometry),
    wireframeMaterial,
  );
  wireframe.isWireframe = true;
  wireframe.visible = false;

  // Create environment map for mesh
  const path = BODY_CONFIG.maps.env;
  const format = ".jpg";
  const urls = [
    path + "nx" + format,
    path + "px" + format,
    path + "pz" + format,
    path + "nz" + format,
    path + "py" + format,
    path + "ny" + format,
  ];
  const reflectionCube = new THREE.CubeTextureLoader().load(urls);

  // Create mesh material
  const meshMaterial = new THREE.MeshStandardMaterial({
    color: BODY_CONFIG.mesh.color,
    roughness: BODY_CONFIG.mesh.roughness,
    metalness: BODY_CONFIG.mesh.metalness,
    envMap: reflectionCube,
    envMapIntensity: BODY_CONFIG.mesh.envMapIntensity,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: BODY_CONFIG.mesh.opacity,
  });

  // Create mesh
  const mesh = new THREE.Mesh(geometry, meshMaterial);
  mesh.isMesh = true;
  mesh.visible = true;

  // Create points
  let points = null;
  if (pointsArray.length > 0) {
    const pointsGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(pointsArray.flat());

    const sprite = new THREE.TextureLoader().load(BODY_CONFIG.points.texture);
    sprite.colorSpace = THREE.SRGBColorSpace;

    pointsGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );

    const pointsMaterial = new THREE.PointsMaterial({
      size: BODY_CONFIG.points.size,
      sizeAttenuation: true,
      opacity: BODY_CONFIG.points.opacity,
      map: sprite,
      alphaTest: 0.5,
      transparent: BODY_CONFIG.points.transparent,
    });

    points = new THREE.Points(pointsGeometry, pointsMaterial);
    points.isPoints = true;
    points.visible = false;
  }

  return { mesh: mesh, wireframe: wireframe, points: points };
}

function setTransform(group, quaternionData) {
  const position = new THREE.Vector3(
    quaternionData[0],
    quaternionData[1],
    quaternionData[2],
  );

  const rotation = new THREE.Quaternion(
    quaternionData[4],
    quaternionData[5],
    quaternionData[6],
    quaternionData[3],
  );

  group.position.set(position.x, position.y, position.z);
  group.quaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
}
