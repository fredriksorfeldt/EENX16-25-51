
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import math
import trimesh
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

from OCC.Core.TopoDS import TopoDS_Shape

## TODO: 
#  - Implement JSON export.

class ZMap:
    def __init__(self, data: NDArray[np.float64], size: float, sampling_rate: int):
        self.data: NDArray[np.float64] = data
        self.size: float = size
        self.sampling_rate: int = sampling_rate

    def extract_patch(self, radius: float) -> NDArray[np.float64]:
        if self.data is None:
            raise RuntimeError("Z-map data not initialized")

        if radius > (self.size / 2):
            raise ValueError("Radius exceeds half the z-map size")
        
        step_size = self.size / self.sampling_rate
        center = (self.sampling_rate - 1) / 2

        y, x = np.ogrid[:self.sampling_rate, :self.sampling_rate]
        dist = np.sqrt((x - center)**2 + (y - center)**2) * step_size
        mask = dist <= radius
        mask_flat = mask.ravel()

        return self.data[mask_flat]
    
    def is_flat(self, threshold: float, radius: float) -> np.bool_:
        patch = self.extract_patch(radius)
        return np.all(np.abs(patch) < threshold)

class PointSample:
    def __init__(self, position: NDArray[np.float64], normal: NDArray[np.float64]):
        self.position = position
        self.normal = normal
        self.z_map: ZMap | None = None
    
    @property
    def has_z_map(self):
        return self.z_map is not None
    
    def radial_distance_to_cog(self, cog: NDArray[np.float64]):
        v = cog - self.position
        v_proj = v - np.dot(v, self.normal) * self.normal
        distance = np.linalg.norm(v_proj)
        return distance
    
    def get_dict(self, cog: NDArray[np.float64], mass: float) -> Dict[str, Any]:
        from .geometry_utils import normal_to_quaternion
        quat = normal_to_quaternion(self.normal)
        return {
            "position": {
                "x": self.position[0], 
                "y": self.position[1], 
                "z": self.position[2]
            },
            "quaternion": {
                "x": float(quat[0]), 
                "y": float(quat[1]), 
                "z": float(quat[2]),
                "w": float(quat[3])
            },
            "max_torque": self.radial_distance_to_cog(cog) * mass
        }

class PointSet:
    def __init__(self, points: NDArray[np.float64], normals: NDArray[np.float64], cog: NDArray[np.float64], mass: float):
        self.samples = [PointSample(p, n) for p, n in zip(points, normals)]
        self.cog = cog
        self.mass = mass
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> PointSample:
        return self.samples[idx]
    
    @property
    def has_z_map(self) -> bool:
        return bool(self.samples) and all(s.has_z_map for s in self.samples)
    
    @property
    def positions(self) -> NDArray[np.float64]:
        return np.stack([s.position for s in self.samples])
    
    @property
    def normals(self) -> NDArray[np.float64]:
        return np.stack([s.normal for s in self.samples])
    
    @property
    def z_maps(self) -> NDArray[np.float64]:
        return np.stack([s.z_map.data for s in self.samples if s.z_map is not None])

    def assign_z_maps(self, z_maps: NDArray[np.float64], size: float, sampling_rate: int):
        for sample, z_map_data in zip(self.samples, z_maps):
            sample.z_map = ZMap(z_map_data, size, sampling_rate)

    def filter_by_flatness(self, threshold: float, radius: float):
        self.samples = [s for s in self.samples if s.z_map.is_flat(threshold, radius)]

    def remove_excessive_torque_samples(self, torque_threshold: float):
        self.samples = [s for s in self.samples if (s.radial_distance_to_cog(self.cog) * self.mass) < torque_threshold]

    def cluster_samples(self, degree_eps: float, pos_eps: float, min_samples: int = 1) -> List[List[int]]:
        if len(self.samples) == 0:
            raise ValueError("Cannot cluster: no samples available.")
        if len(self.samples) == 1:
            return [[0]]
        
        # Normal clustering
        normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)

        # Angular distance matrix
        cos_sim = np.clip(normals @ normals.T, -1.0, 1.0)
        angular_dist = np.arccos(cos_sim)   # In radians

        angle_radians = np.radians(degree_eps)

        normal_db = DBSCAN(eps=angle_radians, min_samples=1, metric='precomputed')
        normal_labels = normal_db.fit_predict(angular_dist)

        final_clusters: List[List[int]] = []
        
        # Position clustering for each normal cluster
        unique_labels = set(normal_labels)
        unique_labels.discard(-1)

        for normal_label in unique_labels:
            indices = [i for i, lbl in enumerate(normal_labels) if lbl == normal_label]
            pos_subset = self.positions[indices]

            pos_db = DBSCAN(eps=pos_eps, min_samples=min_samples).fit(pos_subset)
            pos_labels = pos_db.labels_

            for pos_label in set(pos_labels):
                if pos_label == -1:
                    continue
                cluster_indices = [indices[i] for i, lbl in enumerate(pos_labels) if lbl == pos_label]
                final_clusters.append(cluster_indices)

        return final_clusters
    
    def order_samples_by_cog(self):
        self.samples = sorted(self.samples, key=lambda s: s.radial_distance_to_cog(self.cog))
    
    def get_dict(self):
        point_set = {}
        for i, sample in enumerate(self.samples):
            point_set[f"point_{i}"] = sample.get_dict(self.cog, self.mass)
        return point_set

class Shape:
    def __init__(self, shape: TopoDS_Shape, mesh: trimesh.Trimesh, point_distance: float):
        from .topods_utils import compute_shape_properties
        self.shape = shape
        self.mesh = mesh
        self.point_distance = point_distance
        volume, cog, _ = compute_shape_properties(shape)
        self.cog = np.array([cog.X(), cog.Y(), cog.Z()])
        density = 7.85e-3 # Steel g/mm^3
        self.mass = volume * density
        self.samples: PointSet = self.generate_samples()

    def filter_by_mask(self, mask: NDArray[np.bool_]):
        self.samples = self.samples.filter_by_mask(mask)

    def generate_samples(self) -> PointSet:
        from .topods_utils import generate_face_point_clouds, filter_points_outside_face, uv_position_to_global

        samples = generate_face_point_clouds(self.shape, self.point_distance)
        sample_points, sample_normals = [], []

        for face, points in samples.items():
            surface_points = filter_points_outside_face(points, face)
            _points, _normals = uv_position_to_global(surface_points, face)
            sample_points.extend(_points)
            sample_normals.extend(_normals)

        return PointSet(points=np.array(sample_points), normals=np.array(sample_normals), cog=self.cog, mass=self.mass)

    def create_z_maps(self, size: float, z_threshold: float, sampling_rate: int):
        from .geometry_utils import trimesh_z_maps
        z_maps = trimesh_z_maps(self.samples.positions, self.samples.normals, size, self.mesh, sampling_rate, z_threshold)
        self.samples.assign_z_maps(z_maps, size, sampling_rate)

    def remove_excessive_torque_samples(self, torque_threshold: float):
        self.samples.remove_excessive_torque_samples(torque_threshold)

    def remove_nonflat_samples(self, threshold: float, radius: float):
        self.samples.filter_by_flatness(threshold, radius)

    def get_best_cluster_samples(self) -> PointSet:
        self.samples.order_samples_by_cog()
        clusters = self.samples.cluster_samples(degree_eps=5, pos_eps=self.point_distance * 2)
        com = self.cog
        best_indices: List[int] = []
        for indices in clusters:
            sorted_indices = sorted(indices, key=lambda i: np.linalg.norm(self.samples.positions[i] - com))
            best_indices.append(sorted_indices[0])
        return PointSet(
            np.array([self.samples.positions[i] for i in best_indices]),
            np.array([self.samples.normals[i] for i in best_indices]),
            cog=self.cog,
            mass=self.mass
        )

class Tools(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def filter_points(self, shape: Shape) -> PointSet:
        pass

class SpongeTool(Tools):
    def __init__(self, width: float, height: float, max_torque: float, max_convex_curve: float, min_coverage: float):
        super().__init__("SpongeTool")
        self.width = width
        self.height = height
        self.max_width = math.sqrt(width*width + height*height)
        self.max_torque = max_torque
        self.max_convex_curve = max_convex_curve
        self.min_coverage = min_coverage

class SuctionTool(Tools):
    def __init__(self, max_width: float, max_height_diff: float, max_torque: float):
        super().__init__("SuctionTool")
        self.max_width = max_width
        self.max_height_diff = max_height_diff
        self.max_torque = max_torque
    
    def filter_points(self, shape: Shape) -> PointSet:
        shape.remove_excessive_torque_samples(self.max_torque)
        shape.create_z_maps(self.max_width * 2, z_threshold=400, sampling_rate=20)
        shape.remove_nonflat_samples(self.max_height_diff, self.max_width)
        best_samples = shape.get_best_cluster_samples()
        return best_samples
    
class GripperTool(Tools):
    def __init__(self, min_width: float, max_width: float, min_depth: float, max_depth: float, max_force: float, friction: float):
        super().__init__("GripperTool")
        self.min_width = min_width
        self.max_width = max_width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_force = max_force
        self.friction = friction