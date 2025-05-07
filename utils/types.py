
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import trimesh
import copy
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from scipy.ndimage import rotate

from OCC.Core.TopoDS import TopoDS_Shape

from .geometry_utils import trimesh_z_maps

class ZMap:
    def __init__(self, data: NDArray[np.float64], transformation_matrix: NDArray[np.float64], size: float, sampling_rate: int):
        self.data: NDArray[np.float64] = data
        self.transformation_matrix = transformation_matrix
        self.size: float = size
        self.sampling_rate: int = sampling_rate

    def rotate(self, theta: float):
        Rz = np.array([
              [np.cos(theta),-np.sin(theta), 0, 0], 
              [np.sin(theta), np.cos(theta), 0, 0],
              [0,             0,             1, 0],
              [0,             0,             0, 1]
        ])
        self.transformation_matrix = np.dot(self.transformation_matrix, Rz)

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
    
    def extract_rectangle_patches(self, width: float, height: float, rotations: int = 4):
        step_size = self.size / self.sampling_rate

        patch_width_px = int(width / step_size)
        patch_height_px = int(height / step_size)

        if patch_width_px > self.sampling_rate or patch_height_px > self.sampling_rate:
            raise ValueError("Patch goes out of bounds")
        
        center_y, center_x = self.sampling_rate // 2, self.sampling_rate // 2
        half_h, half_w = patch_height_px // 2, patch_width_px // 2

        mask = np.zeros((self.sampling_rate, self.sampling_rate), dtype=bool)
        mask[
            center_y - half_h:center_y + half_h,
            center_x - half_w:center_x + half_w
        ] = True
        
        patches, angles = [], []
        for i in range(rotations):
            angle = (2 * np.pi / rotations) * i
            rotated_mask = rotate(mask.astype(float), angle, reshape=False, order=0) > 0.5
            rotated_mask = np.resize(rotated_mask, self.data.shape)
            patch = np.where(rotated_mask, self.data, np.nan)
            patches.append(patch)
            angles.append(angle)

        return patches, angles
    
    def is_flat(self, penetration_threshold: float, patch: NDArray[np.float64]) -> np.bool_:
        return np.all(np.abs(patch) < penetration_threshold)
    
    def is_covered(self, coverage_threshold: float, penetration_threshold: float, patch: NDArray[np.float64]) -> np.bool_:
        peak_value = np.nanmax(patch)
        if peak_value > penetration_threshold:
            return False
        difference = np.abs(patch - peak_value)

        valid_mask = difference <= penetration_threshold

        n_valids = np.sum(valid_mask)
        total_values = patch.size
        fraction = n_valids / total_values

        return fraction >= coverage_threshold


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
            "position": { "x": self.position[0], "y": self.position[1], "z": self.position[2]},
            "quaternion": { "x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2]),"w": float(quat[3])},
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

    def assign_z_maps(self, z_maps: NDArray[np.float64], transform_matrices: NDArray[np.float64], size: float, sampling_rate: int):
        for sample, z_map_data, transform_matrix in zip(self.samples, z_maps, transform_matrices):
            sample.z_map = ZMap(z_map_data, transform_matrix, size, sampling_rate)

    def filter_nonflat(self, penetration_threshold: float, radius: float) -> None:
        self.samples = [s for s in self.samples if s.z_map.is_flat(penetration_threshold, s.z_map.extract_patch(radius))]

    def filter_noncovered(self, coverage_threshold: float, penetration_threshold: float, width: float, height: float):
        new_samples = []
        for s in self.samples:
            patches, angles = s.z_map.extract_rectangle_patches(width, height)
            for patch, angle in zip(patches, angles):
                if s.z_map.is_covered(coverage_threshold, penetration_threshold, patch):
                    new_sample = copy.deepcopy(s)
                    new_sample.z_map.rotate(angle)
                    new_samples.append(new_sample)
                    break
        self.samples = new_samples

    def remove_excessive_torque_samples(self, torque_threshold: float) -> None:
        self.samples = [s for s in self.samples if (s.radial_distance_to_cog(self.cog) * self.mass) < torque_threshold]

    def cluster_samples(self, degree_eps: float, pos_eps: float, min_samples: int = 1) -> List[List[int]]:
        if len(self.samples) == 0:
            return [[]]
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

        # Position clustering for each normal cluster
        unique_labels = set(normal_labels)
        unique_labels.discard(-1)

        clusters: List[List[int]] = []
        for normal_label in unique_labels:
            indices = [i for i, lbl in enumerate(normal_labels) if lbl == normal_label]
            pos_subset = self.positions[indices]

            pos_db = DBSCAN(eps=pos_eps, min_samples=min_samples).fit(pos_subset)
            pos_labels = pos_db.labels_

            for pos_label in set(pos_labels):
                if pos_label == -1:
                    continue
                cluster_indices = [indices[i] for i, lbl in enumerate(pos_labels) if lbl == pos_label]
                clusters.append(cluster_indices)

        return clusters
    
    def order_samples_by_cog(self):
        self.samples = sorted(self.samples, key=lambda s: s.radial_distance_to_cog(self.cog))
    
    def get_dict(self):
        point_set = {}
        for i, sample in enumerate(self.samples):
            point_set[f"point_{i}"] = sample.get_dict(self.cog, self.mass)
        return point_set

class Shape:
    def __init__(self, shape: TopoDS_Shape, mesh: trimesh.Trimesh, point_distance: float):
        self.shape = shape
        self.mesh = mesh
        self.point_distance = point_distance
        self.mass, self.cog, self.matrix_of_intertia = self._compute_properties()
        self.samples: PointSet = self.generate_samples()

    def _compute_properties(self):
        from .topods_utils import compute_shape_properties
        volume, cog, matrix_of_inertia = compute_shape_properties(self.shape)
        cog_array = np.array([cog.X(), cog.Y(), cog.Z()])
        density = 7.85e-3 # Steel g/mm^3
        mass = volume * density
        intertia_matrix = [
            [matrix_of_inertia.Value(1, 1), matrix_of_inertia.Value(1, 2), matrix_of_inertia.Value(1, 3)],
            [matrix_of_inertia.Value(2, 1), matrix_of_inertia.Value(2, 2), matrix_of_inertia.Value(2, 3)],
            [matrix_of_inertia.Value(3, 1), matrix_of_inertia.Value(3, 2), matrix_of_inertia.Value(3, 3)],
        ]
        return mass, cog_array, intertia_matrix

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
        if len(self.samples) == 0:
            return 0
        z_maps, transform_matrices = trimesh_z_maps(self.samples.positions, self.samples.normals, size, self.mesh, sampling_rate, z_threshold)
        self.samples.assign_z_maps(z_maps, transform_matrices, size, sampling_rate)  

    def remove_excessive_torque_samples(self, torque_threshold: float):
        self.samples.remove_excessive_torque_samples(torque_threshold)

    def remove_nonflat_samples(self, penetration_threshold: float, radius: float):
        self.samples.filter_nonflat(penetration_threshold, radius)

    def remove_noncovered_footprint(self, coverage_threshold: float, penetration_threshold: float, width: float, height: float):
        self.samples.filter_noncovered(coverage_threshold, penetration_threshold, width, height)

    def get_best_cluster_samples(self) -> PointSet:
        if len(self.samples) == 0:
            return self.samples

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
    
    def get_dict(self) -> Dict[str, Any]:
        return {
            "mass": self.mass, 
            "centre_of_mass": list(self.cog),
            "matrix_of_inertia": self.matrix_of_intertia,
            "descriptions": {
                "units": {
                    "length": "mm",
                    "mass": "g",
                    "torque": "g*mm"
                },
                "coordinate_frame": "step_file_origin",
                "tool_origin": "step_file_origin",
                "position": "Global coordinate using step file coordinate space.",
                "normal": "Pointing towards the approach direction."
            }
        }

class Tools(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def filter_points(self, shape: Shape) -> PointSet:
        pass

class SpongeTool(Tools):
    def __init__(self, name: str, max_width: float, max_height: float, max_penetration: float, max_torque: float, min_coverage: float):
        super().__init__(name)
        self.max_width = max_width
        self.max_height = max_height
        self.max_penetration = max_penetration
        self.max_torque = max_torque
        self.min_coverage = min_coverage

    def filter_points(self, shape: Shape) -> PointSet:
        shape.remove_excessive_torque_samples(self.max_torque)
        shape.create_z_maps(np.linalg.norm(np.array([self.max_width, self.max_height])), z_threshold=1e5, sampling_rate=20)
        shape.remove_noncovered_footprint(self.min_coverage, self.max_penetration, self.max_width, self.max_height)
        # best_samples = shape.get_best_cluster_samples()
        return shape.samples
    
    def get_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "type": "sponge",
                "max_width": self.max_width,
                "max_penetration": self.max_penetration,
                "max_torque": self.max_torque,
                "min_coverage": self.min_coverage}

class SuctionTool(Tools):
    def __init__(self, name: str, max_width: float, max_penetration: float, max_torque: float):
        super().__init__(name)
        self.max_width = max_width
        self.max_penetration = max_penetration
        self.max_torque = max_torque
    
    def filter_points(self, shape: Shape) -> PointSet:
        shape.remove_excessive_torque_samples(self.max_torque)
        shape.create_z_maps(self.max_width * 2 + 1, z_threshold=1e5, sampling_rate=20)
        shape.remove_nonflat_samples(self.max_penetration, self.max_width)
        best_samples = shape.get_best_cluster_samples()
        return best_samples
    
    def get_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "type": "suction",
                "max_width": self.max_width,
                "max_penetration": self.max_penetration,
                "max_torque": self.max_torque}
    
class GripperTool(Tools):
    def __init__(self, name: str, min_width: float, max_width: float, min_depth: float, max_depth: float, max_force: float, friction: float):
        super().__init__("GripperTool")
        self.min_width = min_width
        self.max_width = max_width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_force = max_force
        self.friction = friction


import matplotlib.pyplot as plt

def display_zmap(z_map: NDArray[np.float64], sampling_rate: int, filename: str = "zmap_plot.png"):
    z_map_2d = z_map.reshape((sampling_rate, sampling_rate))
    z_masked = np.ma.masked_invalid(z_map_2d)
    plt.figure(figsize=(6, 5))
    plt.imshow(z_masked, cmap='plasma', origin='lower')
    plt.colorbar(label='Z Height')
    plt.title('Z-Map Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(filename)  # Save instead of showing
    print(f"Saved Z-map to {filename}")