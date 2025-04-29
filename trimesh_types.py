
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import math
import trimesh
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

from OCC.Core.TopoDS import TopoDS_Shape

## TODO: 
#  - Implement normal clustering.
#  - Generate points from parametric surface
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

class PointSet:
    def __init__(self, points: NDArray[np.float64], normals: NDArray[np.float64]):
        self.samples = [PointSample(p, n) for p, n in zip(points, normals)]
    
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
    
    def filter_by_mask(self, mask: NDArray[np.bool_]) -> "PointSet":
        return PointSet(self.positions[mask], self.normals[mask])

    def assign_z_maps(self, z_maps: NDArray[np.float64], size: float, sampling_rate: int):
        for sample, z_map_data in zip(self.samples, z_maps):
            sample.z_map = ZMap(z_map_data, size, sampling_rate)

    def filter_by_flatness(self, threshold: float, radius: float):
        self.samples = [s for s in self.samples if s.z_map.is_flat(threshold, radius)]

    def cluster_samples(self, eps: float, min_samples: int = 1) -> List[List[int]]:
        if len(self.samples) == 0:
            raise ValueError("Cannot cluster: no samples available.")
        if len(self.samples) == 1:
            return [[0]]
        
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.positions)
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        clusters: List[List[int]] = [[] for _ in range(n_clusters)]

        for idx, label in enumerate(labels):
            if label != -1:
                clusters[label].append(idx)
        return clusters


class Shape:
    def __init__(self, shape: TopoDS_Shape, mesh: trimesh.Trimesh, point_distance: float):
        self.shape = shape
        self.mesh = mesh
        self.point_distance = point_distance
        self.samples: PointSet = self.generate_samples(point_distance)
        
    @property
    def center_of_mass(self):
        return self.mesh.center_mass

    def filter_by_mask(self, mask: NDArray[np.bool_]):
        self.samples = self.samples.filter_by_mask(mask)

    def generate_samples(self, point_distance: float) -> PointSet:
        from trimesh_utils.topods_utils import generate_face_point_clouds, filter_points_outside_face, uv_position_to_global

        samples = generate_face_point_clouds(self.shape, point_distance)
        sample_points, sample_normals = [], []

        for face, points in samples.items():
            surface_points = filter_points_outside_face(points, face)
            _points, _normals = uv_position_to_global(surface_points, face)
            sample_points.extend(_points)
            sample_normals.extend(_normals)

        return PointSet(points=np.array(sample_points), normals=np.array(sample_normals))

    def create_z_maps(self, size: float, z_threshold: float, sampling_rate: int):
        from .trimesh_geometry_utils import trimesh_z_maps
        z_maps = trimesh_z_maps(self.samples.positions, self.samples.normals, size, self.mesh, sampling_rate, z_threshold)
        self.samples.assign_z_maps(z_maps, size, sampling_rate)

    def remove_nonflat_samples(self, threshold: float, radius: float):
        self.samples.filter_by_flatness(threshold, radius)

    def get_best_cluster_point(self) -> List[int]:
        clusters = self.samples.cluster_samples(self.point_distance * 2)
        com = self.center_of_mass
        best_indices: List[int] = []
        for indices in clusters:
            sorted_indices = sorted(indices, key=lambda i: np.linalg.norm(self.samples.positions[i] - com))
            best_indices.append(sorted_indices[0])
        return best_indices

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
    def __init__(self, max_width: float, max_height_diff: float):
        super().__init__("SuctionTool")
        self.max_width = max_width
        self.max_height_diff = max_height_diff
    
    def filter_points(self, shape: Shape) -> PointSet:
        shape.create_z_maps(self.max_width * 2, z_threshold=400, sampling_rate=20)
        shape.remove_nonflat_samples(self.max_height_diff, self.max_width)
        best_indices = shape.get_best_cluster_point()
        return PointSet(
            np.array([shape.samples.positions[i] for i in best_indices]),
            np.array([shape.samples.normals[i] for i in best_indices])
        )
    
class GripperTool(Tools):
    def __init__(self, min_width: float, max_width: float, min_depth: float, max_depth: float, max_force: float, friction: float):
        super().__init__("GripperTool")
        self.min_width = min_width
        self.max_width = max_width
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_force = max_force
        self.friction = friction