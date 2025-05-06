import numpy as np
import trimesh
from typing import Any, List
from numpy.typing import NDArray
from numba import njit

def triangle_areas(triangles: NDArray[np.float64]) -> float:
    v1, v2, v3 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = v2 - v1, v3 - v1
    cross = np.cross(ab, ac)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas

def trimesh_z_maps(points: NDArray[np.float64], normals: NDArray[np.float64], size: float, mesh: trimesh.Trimesh, sampling_rate: int, z_max_threshold: float):
    transform_matrices = [transform_matrix_to_z_axis(n, p) for n, p in zip(normals, points)]

    x_samples = np.linspace(-size/2, size/2, sampling_rate)
    y_samples = np.linspace(-size/2, size/2, sampling_rate)

    grid_x, grid_y = np.meshgrid(x_samples, y_samples)
    samples_flat = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    origin_triangles = mesh.triangles
    valid_indices = remove_tiny_triangles(origin_triangles, min_area=0.5)

    z_max_maps: List[NDArray[Any]] = []
    for TR in transform_matrices:
        large_triangles = origin_triangles[valid_indices]
        triangles = apply_transform_to_triangles(large_triangles, TR)
        triangles += np.array([0, 0, -z_max_threshold])
        filtered_triangles = filter_outside_triangles(triangles, z_range=0, x_range=size/2, y_range=size/2)

        point_mask_batch = fast_point_in_triangle_numba(samples_flat, filtered_triangles[:, :, :2])

        normals_batch, D, valid_planes = compute_plane_batch(filtered_triangles)
        filtered_triangles = filtered_triangles[valid_planes]
        point_mask_batch = point_mask_batch[valid_planes]
        normals_batch = normals_batch[valid_planes]
        D = D[valid_planes]

        if len(filtered_triangles) == 0:
            continue

        x_grid = np.repeat(samples_flat[:, 0][None, :], len(filtered_triangles), axis=0)
        y_grid = np.repeat(samples_flat[:, 1][None, :], len(filtered_triangles), axis=0)

        z_values = z_from_xy_numba(x_grid, y_grid, normals_batch, D)
        z_values[~point_mask_batch] = np.nan
        z_values += z_max_threshold
        # NOTE: np.nanmax() triggers: "RuntimeWarning: All-NaN slice encountered", should be safe.
        z_map = np.nanmax(z_values, axis=0) 

        z_max_maps.append(z_map)

    return np.array(z_max_maps), transform_matrices

def remove_tiny_triangles(triangles: NDArray[np.float64], min_area: float) -> NDArray[np.bool_]:
    areas = triangle_areas(triangles)
    valid_indices = areas >= min_area
    return valid_indices

def apply_transform_to_triangles(triangles: NDArray[np.float64], transform: NDArray[np.float64]) -> NDArray[np.float64]:
    N = triangles.shape[0]
    verts_flat = triangles.reshape(-1, 3)
    verts_h = np.hstack([verts_flat, np.ones((verts_flat.shape[0], 1))])
    verts_transformed = (transform @ verts_h.T).T[:, :3]
    return verts_transformed.reshape(N, 3, 3)

# TODO: Explore better methods for this filter
def filter_outside_triangles(triangles: NDArray[np.float64], z_range: float, x_range: float, y_range: float):
    x_values, y_values, z_values = triangles[:, :, 0], triangles[:, :, 1], triangles[:, :, 2]

    z_mask = np.any(z_values < z_range, axis=1)
    x_max_mask = np.all(x_values > x_range, axis=1)
    x_min_mask = np.all(x_values < -x_range, axis=1)
    y_max_mask = np.all(y_values > y_range, axis=1)
    y_min_mask = np.all(y_values < -y_range, axis=1)

    outside_mask = x_max_mask | x_min_mask | y_max_mask | y_min_mask
    final_mask = z_mask & (~outside_mask)

    return triangles[final_mask]

@njit
def z_from_xy_numba(x: NDArray[np.float64], y: NDArray[np.float64], normals: NDArray[np.float64], D: NDArray[np.float64]) -> NDArray[np.float64]:
    T, P = x.shape
    z = np.empty((T, P), dtype=np.float64)

    for i in range(T):
        A, B, C = normals[i, 0], normals[i, 1], normals[i, 2]
        d = D[i]
        for j in range(P):
            z[i, j] = -(A * x[i, j] + B * y[i, j] + d) / C
    return z

def compute_plane_batch(triangles: NDArray[np.float64], epsilon: float = 1e-8):
    v1, v2, v3 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = v2-v1, v3-v1
    normals = np.cross(ab, ac)
    valid = np.abs(normals[:, 2]) > epsilon
    D = -np.einsum('ij,ij->i', normals, v1)
    return normals, D, valid

@njit
def fast_point_in_triangle_numba(points: NDArray[np.float64], tri_batch: NDArray[np.float64], epsilon: float=1e-7):
    T, P = tri_batch.shape[0], points.shape[0]
    mask = np.zeros((T, P), dtype=np.bool_)

    for t in range(T):
        a, b, c = tri_batch[t, 0], tri_batch[t, 1], tri_batch[t, 2]
        area_abc = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        for p in range(P):
            x, y = points[p]
            pa = (x - a[0], y - a[1])
            pb = (x - b[0], y - b[1])
            pc = (x - c[0], y - c[1])
            area_pbc = 0.5 * abs(pb[0] * pc[1] - pc[0] * pb[1])
            area_pac = 0.5 * abs(pa[0] * pc[1] - pc[0] * pa[1])
            area_pab = 0.5 * abs(pa[0] * pb[1] - pb[0] * pa[1])
            if abs((area_pbc + area_pac + area_pab) - area_abc) < epsilon:
                mask[t, p] = True

    return mask

def rotation_matrix_to_z(normal: NDArray[np.float64]) -> NDArray[np.float64]:
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal = normal / np.linalg.norm(normal)
    z_axis = np.array([0, 0, 1])

    if np.allclose(normal, z_axis):
        return np.eye(3)
    if np.allclose(normal, -z_axis):
        arb = np.array([1, 0, 0])
        if np.allclose(normal, arb):
            arb = np.array([0, 1, 0])
        k = np.cross(normal, arb)
        k /= np.linalg.norm(k)
        theta = np.pi
    else:
        k = np.cross(normal, z_axis)
        k /= np.linalg.norm(k)
        ab_dot = np.clip(np.dot(normal, z_axis), -1, 1)
        theta = np.arccos(ab_dot)

    skew_k = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * skew_k + (1 - np.cos(theta)) * (skew_k @ skew_k)
    return R

def transform_matrix_to_z_axis(normal: NDArray[np.float64], point: NDArray[np.float64]):
    R = rotation_matrix_to_z(normal)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    t = -R @ point
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

from scipy.spatial.transform import Rotation as R

def normal_to_quaternion(normal: NDArray[np.float64]) -> NDArray[np.float64]:
    ref = np.array([0.0, 0.0, 1.0])  # Default tool direction
    normal = normal / np.linalg.norm(normal)  # Just in case

    if np.allclose(normal, ref):
        return np.array([0, 0, 0, 1])  # Identity quaternion
    elif np.allclose(normal, -ref):
        # 180° rotation around any axis perpendicular to ref
        return R.from_rotvec(np.pi * np.array([1, 0, 0])).as_quat()
    else:
        rot_axis = np.cross(ref, normal)
        rot_angle = np.arccos(np.clip(np.dot(ref, normal), -1.0, 1.0))
        rot = R.from_rotvec(rot_axis / np.linalg.norm(rot_axis) * rot_angle)
        return rot.as_quat()