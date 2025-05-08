import trimesh
from trimesh.path.entities import Line
from trimesh.path import Path3D
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List
import numpy as np
from utils.types import Shape

def display_shape(shape: Shape):
    scene = trimesh.Scene()
    scene.add_geometry(shape.mesh)
    if len(shape.samples.samples):
        scene.add_geometry(create_normal_lines(shape.samples.positions, shape.samples.normals, color=[0, 255, 0, 255]))
    scene.show(line_settings={'line_width': 0.1}, background=[10,10,10,255])

def create_normal_lines(points: NDArray[np.float64], normals: NDArray[np.float64], scale:float=4.0, color:List[float]=[0, 0, 255, 255]) -> Path3D:
    points = np.asarray(points)
    normals = np.asarray(normals)
    vertices = np.vstack((points, points + normals * scale))
    n = points.shape[0]
    lines = [Line([i, i + n]) for i in range(n)]
    colors = [color] * n
    paths = Path3D(entities=lines, vertices=vertices, colors=colors)
    return paths

def display_zmap(z_map: NDArray[np.float64], sampling_rate: int):
    z_map_2d = z_map.reshape((sampling_rate, sampling_rate))
    z_masked = np.ma.masked_invalid(z_map_2d)
    plt.figure(figsize=(6, 5))
    plt.imshow(z_masked, cmap='plasma', origin='lower')
    plt.colorbar(label='Z Height')
    plt.title('Z-Map Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()