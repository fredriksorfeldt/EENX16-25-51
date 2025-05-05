import sys
import trimesh
import json
import argparse
from typing import List, Dict

from utils.types import *
from utils.topods_utils import load_step_shape, compute_shape_properties

def export_JSON(point_sets: Dict, topods_shape: TopoDS_Shape, filename: str = "points.json"):
    volume, cog, matrix_of_inertia = compute_shape_properties(topods_shape)
    density = 7.85e-3 # Steel g/mm^3
    mass = volume * density

    inertia_matrix = [
        [matrix_of_inertia.Value(1, 1), matrix_of_inertia.Value(1, 2), matrix_of_inertia.Value(1, 3)],
        [matrix_of_inertia.Value(2, 1), matrix_of_inertia.Value(2, 2), matrix_of_inertia.Value(2, 3)],
        [matrix_of_inertia.Value(3, 1), matrix_of_inertia.Value(3, 2), matrix_of_inertia.Value(3, 3)],
    ]

    export_dict = {
        "meta": {
            "mass": mass, 
            "centre_of_mass": [cog.X(), cog.Y(), cog.Z()],
            "matrix_of_inertia": inertia_matrix,
            "descriptions": {
                "units": {
                    "length": "mm",
                    "mass": "g",
                    "torque": "g*mm"
                },
                "position": "Global coordinate using step file coordinate space.",
                "normal": "Pointing towards the approach direction."
            }
        },
        "point_sets": point_sets
    }

    with open(filename, "w") as file:
        json.dump(export_dict, file, indent=4)

def main(file_path: str):
    # Load as opencascade object
    root_shape = load_step_shape(file_path)

    # Load as trimesh object
    trimesh_mesh  = trimesh.load_mesh(file_path)

    # This ensures the mesh is of type trimesh.trimesh
    if isinstance(trimesh_mesh, trimesh.Scene):
        # Try to combine geometry into a single mesh
        mesh = trimesh.util.concatenate(
            [g for g in trimesh_mesh.geometry.values()]
        )
    else:
        mesh = trimesh_mesh

    # Mesh needs to be scaled by 1000
    scale_factor = 1000

    S = np.eye(4)
    S[:3, :3] *= scale_factor
    mesh.apply_transform(S)

    # Creating Shape
    shape = Shape(root_shape, mesh, point_distance = 8)

    # Creating Tools
    suction = SuctionTool(10, 1, 5)

    # Applying the tool-filters on the shape points
    suction_point_set = suction.filter_points(shape)

    # Creating to export dict
    export_dict = {}
    export_dict["suction_0"] = suction_point_set.get_dict()

    export_JSON(export_dict, root_shape)

if __name__ == "__main__":
    # Set up parsing arguments
    parser = argparse.ArgumentParser(description="Run the CAD visualizer")
    parser.add_argument("file_path", type=str, help="Path to the STEP (.stp) file to visualize")

    # Parse arguments
    args = parser.parse_args()

    # Ensure a file path is provided
    if not args.file_path:
        print("Error: Please provide a STEP file path")
        sys.exit(1)

    main(args.file_path)