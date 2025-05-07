import sys
import trimesh
import argparse
import copy
import numpy as np

from utils.types import *
from utils.topods_utils import load_step_shape
from utils.json_utils import import_tools, export_JSON
from OCC.Display.SimpleGui import init_display

def main(file_path: str, tool_folder_path: str):
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

    # Load tools (Suction and sponge)
    tools = import_tools(tool_folder_path)

    if not tools:
        print("No valid tools found in the specified folder.")
        sys.exit(1)

    # Applying the tool-filters on the shape points
    export_dict = {}
    for i, tool in enumerate(tools):
        shape_copy = copy.deepcopy(shape)
        point_set = tool.filter_points(shape_copy)
        export_dict[f"tool_{i}"] = {"specifications": tool.get_dict(), "points": point_set.get_dict()}

        display, start_display, _, _ = init_display()
        display.DisplayShape(shape.shape)
        display.DisplayShape([gp_Pnt(*point.position) for point in point_set.samples])

        start_display()

    export_JSON(export_dict, shape, directory="./exports")

if __name__ == "__main__":
    # Set up parsing arguments
    parser = argparse.ArgumentParser(description="Run the CAD visualizer")
    parser.add_argument("file_path", type=str, help="Path to the STEP (.stp) file")
    parser.add_argument("tool_folder_path", type=str, help="Path to the folder containing JSON tool data")

    # Parse arguments
    args = parser.parse_args()

    main(args.file_path, args.tool_folder_path)
