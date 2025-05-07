import sys
import os
import trimesh
import argparse
import copy
import numpy as np
import json

from utils.types import *
from utils.json_utils import import_tools

# OpenCASCADE STEPControl for better STEP parsing
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

# Replaces utils.topods_utils.load_step_shape
def load_step_shape(path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read error (status {status}) for file '{path}'")
    reader.TransferRoots()
    return reader.Shape()


def process_file(file_path: str, tool_folder_path: str):
    root_shape = load_step_shape(file_path)

    trimesh_mesh = trimesh.load_mesh(file_path)
    mesh = trimesh.util.concatenate(trimesh_mesh.geometry.values()) if isinstance(trimesh_mesh, trimesh.Scene) else trimesh_mesh
    mesh.apply_transform(np.diag([1000, 1000, 1000, 1]))

    shape = Shape(root_shape, mesh, point_distance=8)

    tools = import_tools(tool_folder_path)
    if not tools:
        print("No valid tools found in the specified folder.")
        sys.exit(1)

    results = {}
    for i, tool in enumerate(tools):
        sc = copy.deepcopy(shape)
        pts = tool.filter_points(sc)
        results[f"tool_{i}"] = {
            "specifications": tool.get_dict(),
            "points": pts.get_dict()
        }

    # Ensure exports directory exists
    os.makedirs("./exports", exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join("./exports", f"{base}_points.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Exported points to {out_path}")


def main(cad_folder: str, tool_folder: str):
    cad_files = [
        os.path.join(cad_folder, f)
        for f in os.listdir(cad_folder)
        if f.lower().endswith(('.stp', '.step'))
    ]
    if not cad_files:
        print("No STEP files found in the specified CAD folder.")
        sys.exit(1)

    for path in cad_files:
        print(f"Processing {path}...")
        try:
            process_file(path, tool_folder)
        except Exception as e:
            print(f"Failed to process {path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process STEP files with tool filters"
    )
    parser.add_argument(
        "cad_folder_path", 
        type=str,
        help="Folder containing .stp/.step files"
    )
    parser.add_argument(
        "tool_folder_path", 
        type=str,
        help="Folder containing JSON tool definitions"
    )

    args = parser.parse_args()
    main(args.cad_folder_path, args.tool_folder_path)
