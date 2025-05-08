import os
import json
from typing import Dict, List, Any
import numpy as np
import time

from utils.types import SuctionTool, Tools, SpongeTool, Shape

def get_export_filename(base_name: str = "points", directory: str = "."):
    index = 1
    while True:
        filename = f"{base_name}_{index}.json"
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return filepath
        index += 1

def export_JSON(point_sets: Dict[str, Any], shape: Shape, directory: str = ".", base_filename: str = "points", cad_file_path: str = "unspecified"):
    export_dict: Dict[str, Any] = {
        "cad_file_path": cad_file_path,
        "export_time": time.asctime(),
        "meta": shape.get_dict(),
        "point_sets": point_sets
    }

    os.makedirs(directory, exist_ok=True)
    filepath = get_export_filename(base_filename, directory)

    with open(filepath, "w") as file:
        json.dump(export_dict, file, indent=4)
    
    print(f"Exported to {filepath}")
    return filepath


def import_tools(tool_folder_path: str) -> List[Tools]:
    tools: List[Tools] = []
    for filename in os.listdir(tool_folder_path):
        if filename.endswith('.json'):
            f_path = os.path.join(tool_folder_path, filename)
            with open(f_path, 'r') as file:
                try:
                    data = json.load(file)
                    tool_type = data.get("type")
                    if tool_type in ["sponge", "suction"]:
                        tools.append(extract_tool_json(data, tool_type))
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding {filename}: {e}")

    return tools

def extract_tool_json(data: Dict[str, Any], tool_type: str) -> Tools:
    name = data.get("name")
    end_effector = data.get("specification", {}).get("end_effector", {})
    load = data.get("specification", {}).get("max_load", {})
    distances = data.get("specification", {}).get("max_distances", {})
    min_coverage = data.get("specification", {}).get("min_coverage")

    dimensions = end_effector.get("dimensions", {})
    max_torque = load.get("max_torque", 1e12)
    max_height_diff = distances.get("max_height_diff", 0.5)


    if tool_type == "sponge":
        if max_torque:
            return SpongeTool(name, max_width=dimensions["x"], max_height=dimensions["y"], max_penetration=max_height_diff, max_torque=max_torque, min_coverage=min_coverage)
        else:
            return SpongeTool(name, max_width=dimensions["x"], max_height=dimensions["y"], max_penetration=max_height_diff, max_torque=1e8, min_coverage=min_coverage)
    elif tool_type == "suction":
        diameter = dimensions.get("diameter", 0)
        if max_torque:
            return SuctionTool(name, max_width=diameter / 2, max_penetration=max_height_diff, max_torque=max_torque)
        else:
            return SuctionTool(name, max_width=diameter / 2, max_penetration=max_height_diff, max_torque=1e8)