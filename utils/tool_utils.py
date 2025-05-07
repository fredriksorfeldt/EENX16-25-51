import json
import numpy as np
from types import *

def load_tool_from_json(json_path: str) -> (SuctionTool | SpongeTool | GripperTool): 
    '''
    Loads tool specification from JSON file
    '''

    # Load json file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Check that tool type is valid
    valid_tool_types: list[str] = ["suction", "gripper", "sponge"]
    tool_type: str = data["type"]
    assert tool_type in valid_tool_types, "Invalid tool type provided"

    # Suction tools
    if tool_type == "suction":

        # Fetch max width correctly for different end-effector faces
        if data["specification"]["end_effector"]["type"] == "circular":
            width = data["specification"]["end_effector"]["dimensions"]["diameter"]

        elif data["specification"]["end_effector"]["type"] == "rectangular":
            x = data["specification"]["end_effector"]["dimensions"]["x"]
            y = data["specification"]["end_effector"]["dimensions"]["y"]

            width = np.sqrt(x**2 + y**2)
        
        # Fetch other needed specification data
        max_height_diff = data["specification"]["max_distances"]["max_height_diff"]
        max_force = data["specification"]["max_load"]["max_force"]
        max_torque = data["specification"]["max_load"]["max_torque"]

        # Return a suction tool
        return SuctionTool(width, max_height_diff, max_force, max_torque)
    
    # Sponge tools
    if tool_type == "sponge":

        # Fetch max width correctly for different end-effector faces
        if data["specification"]["end_effector"]["type"] == "circular":
            width = data["specification"]["end_effector"]["dimensions"]["diameter"]

        elif data["specification"]["end_effector"]["type"] == "rectangular":
            x = data["specification"]["end_effector"]["dimensions"]["x"]
            y = data["specification"]["end_effector"]["dimensions"]["y"]

            width = np.sqrt(x**2 + y**2)
        
        # Fetch other needed specification data
        max_height_diff = data["specification"]["max_distances"]["max_height_diff"]
        max_convex_curve = data["specification"]["max_distances"]["max_convex_curve"]
        max_force = data["specification"]["max_load"]["max_force"]
        max_torque = data["specification"]["max_load"]["max_torque"]
        min_coverage = data["specification"]["min_coverage"]

        # Return a sponge tool
        return SpongeTool(width, max_torque, max_convex_curve, min_coverage)
    
    # Gripper tools
    if tool_type == "gripper":
        min_width = data["specification"]["max_distances"]["min_stroke"]
        max_width = data["specification"]["max_distances"]["max_stroke"]
        min_depth = data["specification"]["max_distances"]["min_reach"]
        max_depth = data["specification"]["max_distances"]["max_reach"]
        max_force = data["specification"]["max_load"]["max_force"]
        return GripperTool(min_width, max_width, min_depth, max_depth, max_force)
