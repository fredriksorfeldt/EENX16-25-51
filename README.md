# EENX16-25-51

NOTE 2025-04-29:
When creating a shape object a topoDS_Shape amd a trimesh Mesh should be passed, however trimesh uses meters and topoDS_Shape uses milimeters as unit. 

# This requires us to scale the trimesh object like:

root_shape = load_step_shape(file_path)
loaded  = trimesh.load_mesh(file_path)

if isinstance(loaded, trimesh.Scene):
    # Try to combine geometry into a single mesh
    mesh = trimesh.util.concatenate(
        [g for g in loaded.geometry.values()]
    )
else:
    mesh = loaded

scale_factor = 1000

S = np.eye(4)
S[:3, :3] *= scale_factor
mesh.apply_transform(S)

shape = Shape(root_shape, mesh, point_distance = 8)
