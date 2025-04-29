from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_ShapeEnum, TopAbs_REVERSED
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_Cone, GeomAbs_SurfaceOfRevolution
from OCC.Core.gp import gp_Vec
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from typing import List, Any, Dict
from numpy.typing import NDArray
import numpy as np

def load_step_shape(file_path: str) -> TopoDS_Shape:
    doc = TDocStd_Document("")
    step_reader = STEPCAFControl_Reader()

    if not step_reader.ReadFile(file_path):
        raise RuntimeError("Unable to read STEP file.")
    
    step_reader.Transfer(doc)
    simple_reader = step_reader.ChangeReader()

    if simple_reader.NbRootsForTransfer() > 1:
        raise RuntimeError("Object contains more than one root.")
    
    simple_reader.TransferRoot()
    return simple_reader.Shape()

def uv_position_to_global(points: List[NDArray[np.float64]], face: TopoDS_Face) -> List[NDArray[np.float64]]:
    surface = BRep_Tool.Surface(face)
    sas = ShapeAnalysis_Surface(surface)
    normals = [uv_normal(face, u, v) for u, v in points]
    global_points = [sas.Value(u, v) for u, v in points]
    return [[pnt.X(), pnt.Y(), pnt.Z()] for pnt in global_points], normals

def uv_normal(face: TopoDS_Face, u: float, v: float) -> NDArray[np.float64]:
    adaptor_surface = BRepAdaptor_Surface(face)
    props = BRepLProp_SLProps(adaptor_surface, u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        raise ValueError(f"Normal is not defined at (u={u}, v={v}).")
    
    normal = props.Normal()

    if face.Orientation() == TopAbs_REVERSED:
        normal.Reverse()

    return np.array([normal.X(), normal.Y(), normal.Z()])

def extract_subshapes(shape: TopoDS_Shape, shape_type: TopAbs_ShapeEnum) -> List[Any]:
    explorer = TopExp_Explorer(shape, shape_type)
    
    subshapes: List[Any] = []
    while explorer.More():
        subshape = explorer.Current()
        explorer.Next()
        if not subshape.IsNull() and not subshape.IsSame(shape):
            subshapes.append(subshape)

    return subshapes

def filter_points_outside_face(point_param_coords: List[tuple[float, float]], face: TopoDS_Face) -> List[tuple[float, float]]:
    surface = BRep_Tool.Surface(face)
    classifier = BRepClass_FaceClassifier()
    return [(u,v) for u, v in point_param_coords if classifier.Perform(face, surface.Value(u, v), 1e-9) or classifier.State() == TopAbs_IN]

def compute_face_area(face: TopoDS_Face) -> float:
    props = GProp_GProps() 
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()

def compute_total_face_area(faces: List[TopoDS_Face]) -> float:
    return sum(compute_face_area(face) for face in faces)

def filter_large_faces(faces: List[TopoDS_Face], min_area: float = 75) -> List[TopoDS_Face]:
    return [f for f in faces if compute_face_area(f) > min_area]        

def generate_face_point_clouds(shape: TopoDS_Shape, point_distance: float) -> Dict[TopoDS_Face, List[NDArray[np.float64]]]:
    faces = list(extract_subshapes(shape, TopAbs_FACE))
    large_faces = filter_large_faces(faces, min_area=10)

    face_point_clouds: Dict[TopoDS_Face, List[tuple[float, float]]] = {}
    for face in large_faces:
        face_point_clouds[face] = generate_uniform_points_on_face(face, point_distance)
    
    return face_point_clouds

def generate_uniform_points_on_face(face: TopoDS_Face, resolution: float) -> List[NDArray[np.float64]]:
    adaptor_surface = BRepAdaptor_Surface(face)

    umin, umax = adaptor_surface.FirstUParameter(), adaptor_surface.LastUParameter()
    vmin, vmax = adaptor_surface.FirstVParameter(), adaptor_surface.LastVParameter()
    u_range, v_range = umax - umin, vmax - vmin

    # Handling different surface-type cases
    surface_type = adaptor_surface.GetType()
    if surface_type == GeomAbs_Cylinder:   
        radius = adaptor_surface.Cylinder().Radius()
        u_range = (radius * u_range)

    elif surface_type == GeomAbs_Sphere:
        radius = adaptor_surface.Sphere().Radius()
        u_range = (radius * u_range)
        v_range = (radius * v_range)

    elif surface_type == GeomAbs_Torus:
        torus = adaptor_surface.Torus()
        u_range = (torus.MajorRadius() * u_range)
        v_range = (torus.MinorRadius() * v_range)
    
    # Special handling for Cone (non-uniform uv density)
    if surface_type in (GeomAbs_Cone, GeomAbs_SurfaceOfRevolution):
        return _generate_nonuniform_revolution_uv(adaptor_surface, umin, umax, vmin, vmax, v_range, resolution)
    
    # Default uniform grid for other surface types
    return _generate_uniform_uv(umin, umax, u_range, vmin, vmax, v_range, resolution)

def _generate_nonuniform_revolution_uv(adaptor_surface: BRepAdaptor_Surface, umin: float, umax: float, vmin: float, vmax: float, v_range: float, resolution: float, epsilon: float = 1e-6) -> List[NDArray[np.float64]]:
    n_v = max(3, min(200, round(v_range / resolution)))
    n_v += (n_v % 2 == 0)
    v_vals = np.linspace(vmin + epsilon, vmax - epsilon, n_v)

    if adaptor_surface.GetType() == GeomAbs_Cone:
        axis = adaptor_surface.Cone().Axis()
    else:
        axis = adaptor_surface.AxeOfRevolution()

    origin = axis.Location()
    direction = gp_Vec(axis.Direction())

    u_grid: List[NDArray[np.float32]] = []
    for v in v_vals:
        p = adaptor_surface.Value(0, v)
        vec = gp_Vec(origin, p)
        radius = vec.Crossed(direction).Magnitude() / direction.Magnitude()
        circumference = 2 * np.pi * radius
        n_u = max(1, min(200, round(circumference / resolution)))
        u_grid.append(np.linspace(umin + epsilon, umax - epsilon, n_u))

    return [np.array([u, v]) for u_row, v in zip(u_grid, v_vals) for u in u_row]

def _generate_uniform_uv(umin: float, umax: float, u_range: float, vmin: float, vmax: float, v_range: float, resolution: float, epsilon: float = 1e-6) -> List[NDArray[np.float64]]:
    n_u = max(3, min(200, round(u_range / resolution)))
    n_v = max(3, min(200, round(v_range / resolution)))
    n_u += (n_u % 2 == 0)
    n_v += (n_v % 2 == 0)
    u_vals = np.linspace(umin + epsilon, umax - epsilon, n_u)
    v_vals = np.linspace(vmin + epsilon, vmax - epsilon, n_v)

    return [np.array([u, v]) for u in u_vals for v in v_vals]