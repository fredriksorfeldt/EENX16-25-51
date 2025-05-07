from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.gp import gp_Vec, gp_Dir, gp_Lin, gp_Pln, gp_Ax2, gp_Ax3, gp_Trsf, gp_Pnt, gp_Mat
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_Cone, GeomAbs_SurfaceOfRevolution
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Wire, TopoDS_Edge, TopoDS_Compound, TopoDS_Face
from OCC.Core.TopAbs import TopAbs_ShapeEnum, TopAbs_REVERSED, TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.TDocStd import TDocStd_Document

from typing import List, Any, Dict, Tuple, Optional, Callable
from numpy.typing import NDArray

import numpy as np
import math

from .types import PointSet, PointSample

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

def compute_shape_properties(shape: TopoDS_Shape) -> Tuple[float, gp_Pnt, gp_Mat]:
    """ Returns: Volume, Center of mass point, Matrix of Inertia """
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass(), props.CentreOfMass(), props.MatrixOfInertia()

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

def generate_face_point_clouds(shape: TopoDS_Shape, point_distance: float) ->   Dict[TopoDS_Face, List[NDArray[np.float64]]]:
    faces = list(extract_subshapes(shape, TopAbs_FACE))
    large_faces = filter_large_faces(faces, min_area=10)

    face_point_clouds: Dict[TopoDS_Face, List[NDArray[np.float64]]] = {}
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


def filter_distance_from_outer_wire(points: PointSet, face: TopoDS_Face, min_dist: float, max_dist: float) -> PointSet:
    wire = breptools.OuterWire(face)

    mask = np.array([
        min_dist <= compute_distance_to_shape(gp_Pnt(*point.position), wire) <= max_dist
        for point in points
    ])
    return points.filter_by_mask(mask)

def compute_distance_to_shape(point: gp_Pnt, shape: TopoDS_Shape) -> float:
    # Create vertex from gp_pnt
    vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
    # Shortest distance measure
    dss = BRepExtrema_DistShapeShape(vertex, shape)
    dss.Perform()

    if not dss.IsDone():
        raise RuntimeError("Distance computation failed.")

    return dss.Value()

def get_closest_point_on_wire(point: gp_Pnt, wire: TopoDS_Wire) -> Optional[PointSample]:
    vertex = BRepBuilderAPI_MakeVertex(point).Vertex()
    dss = BRepExtrema_DistShapeShape(wire, vertex)
    dss.Perform()

    if dss.IsDone() and dss.NbSolution() > 0:
        new_point = dss.PointOnShape1(1)

        normal_vec = gp_Vec(point, new_point)
        normal = gp_Dir(normal_vec)

        return PointSample(np.array(new_point.Coord()), np.array(normal.Coord()))
    return None

def filter_point_pairs(points: PointSet, shape: TopoDS_Shape, max_angle: float, min_dist: float, max_dist: float) -> List[Tuple[PointSample, PointSample]]:
    from numpy import pi

    point_pairs = []
    for point in points:

        back = ray_intersect(point, shape, False)
        if not back:
            continue

        if pi - gp_Dir(*point.normal).Angle(gp_Dir(*back.normal)) > max_angle:
            continue

        if not (min_dist <= gp_Pnt(*point.position).Distance(gp_Pnt(*back.position)) <= max_dist):
            continue

        point_pairs.append((point, back))
    return point_pairs


def ray_intersect(point: PointSample, shape: TopoDS_Shape, up:bool=True) -> Optional[PointSample]:

    if not up:
        normal = gp_Dir(*point.normal).Reversed()
    line = gp_Lin(gp_Pnt(*point.position), normal)

    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(shape, 1e-6)

    intersector.Perform(line, 1e-6, 1e6)

    if intersector.IsDone() and intersector.NbPnt() > 0:
        normal = uv_normal(intersector.Face(1), intersector.UParameter(1), intersector.VParameter(1))
        return PointSample(np.array(intersector.Pnt(1).Coord()), np.array(normal))

    return None

def filter_closest_orthogonal(point_pairs: List[Tuple[PointSample, PointSample]], shape: TopoDS_Shape) -> List[Tuple[PointSample, float]]:

    valid_points = []
    for pair in point_pairs:
        p1, p2 = pair

        x = (p1.position[0] + p2.position[0]) / 2
        y = (p1.position[1] + p2.position[1]) / 2
        z = (p1.position[2] + p2.position[2]) / 2

        mid = gp_Pnt(x, y, z)

        plane = gp_Pln(mid, gp_Dir(*p1.normal))

        section = BRepAlgoAPI_Section(shape, plane).Shape()

        edges = []
        for edge in extract_subshapes(section, TopAbs_EDGE):
            edges.append(edge)

        wires = edges_to_wires(edges)
        outer_wire = None

        for wire in wires:
            face = BRepBuilderAPI_MakeFace(wire).Face()
            classifier = BRepClass3d_SolidClassifier(face, mid, 1e-6)
            if classifier.State() == 1:
                continue
            outer_wire = wire
            break

        if outer_wire is None:
            continue

        min_dist = math.inf
        candidate_points = []

        for edge in extract_subshapes(section, TopAbs_EDGE):
            point = get_closest_point_on_wire(mid, outer_wire)
            if point is not None:
                dist = mid.Distance(gp_Pnt(*point.position))

                if dist > min_dist:
                    continue
                elif dist < min_dist:
                    candidate_points = []

                point.orientation = gp_Ax2(gp_Pnt(*point.position), gp_Dir(*point.normal), gp_Dir(*p1.normal))
                candidate_points.append((point, gp_Pnt(*p1.position).Distance(gp_Pnt(*p2.position))))

        valid_points.extend(candidate_points)
    return valid_points

def edges_to_wires(edges: List[TopoDS_Edge]) -> List[TopoDS_Wire]:
    edge_sequence = TopTools_HSequenceOfShape()
    for edge in edges:
        edge_sequence.Append(edge)

    wire_sequence = ShapeAnalysis_FreeBounds.ConnectEdgesToWires(edge_sequence, 1e-6, False)


    wires = [wire_sequence.Value(i) for i in range(1, wire_sequence.Length() + 1)]
    return wires


def filter_gripper_occlution(bredth: float, depth: float, points: List[Tuple[PointSample, float]], shape: TopoDS_Shape, width_tol=1e-6, depth_tol=1) -> List[PointSample]:

    valid_points = []
    for point, width in points:
        box1 = BRepPrimAPI_MakeBox(
            gp_Pnt((-width / 2) - width_tol, (-bredth / 2), depth_tol),
            gp_Pnt(( width / 2) + width_tol, ( bredth / 2), 1e6)
        ).Shape()
        box2 = BRepPrimAPI_MakeBox(
            gp_Pnt((-width / 2) - width_tol -10, (-bredth / 2), -depth), 
            gp_Pnt((-width / 2) - width_tol,     ( bredth / 2), 1e6)
        ).Shape()
        box3 = BRepPrimAPI_MakeBox(
            gp_Pnt((width / 2) + width_tol,      (-bredth / 2), -depth), 
            gp_Pnt((width / 2) + width_tol + 10, ( bredth / 2), 1e6)
        ).Shape()

        gripper = TopoDS_Compound()
        builder = BRep_Builder()

        builder.MakeCompound(gripper)
        builder.Add(gripper, box1)
        builder.Add(gripper, box2)
        builder.Add(gripper, box3)

        transform = gp_Trsf()

        origin_ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        transform.SetDisplacement(origin_ax3, gp_Ax3(point.orientation))

        gripper = BRepBuilderAPI_Transform(gripper, transform).Shape()

        distance = BRepExtrema_DistShapeShape(shape, gripper).Value()

        if distance > 1e-6:
            valid_points.append(point)


    return valid_points

def filter_clustered_points(points: List[PointSample], shape: TopoDS_Shape, key: Callable[[PointSample], Any], angle_tol=0.0872664626, dist_tol:float=20):

    points.sort(key=key)
    point_face: Dict[PointSample, TopoDS_Face] = {}
    for face in extract_subshapes(shape, TopAbs_FACE):
        classifier = BRepClass_FaceClassifier()
        for point in points:
            classifier.Perform(face, gp_Pnt(*point.position), 1e-6)
            state = classifier.State()
            if state == TopAbs_ON:
                point_face[point] = face

    valid_points: List[PointSample] = []

    for p1 in points:
        flag = False
        for p2 in valid_points:
            if p1 == p2:
                continue

            if gp_Dir(*p1.normal).Angle(gp_Dir(*p2.normal)) <= angle_tol and (point_face.get(p1).IsEqual(point_face.get(p2)) or gp_Pnt(*p1.position).Distance(gp_Pnt(*p2.position)) <= dist_tol) :
                flag = True
                break

        if not flag:
            valid_points.append(p1)
    
    return valid_points
