"""
SceneAdapter: bridges compare_models scene representation to SceneEval's Scene interface.

3D-FRONT uses y-up coordinates (x, y=height, z).
SceneEval uses z-up coordinates (x, y, z=height).

This adapter converts on the fly so SceneEval metrics can be used as-is.
"""

import os
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transform: 3D-FRONT (y-up) → SceneEval (z-up)
#   new_x = old_x
#   new_y = old_z
#   new_z = old_y

Y_UP_TO_Z_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)


def _to_z_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_transform(Y_UP_TO_Z_UP)
    return m


# Cached human mesh (SceneEval `human_model`, y-up GLB → z-up), see blender_scene.py
_human_template_zup: Optional[trimesh.Trimesh] = None
_human_template_load_failed: bool = False


def _resolve_human_glb_path() -> Optional[Path]:
    env = os.environ.get("SCENEEVAL_HUMAN_MODEL")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    repo_root = Path(__file__).resolve().parent
    p = repo_root / "assets" / "human.glb"
    if p.is_file():
        return p
    return None


def _load_human_template_zup() -> Optional[trimesh.Trimesh]:
    """
    Single cached load of SceneEval's reference human (same asset as Blender SceneEval).
    """
    global _human_template_zup, _human_template_load_failed
    if _human_template_load_failed:
        return None
    if _human_template_zup is not None:
        return _human_template_zup
    path = _resolve_human_glb_path()
    if path is None:
        _human_template_load_failed = True
        print(
            "Warning: human.glb not found (set SCENEEVAL_HUMAN_MODEL or place "
            "SceneEval/input/human.glb). SIZE_REFERENCE will omit the human figure."
        )
        return None
    try:
        loaded = trimesh.load(str(path))
        if isinstance(loaded, trimesh.Scene):
            geoms = [
                g
                for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh) and not g.is_empty
            ]
            if not geoms:
                raise ValueError("human.glb: no mesh geometry")
            mesh = (
                trimesh.util.concatenate(geoms)
                if len(geoms) > 1
                else geoms[0]
            )
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise TypeError(f"Unexpected trimesh.load type: {type(loaded)}")
        _human_template_zup = _to_z_up(mesh)
        return _human_template_zup
    except Exception as e:
        _human_template_load_failed = True
        print(f"Warning: failed to load human.glb for SIZE_REFERENCE: {e}")
        return None


# Pyrender on-the-fly renders: align with SceneEval/configs/config.yaml (`resolution_*`, `camera_lens`).
# Override export size with SCENEEVAL_RENDER_RES (default 512). Sensor height matches Blender's typical
# 36×24mm gate when only lens is set in yaml.
SCENEEVAL_RENDER_RESOLUTION = int(os.environ.get("SCENEEVAL_RENDER_RES", "512"))
SCENEEVAL_CAMERA_LENS_MM = 35.0
SCENEEVAL_CAMERA_SENSOR_HEIGHT_MM = 24.0


def _sceneeval_perspective_yfov() -> float:
    """Blender perspective camera vertical FOV (radians) from lens + sensor height."""
    h = SCENEEVAL_CAMERA_SENSOR_HEIGHT_MM
    f = SCENEEVAL_CAMERA_LENS_MM
    return float(2.0 * np.arctan(h / (2.0 * f)))


def _rotation_x(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


def _size_reference_camera_pose(
    R_obj: np.ndarray,
    target: np.ndarray,
    comb_min: np.ndarray,
    comb_max: np.ndarray,
    yfov: float,
    bird_view_degree: float = 80.0,
) -> np.ndarray:
    """
    Rotation matches blender_scene.render_one_obj after:
        b_camera.matrix_world = b_obj.matrix_world
        b_camera.rotation_euler[0] = radians(bird_view_degree)
    i.e. R_cam = R_obj @ Rx(θ) with θ = bird_view_degree (0 = straight-down view, 90 = side; same
    convention as SceneEval docstring on render_one_obj).

    Blender then runs view3d.camera_to_view_selected(), which moves the camera along its view axis;
    that automatic framing is not available in pyrender, so we place the eye along +R_cam[:,2] at a
    distance that fits the combined object+human world AABB in the vertical FOV.
    """
    R_obj = np.asarray(R_obj, dtype=float)[:3, :3]
    target = np.asarray(target, dtype=float).reshape(3)
    theta = np.deg2rad(float(bird_view_degree))
    R_cam = R_obj @ _rotation_x(theta)
    cmin = np.asarray(comb_min, dtype=float).reshape(3)
    cmax = np.asarray(comb_max, dtype=float).reshape(3)
    radius = 0.5 * float(np.linalg.norm(cmax - cmin))
    margin = 1.12
    d = margin * max(radius, 1e-3) / max(np.tan(yfov * 0.5), 1e-6)
    eye = target + d * R_cam[:, 2]
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = R_cam
    pose[:3, 3] = eye
    return pose


# ─────────────────────────────────────────────────────────────────────────────

class NumpyMatrix:
    """
    Minimal stand-in for mathutils.Matrix.
    Wraps a numpy 4×4 array and exposes .to_3x3() so SceneEval metric code
    (opening_clearance, obj_obj_relationship) can call it unchanged.
    """

    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data, dtype=float)

    def to_3x3(self) -> "NumpyMatrix":
        return NumpyMatrix(self._data[:3, :3])

    def __array__(self, dtype=None):
        return self._data if dtype is None else self._data.astype(dtype)

    def __matmul__(self, other):
        return self._data @ np.asarray(other)

    def __repr__(self):
        return f"NumpyMatrix(\n{self._data}\n)"


# ─────────────────────────────────────────────────────────────────────────────

class SceneAdapter:
    """
    Provides the subset of SceneEval's Scene interface that the metrics we use
    actually call:

        t_objs, t_architecture
        get_obj_ids(), get_arch_ids()
        get_obj_bbox_center(), get_default_pose_obj_bbox_extents()
        get_obj_z_rotation(), get_obj_matrix(), get_arch_matrix()
        get_arch_bbox_center(), get_default_pose_arch_bbox_extents()
        get_default_pose_obj_bbox_center()
        obj_descriptions

    All geometry is converted to z-up on construction.

    Args:
        trimesh_meshes:    valid (non-empty) world-positioned meshes in y-up space,
                           in the same order as the other valid_* lists.
        bbox_params_valid: (n_valid, cls_dim+7) — rows from bbox_params_t for valid objects.
                           Layout: [...cls_logits, cx, cy, cz, sx, sy, sz, theta]
        class_ids_valid:   class index per valid object.
        obj_model_ids:     3D-FUTURE model_jid per valid object (may contain None).
        objects_types:     full list of class-type strings.
        wall_meshes:       [floor_mesh, (optional wall_mesh, ...)] in y-up space.
                           wall_meshes[0] is always the floor.
        cls_dim:           number of class-label dimensions in bbox_params.
        output_dir:        optional Path for metrics that save intermediate files.
    """

    def __init__(
        self,
        trimesh_meshes: list,
        bbox_params_valid: np.ndarray,
        class_ids_valid: list,
        obj_model_ids: list,
        objects_types: list,
        wall_meshes: list,
        cls_dim: int,
        output_dir: Optional[Path] = None,
        pth_to_models: Optional[str] = None,   # 3D-FUTURE model root dir
    ):
        self.output_dir = output_dir
        self._cls_dim = cls_dim
        self._objects_types = objects_types
        self._pth_to_models = pth_to_models
        self._model_id_map: dict[str, str] = {}  # obj_id → 3D-FUTURE model_jid

        # ── objects ──────────────────────────────────────────────────────────
        self.t_objs: dict[str, trimesh.Trimesh] = {}
        self.obj_descriptions: dict[str, str] = {}
        self.inverse_obj_descriptions: dict[str, str] = {}

        self._obj_ids_ordered: list[str] = []
        self._bbox_map: dict[str, np.ndarray] = {}   # obj_id → 1-D bbox row

        for k, (mesh, cls_id, model_id) in enumerate(
            zip(trimesh_meshes, class_ids_valid, obj_model_ids)
        ):
            mid = model_id if model_id is not None else f"unknown_{k}"
            obj_id = f"obj{k}_{mid}"

            self.t_objs[obj_id] = _to_z_up(mesh)
            self._obj_ids_ordered.append(obj_id)
            self._bbox_map[obj_id] = bbox_params_valid[k]

            cls_name = objects_types[cls_id] if cls_id < len(objects_types) else "unknown"
            # handle duplicate descriptions
            count = sum(1 for d in self.obj_descriptions.values() if d.startswith(cls_name))
            description = cls_name if count == 0 else f"{cls_name} - instance {count + 1}"
            self.obj_descriptions[obj_id] = description
            self.inverse_obj_descriptions[description] = obj_id
            self._model_id_map[obj_id] = mid

        # ── architecture (floor / walls) ──────────────────────────────────
        self.t_architecture: dict[str, trimesh.Trimesh] = {}

        for i, mesh in enumerate(wall_meshes):
            converted = _to_z_up(mesh)
            # After y→z transform, floor (originally thin in y) is thin in z.
            if converted.extents[2] < 0.05:
                self.t_architecture[f"floor_{i}"] = converted
            else:
                self.t_architecture[f"wall_{i}"] = converted

    # ── identity / iteration ─────────────────────────────────────────────────

    def get_obj_ids(self) -> list[str]:
        return self._obj_ids_ordered

    def get_arch_ids(self) -> list[str]:
        return list(self.t_architecture.keys())

    # ── object spatial queries (z-up) ────────────────────────────────────────

    def get_obj_bbox_center(self, obj_id: str) -> np.ndarray:
        """World-space bbox center in z-up: (cx, cz_old, cy_old)."""
        p = self._bbox_map[obj_id]
        cx, cy, cz = p[self._cls_dim: self._cls_dim + 3]
        return np.array([cx, cz, cy])

    def get_default_pose_obj_bbox_extents(self, obj_id: str) -> np.ndarray:
        """Full bbox extents in z-up: (sx*2, sz*2, sy*2)."""
        p = self._bbox_map[obj_id]
        sx, sy, sz = p[self._cls_dim + 3: self._cls_dim + 6]
        return np.array([sx * 2, sz * 2, sy * 2])

    def get_obj_z_rotation(self, obj_id: str) -> float:
        """Rotation around the vertical axis (z in SceneEval = y in 3D-FRONT)."""
        return float(self._bbox_map[obj_id][self._cls_dim + 6])

    def get_obj_matrix(self, obj_id: str) -> NumpyMatrix:
        """4×4 world transform in z-up space."""
        p = self._bbox_map[obj_id]
        cx, cy, cz = p[self._cls_dim: self._cls_dim + 3]
        theta = p[self._cls_dim + 6]

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        mat = np.eye(4)
        mat[:3, :3] = [
            [ cos_t, -sin_t, 0],
            [ sin_t,  cos_t, 0],
            [     0,      0, 1],
        ]
        mat[:3, 3] = [cx, cz, cy]   # z-up translation
        return NumpyMatrix(mat)

    def get_default_pose_obj_bbox_center(self, obj_id: str) -> np.ndarray:
        """Bbox center in the object's local (default) pose."""
        t_obj_default = self.get_default_pose_t_obj(obj_id)
        return t_obj_default.bounds[0] + t_obj_default.extents / 2

    def get_default_pose_t_obj(self, obj_id: str) -> trimesh.Trimesh:
        """Object trimesh in default (pre-transform) pose."""
        t_obj = self.t_objs[obj_id].copy()
        inv = np.linalg.inv(np.asarray(self.get_obj_matrix(obj_id)))
        t_obj.apply_transform(inv)
        return t_obj

    # ── architecture spatial queries ─────────────────────────────────────────

    def get_arch_bbox_center(self, arch_id: str) -> np.ndarray:
        mesh = self.t_architecture[arch_id]
        return mesh.bounds[0] + mesh.extents / 2

    def get_default_pose_arch_bbox_extents(self, arch_id: str) -> np.ndarray:
        return self.t_architecture[arch_id].extents

    def get_arch_matrix(self, arch_id: str) -> NumpyMatrix:
        """Identity rotation with translation to arch bbox center."""
        mat = np.eye(4)
        mat[:3, 3] = self.get_arch_bbox_center(arch_id)
        return NumpyMatrix(mat)

    def get_obj_render_path(self, obj_id: str, view: str = "FRONT") -> Path:
        """
        Get the path to the rendered image of the object.
        If the image doesn't exist, we render it on the fly using pyrender.
        """
        if self.output_dir:
            render_dir = self.output_dir / "renders"
        else:
            render_dir = Path("renders")
            
        render_dir.mkdir(parents=True, exist_ok=True)
        render_path = render_dir / f"{obj_id}_{view}.png"
        
        # If the image already exists, just return the path
        if render_path.exists():
            return render_path
            
        # Render the image on the fly using pyrender
        try:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            import pyrender
            from PIL import Image
            
            scene = pyrender.Scene(ambient_light=[0.7, 0.7, 0.7])
            yfov = _sceneeval_perspective_yfov()
            render_res = SCENEEVAL_RENDER_RESOLUTION

            t_obj = self.t_objs[obj_id]
            target = t_obj.centroid
            max_ext = np.max(t_obj.extents)
            if max_ext < 0.01: max_ext = 1.0
            
            def look_at(eye, target, up):
                z = eye - target
                z = z / (np.linalg.norm(z) + 1e-6)
                x = np.cross(up, z)
                x = x / (np.linalg.norm(x) + 1e-6)
                y = np.cross(z, x)
                y = y / (np.linalg.norm(y) + 1e-6)
                pose = np.eye(4)
                pose[:3, 0] = x
                pose[:3, 1] = y
                pose[:3, 2] = z
                pose[:3, 3] = eye
                return pose

            # Determine camera pose and which meshes to include
            if view == "FRONT":
                # Front view: Just the object itself, looking from front (-y in z-up)
                mesh = pyrender.Mesh.from_trimesh(t_obj)
                scene.add(mesh)
                
                # Get the object's local front direction
                obj_matrix = self.get_obj_matrix(obj_id).to_3x3()
                obj_front = np.asarray(obj_matrix) @ np.array([0, -1, 0])
                
                dist = max_ext * 1.5
                eye = target + obj_front * dist
                up = np.array([0, 0, 1])
                camera_pose = look_at(eye, target, up)
                
            elif view == "TOP":
                # Top view: Just the object itself, looking straight down
                mesh = pyrender.Mesh.from_trimesh(t_obj)
                scene.add(mesh)
                
                dist = max_ext * 1.5
                eye = target + np.array([0, 0, dist])
                up = np.array([0, 1, 0])
                camera_pose = look_at(eye, target, up)
                
            elif view == "SIZE_REFERENCE":
                # Match SceneEval/blender_scene.render_one_obj(..., with_human_reference=True, bird_view_degree=80)
                mesh = pyrender.Mesh.from_trimesh(t_obj)
                scene.add(mesh)

                M = np.asarray(self.get_obj_matrix(obj_id))
                R = M[:3, :3].copy()
                t = M[:3, 3].copy()

                human_mesh: Optional[trimesh.Trimesh] = None
                human_tpl = _load_human_template_zup()
                if human_tpl is not None:
                    bmin, bmax = t_obj.bounds
                    obj_extent_x = float(bmax[0] - bmin[0])

                    h_ori = human_tpl.copy()
                    h_ori.apply_transform(
                        np.vstack(
                            [
                                np.hstack([R, np.zeros((3, 1))]),
                                np.array([0.0, 0.0, 0.0, 1.0]),
                            ]
                        )
                    )
                    human_extent_x = float(h_ori.bounds[1, 0] - h_ori.bounds[0, 0])

                    dx = -(obj_extent_x * 0.5 + human_extent_x * 0.5)
                    v = R @ np.array([dx, 0.0, 0.0])
                    M_human = np.eye(4)
                    M_human[:3, :3] = R
                    M_human[:3, 3] = t + v
                    human_mesh = human_tpl.copy()
                    human_mesh.apply_transform(M_human)
                    scene.add(pyrender.Mesh.from_trimesh(human_mesh))

                if human_mesh is not None:
                    comb_min = np.minimum(t_obj.bounds[0], human_mesh.bounds[0])
                    comb_max = np.maximum(t_obj.bounds[1], human_mesh.bounds[1])
                else:
                    comb_min, comb_max = t_obj.bounds[0].copy(), t_obj.bounds[1].copy()
                target = (comb_min + comb_max) * 0.5
                # Same bird_view_degree as render_all_objs_front_size_reference() in blender_scene.py
                camera_pose = _size_reference_camera_pose(
                    R, target, comb_min, comb_max, yfov, bird_view_degree=80.0
                )
                
            else: # SURROUNDINGS
                # Surroundings: Object + nearby objects + architecture, but hide objects blocking the camera
                
                # 1. Determine camera pose (Zoomed out, isometric-like view)
                obj_matrix = self.get_obj_matrix(obj_id).to_3x3()
                obj_front = np.asarray(obj_matrix) @ np.array([0, -1, 0])
                obj_right = np.asarray(obj_matrix) @ np.array([1, 0, 0])
                
                dist = max_ext * 3.5
                eye = target + obj_front * dist + obj_right * (dist * 0.5) + np.array([0, 0, dist * 0.8])
                up = np.array([0, 0, 1])
                camera_pose = look_at(eye, target, up)
                
                # 2. Add the target object
                scene.add(pyrender.Mesh.from_trimesh(t_obj))
                
                # 3. Add other objects, filtering out those that block the camera
                for o_id, o_mesh in self.t_objs.items():
                    if o_id == obj_id:
                        continue
                        
                    # Simple raycast check: Does this object block the view from camera to target?
                    # We check if the object's bounding box intersects the line segment from eye to target
                    o_bbox = o_mesh.bounding_box
                    ray_dir = target - eye
                    ray_dist = np.linalg.norm(ray_dir)
                    ray_dir = ray_dir / ray_dist
                    
                    hit_points, _, _ = o_mesh.ray.intersects_location([eye], [ray_dir])
                    
                    is_blocking = False
                    for hit in hit_points:
                        hit_dist = np.linalg.norm(hit - eye)
                        if hit_dist < ray_dist - 0.1: # If hit is closer than target (with small margin)
                            is_blocking = True
                            break
                            
                    if not is_blocking:
                        scene.add(pyrender.Mesh.from_trimesh(o_mesh))
                
                # 4. Add architecture (floor/walls)
                for a_id, a_mesh in self.t_architecture.items():
                    # Don't add walls that might block the view
                    if "wall" in a_id:
                        hit_points, _, _ = a_mesh.ray.intersects_location([eye], [ray_dir])
                        is_blocking = False
                        for hit in hit_points:
                            hit_dist = np.linalg.norm(hit - eye)
                            if hit_dist < ray_dist - 0.1:
                                is_blocking = True
                                break
                        if is_blocking:
                            continue
                    
                    scene.add(pyrender.Mesh.from_trimesh(a_mesh))
            
            camera = pyrender.PerspectiveCamera(
                yfov=yfov, aspectRatio=1.0, znear=0.01, zfar=100.0
            )
            scene.add(camera, pose=camera_pose)
            
            # Add lighting
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=camera_pose)
            
            # Render (resolution matches SceneEval config.yaml blender.resolution_x/y unless overridden)
            r = pyrender.OffscreenRenderer(render_res, render_res)
            color, depth = r.render(scene)
            r.delete()
            
            # Save image
            img = Image.fromarray(color)
            img.save(render_path)
                
        except Exception as e:
            print(f"Warning: Failed to render {obj_id} on the fly: {e}")
            
        return render_path
