import os
import time

from copy import deepcopy
import shutil

import numpy as np
from PIL import Image

from cleanfid import fid
from src.data.threed_front_dataset_base import trs_to_corners
from src.data.utils_text import compute_loc_rel, reverse_rel
from src.data.threed_future_dataset import ThreedFutureDataset
from src.utils.visualize import *

from src.models import CLIPImageEncoder

from src.utils.collision_metric import get_collision_matrix

import pathlib
import trimesh

# Vendored SceneEval metrics — self-contained inside SceneNAT (no external
# ``compare_models`` repo dependency). See src/eval_metrics/.
from src.eval_metrics import (
    SceneAdapter,
    CollisionMetric, CollisionMetricConfig,
    NavigabilityMetric, NavigabilityMetricConfig,
    OutOfBoundMetric, OutOfBoundMetricConfig,
    AccessibilityMetric, AccessibilityMetricConfig,
    GPT, GPTConfig,
    PROMPTS_YAML,
)

class SceneEvaluator:
    """장면 평가를 담당하는 클래스"""
    def __init__(self, 
        raw_dataset, objects_types, predicate_types, max_rel_num, 
        text_encoder, 
        device, save_dir, 
        args, config,
        irecall=True, dfs=True, visualize=True
        ):
        self.raw_dataset = raw_dataset
        self.objects_types = objects_types
        self.predicate_types = predicate_types
        self.max_rel_num = max_rel_num
        self.device = device
        self.text_encoder = text_encoder
        self.img_encoder = CLIPImageEncoder(device = device)
        self.save_dir = save_dir
        self.args = args
        self.config = config

        #######################################
        self.irecall = irecall
        self.dfs = dfs
        self.visualize = visualize
         
        # Build the dataset of 3D models
        self.objects_dataset = ThreedFutureDataset.from_pickled_dataset(
            config["data"]["path_to_pickled_3d_futute_models"])
        print(f"Load [{len(self.objects_dataset)}] 3D-FUTURE models")
        self.pth_to_models = self.objects_dataset[0].path_to_models

        # Get real images to compute FID
        self.real_dir = os.path.join(self.raw_dataset._base_dir, "_test_blender_rendered_scene_256_topdown")

        # Initialize metrics
        self.rel_counts = 1e-9
        self.correct_rel_counts = 0
        self.correct_easy_rel_counts = 0
        # self.mean_dfs = []
        self.mean_dos = []
        self.mean_dos_fixed = []
        self.mean_dos_recall = []
        self.inference_times = []

        self.inter_vol_mean = []
        self.inter_vol_sum = []
        self.iomin = []

        # Vendored SceneEval / floor-penetration metric accumulators
        self.nav_scores = []
        self.nav_components = []
        self.se_collision_ratios = []
        self.se_oob_ratios = []
        self.se_accessibility = []
        self.floor_pen_ratios = []
        self.floor_float_ratios = []
        self.floor_invalid_ratios = []
        self.floor_pen_depths = []
        self.floor_float_heights = []

        self.num_trip_irecall = {i: [] for i in range(1, self.max_rel_num+1)}
        self.num_trip_irecall_easy = {i: [] for i in range(1, self.max_rel_num+1)}

        # Initialize epoch-specific metrics
        self.epoch_metrics = {
            "vol_avg": [],
            "vol_total": [],
            "iomin_avg": [],
            "relation_accs": [],
            "relation_accs_easy": [],
            "fid": [],
            "fid_clip": [],
            "kid": [],
            "time": [],
            "dos": [],
            "dos_fixed": [],
            "dos_recall": [],
            "floor_pen_ratio": [],
            "floor_float_ratio": [],
            "floor_invalid_ratio": [],
            "floor_pen_depth": [],
            "floor_float_height": [],
            "navigability": [],
            "nav_components": [],
            "se_collision_ratio": [],
            "se_oob_ratio": [],
            "se_accessibility": [],
        }
        self.transparent = False
        self.custom_floor = False
        self.custom_wall = False
        self.remove_scene_dir = True

    def save_epoch_metrics(self, epoch):
        """Save current metrics to epoch-specific storage"""
        self.epoch_metrics["vol_avg"].append(np.mean(self.inter_vol_mean))
        self.epoch_metrics["vol_total"].append(np.sum(self.inter_vol_sum))
        self.epoch_metrics["iomin_avg"].append(np.mean(self.iomin))
        self.inter_vol_mean = []
        self.inter_vol_sum = []
        self.iomin = []
        eval_info = ""

        eval_info += f"Collision vol mean: {self.epoch_metrics['vol_avg'][-1]}\n"
        eval_info += f"Collision vol sum: {self.epoch_metrics['vol_total'][-1]}\n"
        eval_info += f"Collision iomin: {self.epoch_metrics['iomin_avg'][-1]}\n"

        if self.irecall:    
            relation_accs = self.correct_rel_counts / self.rel_counts
            relation_accs_easy = self.correct_easy_rel_counts / self.rel_counts
            self.epoch_metrics["relation_accs"].append(relation_accs)
            self.epoch_metrics["relation_accs_easy"].append(relation_accs_easy)
            eval_info += f"Relation acc: [{self.correct_rel_counts:4d}/{int(self.rel_counts):4d}] = {relation_accs*100:.2f}%\n"
            eval_info += f"Relation acc (easy): [{self.correct_easy_rel_counts:4d}/{int(self.rel_counts):4d}] = {relation_accs_easy*100:.2f}%\n"
            self.rel_counts = 1e-9
            self.correct_rel_counts = 0
            self.correct_easy_rel_counts = 0

        if self.visualize:
            self.epoch_metrics["fid"].append(self.fid)
            self.epoch_metrics["fid_clip"].append(self.fid_clip)
            self.epoch_metrics["kid"].append(self.kid)
            eval_info += f"FID score: {self.fid:.2f}\n"
            eval_info += f"CLIP-FID score: {self.fid_clip:.2f}\n"
            eval_info += f"KID score: {self.kid*1000:.2f}\n"

        self.epoch_metrics["time"].append(np.mean(self.inference_times))
        self.inference_times = []

        if self.dfs:
            self.epoch_metrics["dos"].append(np.mean(self.mean_dos))
            self.epoch_metrics["dos_fixed"].append(np.mean(self.mean_dos_fixed))
            self.epoch_metrics["dos_recall"].append(np.mean(self.mean_dos_recall))
            self.mean_dos = []
            self.mean_dos_fixed = []
            self.mean_dos_recall = []
            eval_info += f"DOS: {self.epoch_metrics['dos'][-1]*1000:.2f}\n"
            eval_info += f"DOS_fixed: {self.epoch_metrics['dos_fixed'][-1]*1000:.2f}\n"
            eval_info += f"DOS_recall: {self.epoch_metrics['dos_recall'][-1]*1000:.2f}\n"
            # self.epoch_metrics["dfs"].append(np.mean(self.mean_dfs))
            # self.mean_dfs = []
            # eval_info += f"DFS: {np.mean(self.mean_dfs)*1000:.4f}\n"

        # Floor-penetration metrics (inline, no external deps)
        self.epoch_metrics["floor_pen_ratio"].append(np.mean(self.floor_pen_ratios) if self.floor_pen_ratios else 0.0)
        self.epoch_metrics["floor_float_ratio"].append(np.mean(self.floor_float_ratios) if self.floor_float_ratios else 0.0)
        self.epoch_metrics["floor_invalid_ratio"].append(np.mean(self.floor_invalid_ratios) if self.floor_invalid_ratios else 0.0)
        self.epoch_metrics["floor_pen_depth"].append(np.mean(self.floor_pen_depths) if self.floor_pen_depths else 0.0)
        self.epoch_metrics["floor_float_height"].append(np.mean(self.floor_float_heights) if self.floor_float_heights else 0.0)
        self.floor_pen_ratios = []
        self.floor_float_ratios = []
        self.floor_invalid_ratios = []
        self.floor_pen_depths = []
        self.floor_float_heights = []
        eval_info += (
            f"Floor Invalid Ratio: {self.epoch_metrics['floor_invalid_ratio'][-1]:.4f} "
            f"(Pen: {self.epoch_metrics['floor_pen_ratio'][-1]:.4f}, "
            f"Float: {self.epoch_metrics['floor_float_ratio'][-1]:.4f})\n"
        )
        eval_info += f"Floor Pen Depth: {self.epoch_metrics['floor_pen_depth'][-1]:.4f}\n"
        eval_info += f"Floor Float Height: {self.epoch_metrics['floor_float_height'][-1]:.4f}\n"

        # SceneEval metrics
        self.epoch_metrics["navigability"].append(np.mean(self.nav_scores) if self.nav_scores else 0.0)
        self.epoch_metrics["nav_components"].append(np.mean(self.nav_components) if self.nav_components else 0.0)
        self.epoch_metrics["se_collision_ratio"].append(np.mean(self.se_collision_ratios) if self.se_collision_ratios else 0.0)
        self.epoch_metrics["se_oob_ratio"].append(np.mean(self.se_oob_ratios) if self.se_oob_ratios else 0.0)
        self.epoch_metrics["se_accessibility"].append(np.mean(self.se_accessibility) if self.se_accessibility else 0.0)
        self.nav_scores = []
        self.nav_components = []
        self.se_collision_ratios = []
        self.se_oob_ratios = []
        self.se_accessibility = []
        eval_info += f"Navigability: {self.epoch_metrics['navigability'][-1]:.4f} (components: {self.epoch_metrics['nav_components'][-1]:.2f})\n"
        eval_info += f"SE Collision ratio: {self.epoch_metrics['se_collision_ratio'][-1]:.4f}\n"
        eval_info += f"SE OOB ratio: {self.epoch_metrics['se_oob_ratio'][-1]:.4f}\n"
        eval_info += f"SE Accessibility: {self.epoch_metrics['se_accessibility'][-1]:.4f}\n"

        # Save evaluation results
        with open(os.path.join(self.save_dir, f"eval_result_epoch_{epoch:03d}.txt"), "w") as f:
            f.write(eval_info)
        
    def render_scene(self, trimesh_meshes, bbox_meshes, export_dir, wall_meshes=None, step=None):
        """Render scene using Blender"""
        if len(trimesh_meshes) == 0:
            return
        
        if wall_meshes is None:
            # To get the manually created floor plan, which includes vertices of all meshes in the scene
            all_vertices = np.concatenate([
                tr_mesh.vertices for tr_mesh in trimesh_meshes
            ], axis=0)
            x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
            z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()

            if self.custom_floor:
                path_to_floor_plan_textures = "dataset/etc_texture_floor"
            else:
                path_to_floor_plan_textures = self.config["data"]["path_to_floor_plan_textures"]

            floor_textures = [os.path.join(path_to_floor_plan_textures, fi) 
                for fi in os.listdir(path_to_floor_plan_textures)]
            texture = np.random.choice(floor_textures)

            if self.custom_wall:
                path_to_wall_textures = "dataset/etc_texture_wall"
                wall_textures = [os.path.join(path_to_wall_textures, fi) 
                    for fi in os.listdir(path_to_wall_textures)]
                wall_texture = np.random.choice(wall_textures)
            else:
                wall_texture = None

            wall_meshes = floor_plan(self.raw_dataset[0], texture, room_size=[x_min, z_min, x_max, z_max], wall_texture=wall_texture)

        trimesh_meshes.extend(wall_meshes)
        
        # Create a trimesh scene and export it to a temporary directory
        tmp_dir = os.path.join(export_dir, f"tmp_{step:03d}" if step is not None else "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        export_scene(tmp_dir, trimesh_meshes, bbox_meshes)
        
        # Render the exported scene by calling blender
        blender_render_scene(
            tmp_dir,
            export_dir,
            top_down_view=(not self.args.eight_views),
            resolution_x=self.args.resolution,
            resolution_y=self.args.resolution,
            output_suffix=f"_step{step:02d}" if step is not None else "",
            transparent=self.transparent,
            remove_scene_dir=self.remove_scene_dir
        )

        return wall_meshes

    def evaluate_scene(self, 
            bbox_params_t, objfeats, 
            rels=None, #for iRecall
            descs=None, #for clip score
            texts=None,
            scene_id=None,
            all_steps=False,
            verbose=True,
            cls_dim= None):
        """단일 장면을 평가하는 함수"""
        metrics = {}
        
        # 성능 평가는 마지막 스텝으로만 수행
        if bbox_params_t.ndim == 3:  # (step_num, N, ...)
            bbox_params_t_eval = bbox_params_t[-1]  # 마지막 스텝만 사용
            objfeats_eval = objfeats[-1]
        else:  # (N, ...)
            bbox_params_t_eval = bbox_params_t
            objfeats_eval = objfeats
        
        # Create export directory
        export_dir = os.path.join(self.save_dir, scene_id)
        os.makedirs(export_dir, exist_ok=True)
        
        # Get the textured objects by retrieving the 3D models
        trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, obj_ids = get_textured_objects(
            bbox_params_t_eval,
            self.objects_dataset, self.objects_types,
            objfeats_eval,
            "openshape_vitg14",
            verbose=verbose
        )

        eval_info = ""

        if cls_dim is None:
            cls_dim = len(self.objects_types) + 1
        
        # Get object class IDs
        obj_class_ids = [self.objects_types.index(c) if c is not None else len(self.objects_types)
            for c in obj_classes]
        
        # iRecall evaluation
        if self.irecall:
            current_rel_counts, cur_correct_rel_counts, cur_correct_easy_rel_counts = self.evaluate_relations(
                obj_class_ids, bbox_params_t_eval, obj_sizes, rels, cls_dim= cls_dim
            )
            eval_info += f"triplet num: {int(current_rel_counts):d}\n"
            eval_info += f"Relation acc: [{cur_correct_rel_counts:d}/{int(current_rel_counts):d}] = {cur_correct_rel_counts/current_rel_counts:.4f}\n"
            eval_info += f"Relation acc (easy): [{cur_correct_easy_rel_counts:d}/{int(current_rel_counts):d}] = {cur_correct_easy_rel_counts/current_rel_counts:.4f}\n"
            metrics["current_rel_counts"] = current_rel_counts
            metrics["cur_correct_rel_counts"] = cur_correct_rel_counts
            metrics["cur_correct_easy_rel_counts"] = cur_correct_easy_rel_counts
        
        # Compute DFS scores
        if self.dfs:
            # max_sim_scores = self.compute_dfs(
            dos_metrics = self.compute_dos_metrics(
                obj_class_ids, obj_ids, descs,
                export_dir
            )
        
            # Calculate mean score if available
            # mean_dfs = np.mean(max_sim_scores) if len(max_sim_scores) > 0 else 0.0
            mean_dos = np.mean(dos_metrics["DOS"])
            mean_dos_fixed = np.mean(dos_metrics["DOS_fixed"])
            mean_dos_recall = np.mean(dos_metrics["DOS_recall"])
            eval_info += f"DOS: {mean_dos*1000:.2f}\n"
            eval_info += f"DOS_fixed: {mean_dos_fixed*1000:.2f}\n"
            eval_info += f"DOS_recall: {mean_dos_recall*1000:.2f}\n"
            metrics["mean_dos"] = mean_dos
            metrics["mean_dos_fixed"] = mean_dos_fixed
            metrics["mean_dos_recall"] = mean_dos_recall

            # mean_dfs = np.mean(max_sim_scores)
            # eval_info += f"DFS: {mean_dfs*1000:.2f}"
            # metrics["mean_dfs"] = mean_dfs
        
        import matplotlib.pyplot as plt
        from shapely.geometry import box
        from shapely import affinity

        valid_indices = [i for i in range(len(obj_class_ids)) if obj_class_ids[i] != len(self.objects_types)]

        scene_pen_count = 0
        scene_float_count = 0
        scene_pen_depth_sum = 0.0
        scene_float_height_sum = 0.0

        bbox_data = []
        floor_status = []

        for idx in valid_indices:
            cx, cy, cz = bbox_params_t_eval[idx, cls_dim:cls_dim + 3]
            sx, sy, sz = bbox_params_t_eval[idx, cls_dim + 3:cls_dim + 6]
            theta = bbox_params_t_eval[idx, cls_dim + 6]
            bbox_data.append([cx, cy, cz, sx, sy, sz, theta])

            bottom_y = cy - sy
            eps = 0.01

            if abs(bottom_y) <= 0.2:
                if bottom_y < -eps:
                    scene_pen_count += 1
                    scene_pen_depth_sum += abs(bottom_y)
                    floor_status.append("red")
                elif bottom_y > eps:
                    scene_float_count += 1
                    scene_float_height_sum += bottom_y
                    floor_status.append("blue")
                else:
                    floor_status.append("green")
            else:
                floor_status.append("gray")

        total_objs = len(valid_indices)
        if total_objs > 0:
            metrics["floor_pen_ratio"] = scene_pen_count / total_objs
            metrics["floor_float_ratio"] = scene_float_count / total_objs
            metrics["floor_invalid_ratio"] = (scene_pen_count + scene_float_count) / total_objs
        else:
            metrics["floor_pen_ratio"] = 0.0
            metrics["floor_float_ratio"] = 0.0
            metrics["floor_invalid_ratio"] = 0.0

        metrics["floor_pen_depth"] = scene_pen_depth_sum / scene_pen_count if scene_pen_count > 0 else 0.0
        metrics["floor_float_height"] = scene_float_height_sum / scene_float_count if scene_float_count > 0 else 0.0

        eval_info += (
            f"Floor Invalid Ratio: {metrics['floor_invalid_ratio']:.4f} "
            f"(Pen: {metrics['floor_pen_ratio']:.4f}, Float: {metrics['floor_float_ratio']:.4f})\n"
        )
        eval_info += f"Floor Pen Depth: {metrics['floor_pen_depth']:.4f}\n"
        eval_info += f"Floor Float Height: {metrics['floor_float_height']:.4f}\n"

        bbox_data = np.array(bbox_data)

        if len(bbox_data) > 0:
            fig_floor, ax_floor = plt.subplots(figsize=(10, 10))
            ax_floor.set_facecolor("#f9f9f9")

            for i in range(len(bbox_data)):
                cx, _, cz, sx, _, sz, theta = bbox_data[i]
                poly = box(-sx, -sz, sx, sz)
                poly_rot = affinity.rotate(poly, theta, origin=(0, 0), use_radians=True)
                poly_final = affinity.translate(poly_rot, xoff=cx, yoff=-cz)

                color = floor_status[i]
                alpha = 0.6 if color != "gray" else 0.3
                ls = "-" if color != "gray" else "--"
                lw = 2 if color != "gray" else 1

                x, y = poly_final.exterior.xy
                ax_floor.plot(x, y, color=color, linewidth=lw, linestyle=ls)
                ax_floor.fill(x, y, color=color, alpha=alpha)
                ax_floor.text(cx, -cz, str(i), fontsize=9, ha="center", va="center", color="black", fontweight="bold")

            ax_floor.set_xlim(-4, 4)
            ax_floor.set_ylim(-4, 4)
            ax_floor.set_aspect("equal")
            ax_floor.set_xlabel("X axis")
            ax_floor.set_ylabel("Z axis")
            ax_floor.set_title(f"Floor Validity Map (Pen: {scene_pen_count}, Float: {scene_float_count})")
            ax_floor.grid(True, linestyle=":", alpha=0.5)
            floor_map_path = os.path.join(export_dir, "floor_validity_map.png")
            fig_floor.savefig(floor_map_path, dpi=300, bbox_inches="tight")
            plt.close(fig_floor)

        if len(trimesh_meshes) > 0:
            all_verts = np.concatenate([mesh.vertices for mesh in trimesh_meshes], axis=0)
            x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
            z_min, z_max = all_verts[:, 2].min(), all_verts[:, 2].max()
            floor_corners = np.array(
                [
                    [x_min, 0.0, z_min],
                    [x_min, 0.0, z_max],
                    [x_max, 0.0, z_max],
                    [x_max, 0.0, z_min],
                ]
            )
            se_floor = trimesh.Trimesh(vertices=floor_corners, faces=np.array([[0, 1, 2], [0, 2, 3]]))

            valid_meshes = trimesh_meshes
            valid_bbox = bbox_params_t_eval[valid_indices]
            valid_cls_ids = [obj_class_ids[i] for i in valid_indices]
            valid_model_ids = [obj_ids[i] for i in valid_indices]

            se_scene = SceneAdapter(
                trimesh_meshes=valid_meshes,
                bbox_params_valid=valid_bbox,
                class_ids_valid=valid_cls_ids,
                obj_model_ids=valid_model_ids,
                objects_types=self.objects_types,
                wall_meshes=[se_floor],
                cls_dim=cls_dim,
                output_dir=pathlib.Path(export_dir),
            )

            se_col = CollisionMetric(se_scene, CollisionMetricConfig()).run()
            n_objs = len(se_scene.get_obj_ids())
            se_col_ratio = se_col.data["num_obj_in_collision"] / n_objs if n_objs > 0 else 0.0
            metrics["se_collision_ratio"] = se_col_ratio
            eval_info += f"SE Collision ratio: {se_col_ratio:.4f}\n"

            se_nav = NavigabilityMetric(
                se_scene,
                pathlib.Path(export_dir),
                NavigabilityMetricConfig(),
            ).run()
            metrics["navigability"] = se_nav.data["navigability"]
            metrics["nav_components"] = se_nav.data["connected_components"]
            eval_info += (
                f"Navigability: {se_nav.data['navigability']:.4f} "
                f"(components: {se_nav.data['connected_components']})\n"
            )

            se_oob = OutOfBoundMetric(se_scene, OutOfBoundMetricConfig()).run()
            n_oob = sum(1 for s in se_oob.data.values() if s["out_of_bound"])
            se_oob_ratio = n_oob / n_objs if n_objs > 0 else 0.0
            metrics["se_oob_ratio"] = se_oob_ratio
            eval_info += f"SE OOB ratio: {se_oob_ratio:.4f}\n"

            try:
                prompt_file = PROMPTS_YAML
                vlm = GPT(GPTConfig(prompt_file=str(prompt_file)))
                se_acc = AccessibilityMetric(
                    se_scene,
                    vlm,
                    pathlib.Path(export_dir),
                    AccessibilityMetricConfig(
                        image_resolution=256,
                        scale_margin=0.2,
                        obj_height_threshold=2.0,
                        access_area_width=0.6,
                        access_area_offset=0.1,
                    ),
                ).run()
                acc_scores = [score["max"] for score in se_acc.data.values() if score["max"] >= 0]
                metrics["se_accessibility"] = np.mean(acc_scores) if acc_scores else 0.0
                eval_info += f"SE Accessibility: {metrics['se_accessibility']:.4f}\n"
            except Exception as e:
                print(f"Failed to run VLM metrics: {e}")
                metrics["se_accessibility"] = 0.0

        inter_vol, inter_vol_mean, inter_vol_sum, iomin_score, _ = get_collision_matrix(bbox_data, with_fig=True)

        metrics["inter_mean"] = inter_vol_mean
        metrics["inter_sum"] = inter_vol_sum
        metrics["iomin"] = iomin_score
        eval_info += f"Collision vol mean: {inter_vol_mean:.4f}\n"
        eval_info += f"Collision vol sum: {inter_vol_sum:.4f}\n"
        eval_info += f"Collision iomin: {iomin_score:.4f}\n"

         # Create collision boolean matrix (excluding diagonal)
        collision_bool = inter_vol > 0
        np.fill_diagonal(collision_bool, False)
        
        if np.any(collision_bool):
            pairs = np.argwhere(collision_bool)
            for i, j in pairs:
                if i < j:
                    # Map back to original indices
                    obj_i, obj_j = self.objects_types[obj_class_ids[valid_indices[i]]], self.objects_types[obj_class_ids[valid_indices[j]]]
                    eval_info += f"Obj {obj_i} <-> Obj {obj_j} (vol: {inter_vol[i,j]:.4f})\n"
        
        # # Save collision image
        # if fig is not None:
        #     import matplotlib.pyplot as plt
        #     collision_img_path = os.path.join(export_dir, "collision_map.png")
        #     plt.savefig(collision_img_path, dpi=300, bbox_inches='tight')
        #     plt.close(fig)  # 메모리 해제
        
        # Save conditioned text
        if texts is not None:
            with open(os.path.join(export_dir, "description.txt"), "w") as f:
                f.write(texts)
        
        with open(os.path.join(export_dir, "metrics.txt"), "w") as f:
            f.write(eval_info)
        
        with open(os.path.join(export_dir, "objs.txt"), "w") as f:
            f.write("\n".join(str(obj) if obj is not None else "NULL" for obj in obj_ids))
        
        # Render scene if requested
        if self.visualize:
            self.wall_meshes = self.render_scene(trimesh_meshes, bbox_meshes, export_dir)

            if all_steps and bbox_params_t.ndim == 3:
                # 모든 타임스텝에 대해 렌더링
                for step in range(bbox_params_t.shape[0]):
                    # Get the textured objects for this step
                    step_trimesh_meshes, step_bbox_meshes, _, _, _ = get_textured_objects(
                        bbox_params_t[step],
                        self.objects_dataset, self.objects_types,
                        objfeats[step],
                        "openshape_vitg14",
                        verbose=verbose
                    )
                    
                    self.render_scene(step_trimesh_meshes, step_bbox_meshes, export_dir, self.wall_meshes, step)
        
        return metrics

    def update_metrics(self, scene_results):
        """Update evaluation metrics with scene results"""

        self.inter_vol_mean.append(scene_results["inter_mean"])
        self.inter_vol_sum.append(scene_results["inter_sum"])
        self.iomin.append(scene_results["iomin"])

        # SceneEval metrics (only present when the scene produced meshes)
        if "navigability" in scene_results:
            self.nav_scores.append(scene_results["navigability"])
            self.nav_components.append(scene_results["nav_components"])
            self.se_collision_ratios.append(scene_results["se_collision_ratio"])
            self.se_oob_ratios.append(scene_results.get("se_oob_ratio", 0.0))
            self.se_accessibility.append(scene_results.get("se_accessibility", 0.0))

        # Floor-penetration metrics
        self.floor_pen_ratios.append(scene_results["floor_pen_ratio"])
        self.floor_float_ratios.append(scene_results["floor_float_ratio"])
        self.floor_invalid_ratios.append(scene_results["floor_invalid_ratio"])
        if scene_results["floor_pen_depth"] > 0:
            self.floor_pen_depths.append(scene_results["floor_pen_depth"])
        if scene_results["floor_float_height"] > 0:
            self.floor_float_heights.append(scene_results["floor_float_height"])

        if self.dfs:
            self.mean_dos.append(scene_results["mean_dos"])
            self.mean_dos_fixed.append(scene_results["mean_dos_fixed"])
            self.mean_dos_recall.append(scene_results["mean_dos_recall"])
            # self.mean_dfs.append(scene_results["mean_dfs"])
        
        if self.irecall:
            self.rel_counts += scene_results["current_rel_counts"]
            self.correct_rel_counts += scene_results["cur_correct_rel_counts"]
            self.correct_easy_rel_counts += scene_results["cur_correct_easy_rel_counts"]
            
            self.num_trip_irecall[int(scene_results["current_rel_counts"])].append(
                scene_results["cur_correct_rel_counts"]/scene_results["current_rel_counts"]
            )
            self.num_trip_irecall_easy[int(scene_results["current_rel_counts"])].append(
                scene_results["cur_correct_easy_rel_counts"]/scene_results["current_rel_counts"]
            )
        
            return {
                "rel": scene_results["cur_correct_rel_counts"] / scene_results["current_rel_counts"],
                "erel": scene_results["cur_correct_easy_rel_counts"] / scene_results["current_rel_counts"]
            }
    
        else:
            return
        
    def reset_metrics(self):
        """Reset evaluation metrics"""
        self.rel_counts = 1e-9
        self.correct_rel_counts = 0
        self.correct_easy_rel_counts = 0
        self.num_trip_irecall = {i: [] for i in range(1, self.max_rel_num+1)}
        self.num_trip_irecall_easy = {i: [] for i in range(1, self.max_rel_num+1)}

    def evaluate_relations(self, obj_class_ids, bbox_params_t, obj_sizes, rels, cls_dim= None):
        """Evaluate spatial relations between objects"""
        relations = []  # [[cls_id1, pred_id, cls_id2], ...]

        # Find all relations between objects
        for idx in range(len(obj_class_ids)):
            if obj_class_ids[idx] == len(self.objects_types):  # empty object
                continue
            c1_id = obj_class_ids[idx]
            t1 = bbox_params_t[idx, cls_dim:cls_dim+3]
            r1 = bbox_params_t[idx, cls_dim+6]
            s1 = obj_sizes[idx]
            corners1 = trs_to_corners(t1, r1, s1)
            name1 = self.objects_types[c1_id]
            
            for other_idx in range(idx+1, len(obj_class_ids)):
                if obj_class_ids[other_idx] == len(self.objects_types):  # empty object
                    continue 
                c2_id = obj_class_ids[other_idx]
                t2 = bbox_params_t[other_idx, cls_dim:cls_dim+3]
                r2 = bbox_params_t[other_idx, cls_dim+6]
                s2 = obj_sizes[other_idx]
                corners2 = trs_to_corners(t2, r2, s2)
                name2 = self.objects_types[c2_id]
                
                loc_rel_str = compute_loc_rel(corners1, corners2, name1, name2)
                if loc_rel_str is not None:
                    relation_id = self.predicate_types.index(loc_rel_str)
                    relations.append((int(obj_class_ids[idx]), int(relation_id), int(obj_class_ids[other_idx])))
                    # Add the reverse relation
                    rev_relation_id = self.predicate_types.index(reverse_rel(loc_rel_str))
                    relations.append((int(obj_class_ids[other_idx]), int(rev_relation_id), int(obj_class_ids[idx])))

        # Compare with ground truth
        relations_copy = deepcopy(relations)
        current_rel_counts, cur_correct_rel_counts, cur_correct_easy_rel_counts = 1e-9, 0, 0
        
        for rel in rels:
            current_rel_counts += 1  # ground truth
            if rel in relations:
                cur_correct_rel_counts += 1
                relations.remove(rel)
            
            # Ease the evaluation by ignoring `closely`
            if "closely" in self.predicate_types[rel[1]]:
                easy_rel = (rel[0], rel[1]-2, rel[2])
            elif self.predicate_types[rel[1]] not in ["above", "below"]:
                easy_rel = (rel[0], rel[1]+2, rel[2])
            else:
                easy_rel = rel
            if rel in relations_copy:
                cur_correct_easy_rel_counts += 1
                relations_copy.remove(rel)
            elif easy_rel in relations_copy:
                cur_correct_easy_rel_counts += 1
                relations_copy.remove(easy_rel)
                
        return current_rel_counts, cur_correct_rel_counts, cur_correct_easy_rel_counts

    def compute_dfs(self, obj_class_ids, obj_ids, descs, export_dir):
        """Compute DFS similarity scores between object images and descriptions"""
        max_sim_scores = []
        
        for j, (desc, obj_class_id) in enumerate(descs):
            obj_class_str = self.objects_types[obj_class_id]
            
            # Generate text embeddings
            _, descs_f = self.text_encoder([desc, obj_class_str])
            desc_f = descs_f[0].cpu().numpy()
            obj_class_f = descs_f[1].cpu().numpy()
            
            # Save descriptions
            with open(os.path.join(export_dir, f"desc_{2*j}.txt"), "w") as f:
                f.write(desc)
            
            # Find best matching images
            max_sim = 0
            for obj_idx, obj in enumerate(obj_class_ids):
                if obj == obj_class_id:
                    obj_img_path = os.path.join(self.pth_to_models, obj_ids[obj_idx], "image.jpg")
                    obj_img = Image.open(obj_img_path)
                    obj_img_f = self.img_encoder(obj_img).squeeze(0).cpu().numpy()
                    
                    cos_sim = np.dot(obj_img_f, desc_f)
                    cos_sim_offset = np.dot(obj_img_f, obj_class_f)
                    cos_sim = cos_sim - cos_sim_offset
                    if cos_sim > max_sim:
                        obj_img.save(os.path.join(export_dir, f"max_img_{2*j}.jpg"))
                        max_sim = cos_sim
                        
            max_sim_scores.append(max_sim)
            
        return max_sim_scores

    def compute_dos_metrics(self, obj_class_ids, obj_ids, descs, export_dir, penalty_value=-2.0):
        """Compute DOS (Description-Object Similarity) variants:
        - DOS: clipped at 0
        - DOS_fixed: penalty for missing object
        - DOS_recall: 0 for missing, otherwise raw
        """
        dos_scores = []
        dos_fixed_scores = []
        dos_recall_scores = []

        for j, (desc, obj_class_id) in enumerate(descs):
            obj_class_str = self.objects_types[obj_class_id]

            # Generate text and class embeddings
            _, descs_f = self.text_encoder([desc, obj_class_str])
            desc_f = descs_f[0].cpu().numpy()
            obj_class_f = descs_f[1].cpu().numpy()

            # Save description
            with open(os.path.join(export_dir, f"desc_{2*j}.txt"), "w") as f:
                f.write(desc)

            # Search over matching category images
            max_sim = float("-inf")
            matched = False
            for obj_idx, obj in enumerate(obj_class_ids):
                if obj == obj_class_id:
                    matched = True
                    obj_img_path = os.path.join(self.pth_to_models, obj_ids[obj_idx], "image.jpg")
                    obj_img = Image.open(obj_img_path)
                    obj_img_f = self.img_encoder(obj_img).squeeze(0).cpu().numpy()

                    cos_sim = np.dot(obj_img_f, desc_f)
                    cos_sim_offset = np.dot(obj_img_f, obj_class_f)
                    cos_sim -= cos_sim_offset

                    if cos_sim > max_sim:
                        obj_img.save(os.path.join(export_dir, f"max_img_{2*j}.jpg"))
                        max_sim = cos_sim

            # Aggregate scores for each variant
            dos_scores.append(max(0, max_sim) if matched else 0)
            dos_fixed_scores.append(max_sim if matched else penalty_value)
            dos_recall_scores.append(max_sim if matched else 0)

        return {
            "DOS": dos_scores,
            "DOS_fixed": dos_fixed_scores,
            "DOS_recall": dos_recall_scores
        }

    def eval_rendered_images(self, epoch):
        """렌더링된 이미지를 수집하고 평가하는 함수"""
        # gather all images in this epoch
        epoch_start_idx = epoch * len(self.raw_dataset)
        epoch_end_idx = (epoch + 1) * len(self.raw_dataset) - 1
        
        syn_dir = os.path.join(self.save_dir, f"all_syns_{epoch:02d}")
        os.makedirs(syn_dir, exist_ok=True)
        
        syn_images = []
        for scene_id in os.listdir(self.save_dir):
            try:
                scene_num = int(scene_id.split("@")[0])
                if epoch_start_idx <= scene_num <= epoch_end_idx:
                    topdown_path = os.path.join(self.save_dir, scene_id, "topdown.png")
                    if os.path.exists(topdown_path):
                        syn_images.append(topdown_path)
            except ValueError:
                continue
        
        for path in syn_images:
            name = os.path.basename(os.path.dirname(path)) + "_topdown.png"
            shutil.copyfile(path, os.path.join(syn_dir, name))
        
        num_syn_images = len(syn_images)
        print(f"Found [{num_syn_images}] synthesized images for epoch {epoch}\n\n")
        
        configs = {"fdir1": self.real_dir,
                    "fdir2": syn_dir,
                    "device": self.device}
        
        self.fid = fid.compute_fid(verbose=False, **configs)
        self.fid_clip = fid.compute_fid(model_name="clip_vit_b_32", verbose=False, **configs)
        self.kid = fid.compute_kid(verbose=False, **configs)

    def save_final_statistics(self):
        """평가 결과의 통계를 계산하고 저장하는 함수"""
        # Calculate statistics
        stat_dict = {}
        for key in self.epoch_metrics.keys(): # inference_times, relation_accs, relation_accs_easy, mean_scores, fids, fid_clips, kids
            stat_dict[key] = np.mean(self.epoch_metrics[key])
            stat_dict[f"{key}_std"] = np.std(self.epoch_metrics[key])
        
        num_trip_irecall_mean = {k: np.mean(v) if v else 0 for k, v in self.num_trip_irecall.items()}
        num_trip_irecall_std = {k: np.std(v) if v else 0 for k, v in self.num_trip_irecall.items()}
        num_trip_irecall_easy_mean = {k: np.mean(v) if v else 0 for k, v in self.num_trip_irecall_easy.items()}
        num_trip_irecall_easy_std = {k: np.std(v) if v else 0 for k, v in self.num_trip_irecall_easy.items()}
        
        # Save statistics
        stat_file = os.path.join(self.save_dir, f"stat.txt")
        with open(stat_file, "w") as f:
            f.write(f"Inference Time (std): {stat_dict['time']:.2f} ({stat_dict['time_std']:.2f}) sec\n")
            f.write(f"Relation Accuracy (std): {stat_dict['relation_accs']*100:.2f} ({stat_dict['relation_accs_std']*100:.2f})\n")
            f.write(f"Relation Accuracy (Easy) (std): {stat_dict['relation_accs_easy']*100:.2f} ({stat_dict['relation_accs_easy_std']*100:.2f})\n")
            
            f.write(f"Collision vol mean (std): {stat_dict['vol_avg']:.4f} ({stat_dict['vol_avg_std']:.4f})\n")
            f.write(f"Collision vol total (std): {stat_dict['vol_total']:.4f} ({stat_dict['vol_total_std']:.4f})\n")
            f.write(f"Collision iomin (std): {stat_dict['iomin_avg']:.4f} ({stat_dict['iomin_avg_std']:.4f})\n")
            
            f.write(f"FID score (std): {stat_dict['fid']:.4f} ({stat_dict['fid_std']:.4f})\n")
            f.write(f"FID_CLIP score (std): {stat_dict['fid_clip']:.4f} ({stat_dict['fid_clip_std']:.4f})\n")
            f.write(f"KID score (std): {stat_dict['kid']*1000:.4f} ({stat_dict['kid_std']*1000:.4f})\n")
            # f.write(f"DFS (std): {stat_dict['dfs']*1000:.2f} ({stat_dict['dfs_std']*1000:.2f})\n")
            f.write(f"DOS (std): {stat_dict['dos']*1000:.2f} ({stat_dict['dos_std']*1000:.2f})\n")
            f.write(f"DOS_fixed (std): {stat_dict['dos_fixed']*1000:.2f} ({stat_dict['dos_fixed_std']*1000:.2f})\n")
            f.write(f"DOS_recall (std): {stat_dict['dos_recall']*1000:.2f} ({stat_dict['dos_recall_std']*1000:.2f})\n")

            f.write(f"Floor Invalid Ratio (std): {stat_dict['floor_invalid_ratio']:.4f} ({stat_dict['floor_invalid_ratio_std']:.4f})\n")
            f.write(f"Floor Pen Ratio (std): {stat_dict['floor_pen_ratio']:.4f} ({stat_dict['floor_pen_ratio_std']:.4f})\n")
            f.write(f"Floor Float Ratio (std): {stat_dict['floor_float_ratio']:.4f} ({stat_dict['floor_float_ratio_std']:.4f})\n")
            f.write(f"Floor Pen Depth (std): {stat_dict['floor_pen_depth']:.4f} ({stat_dict['floor_pen_depth_std']:.4f})\n")
            f.write(f"Floor Float Height (std): {stat_dict['floor_float_height']:.4f} ({stat_dict['floor_float_height_std']:.4f})\n")
            f.write(f"Navigability (std): {stat_dict['navigability']:.4f} ({stat_dict['navigability_std']:.4f})\n")
            f.write(f"Nav Components (std): {stat_dict['nav_components']:.2f} ({stat_dict['nav_components_std']:.2f})\n")
            f.write(f"SE Collision Ratio (std): {stat_dict['se_collision_ratio']:.4f} ({stat_dict['se_collision_ratio_std']:.4f})\n")
            f.write(f"SE OOB Ratio (std): {stat_dict['se_oob_ratio']:.4f} ({stat_dict['se_oob_ratio_std']:.4f})\n")
            f.write(f"SE Accessibility (std): {stat_dict['se_accessibility']:.4f} ({stat_dict['se_accessibility_std']:.4f})\n")
            
            # Write num_trip_irecall statistics
            f.write("\nTriplet-wise Relation Accuracy (std):\n")
            for k in range(1, self.max_rel_num+1):
                f.write(f"  Triplet Num {k}: {num_trip_irecall_mean[k]:.4f} ({num_trip_irecall_std[k]:.4f})\n")
            
            # Write num_trip_irecall_easy statistics
            f.write("\nTriplet-wise Relation Accuracy (Easy) (std):\n")
            for k in range(1, self.max_rel_num+1):
                f.write(f"  Triplet Num {k}: {num_trip_irecall_easy_mean[k]:.4f} ({num_trip_irecall_easy_std[k]:.4f})\n")
