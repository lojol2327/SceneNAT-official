import argparse
import random
import os
import time
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import shutil
from cleanfid import fid
import matplotlib.pyplot as plt

from src.data import get_dataset_raw_and_encoded, filter_function
from src.data.utils_text import TextPreprocessor

from src.utils import *

from src.models import ObjectFeatureVQVAE, CLIPTextEncoder
from src.models.utils import process_model_output
from .utils import load_model

parser = argparse.ArgumentParser(description="SceneNAT stylization")
parser.add_argument("--config_file", type=str, default=None, help="Path to the file that contains the experiment configuration")
parser.add_argument("--tag", type=str, default=None, help="Tag that refers to the current experiment")
parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
parser.add_argument("--checkpoint_epoch", type=int, default=None, help="The epoch to load the checkpoint from")
parser.add_argument("--use_best", action="store_true", help="Use best epoch")
parser.add_argument("--num_workers", type=int, default=1, help="The epoch to load the checkpoint from")
parser.add_argument("--visualize", action="store_true", help="Visualize the generated scenes")
parser.add_argument("--eight_views", action="store_true", help="Render 8 views of the scene")
parser.add_argument("--resolution", type=int, default=256,help="Resolution of the rendered image")
parser.add_argument("--visualize_original", action="store_true", help="Visualize the partial scenes with masking")
parser.add_argument("--timesteps", type=int, default=10, help="The number of timesteps for evaluation")
parser.add_argument("--all_steps", action="store_true", help="Save rendering of each iteration result")
parser.add_argument("--vis_mask", action="store_true", help="Visualize the token mask")
parser.add_argument("--temperature", type=float, default=None, help="The temperature for evaluation")
parser.add_argument("--model_version", type=str, default="0709", help="Model version (e.g., 0709, 0903v1)")
parser.add_argument("--result_dir", type=str, default="downstream", help="Result directory")
parser.add_argument("--exp_name", type=str, default="stylization", help="Experiment name")

args = parser.parse_args()

seed = 250130
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"Available GPUs: {torch.cuda.device_count()}, Run code on device [{device}]\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.tag is not None:
        exp_dir = os.path.join(args.output_dir, args.tag)
    else:
        exp_dir = args.output_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    assert os.path.exists(ckpt_dir), f"Checkpoint directory {ckpt_dir} does not exist"
    
    if args.config_file is None:
        # Find json file in exp_dir
        json_files = [f for f in os.listdir(exp_dir) if f.endswith('.json')]
        assert len(json_files) > 0, f"No json file found in {exp_dir}"
        args.config_file = os.path.join(exp_dir, json_files[0])
        print(f"Using config file: {args.config_file}")
    
    config: Dict[str, Dict[str, Any]] = load_config(args.config_file)
    
    # region) Prepare dataset, dataloader
    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        config["data"]["path_to_pickled_3d_futute_models"])
    print(f"Load [{len(objects_dataset)}] 3D-FUTURE models")
    
    config["data"]["encoding_type"] += config["data"]["encoding_type"]+ "_eval"
    filter_function(config["data"], split=config["validation"].get("splits", ["test"]))
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        augmentations=None,
        split=config["validation"].get("splits", ["test"]))
    print(f"Load [{len(dataset)}] validation scenes with [{dataset.n_object_types}] object types\n")
    
    B = config["validation"]["batch_size"]

    dataloader = DataLoader(
        dataset,
        batch_size=B,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )
    
    # Get real images to compute FID
    print(f"Collect real images from [{raw_dataset._base_dir}]")
    real_dir = os.path.join(raw_dataset._base_dir, "_test_blender_rendered_scene_256_topdown")
    num_real_images = len(os.listdir(real_dir))
    print(f"Found [{num_real_images}] real images\n")
    
    # endregion
    
    text_encoder = CLIPTextEncoder(config["model"]["text_encoder"], device=device)
    
    # region) Load models
    # 1. Load pretrained VQ-VAE codebook weights
    print("Load pretrained VQ-VAE ...\n")
    with open("output/vqvae_openshape/objfeat_bounds.pkl", "rb") as f:
        kwargs = pickle.load(f)
    vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
    ckpt_path = f"output/vqvae_openshape/epoch_01999.pth"
    vqvae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()
    
    # 2. Initialize the model
    print("Load model ...\n")
    SceneNAT_model = load_model(args.model_version, None)
    # SceneNAT_model = load_model(args.model_version, exp_dir)
    if SceneNAT_model is None:
        raise RuntimeError(f"Failed to load model class for version {args.model_version}")

    model = SceneNAT_model(
        dataset.n_object_types,
        dataset.n_predicate_types,
        objfeat_dim=64,
        text_dim=512,
        max_obj_num=dataset.max_length,
        bounds=dataset.bounds,
        t_disc_dim=dataset.t_disc_dim,
        r_disc_dim=dataset.r_disc_dim,
        s_disc_dim=dataset.s_disc_dim,
        **config["model"]["transformer_config"]
        ).to(device)
    
    # Load checkpoint
    load_epoch = load_checkpoints_for_test(
        model, ckpt_dir,
        epoch=args.checkpoint_epoch,
        get_last=not args.use_best,
        get_best=args.use_best, 
        device=device)
    model.eval()
    # endregion
    
    # save_dir = os.path.join(exp_dir, "generated_scenes_dstream", f"stylization_testtest")
    save_dir = os.path.join(exp_dir, f"{args.result_dir}/epoch_{load_epoch:05d}/{args.exp_name}/{args.timesteps}_steps")
    if args.temperature is not None:
        save_dir += f"_temp_{args.temperature}"
    os.makedirs(save_dir, exist_ok=True)

    # Create scene evaluator
    max_rel_num = 4
    scene_evaluator = SceneEvaluator(
        raw_dataset, 
        dataset.object_types,
        dataset.predicate_types,
        max_rel_num,
        text_encoder,
        device,
        save_dir,
        args, config,
        dfs=False
    )
    scene_evaluator.transparent = False
    
    classes = np.array(dataset.object_types)
    templates = [
        "Make the room {} style.", "Make objects in the room {}", "Let the room be in {} style",
        "Make the room style {}.", "Make the room {}.", "Let objects be in {}.",
    ]
    styles = ["black", "white", "gray", "brown"]
    clip_cossims = []
 
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), ncols=125):
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device)

            np.random.seed(batch_idx)
            selected_styles = list(np.random.choice(styles, size=len(batch["scene_uids"])))
            texts = [np.random.choice(templates).format(style) for style in selected_styles]

            text_last_hidden_state, text_embeds = text_encoder(texts)

            model_output, token_masks = getattr(model, args.exp_name)(
                batch,
                max_length=config["data"]["max_length"],
                text_last_hidden_state=text_last_hidden_state,
                text_embeds=text_embeds,
                timesteps=args.timesteps,
                all_timesteps=args.all_steps,
                gsample=True if args.temperature is not None else False,
                temperature=args.temperature if args.temperature is not None else 1,
            )

            token_masks = torch.stack(token_masks, dim=1)

            
            # Process model output
            objfeats, bbox_params_t = process_model_output(
                model_output, dataset, vqvae_model,
                model.get_pad_ids())
            
            gt_objfeats = batch["objfeat_vq_indices"].long().to(device)
            BB, N = gt_objfeats.shape[:2]
            objfeat_vq_indices_rand = torch.randint_like(gt_objfeats, 0, 64)
            gt_objfeats[gt_objfeats == 64] = objfeat_vq_indices_rand[gt_objfeats == 64]
            objfeats_original = vqvae_model.reconstruct_from_indices(gt_objfeats.reshape(BB*N, -1)).reshape(BB, N, -1) # (BB, N, D)
            objfeats_original = objfeats_original.cpu().numpy() # (192, 21, 1280)
            
            # Evaluate (and visualize) each scene in the batch
            progress_bar = tqdm(total=len(bbox_params_t), desc="Visualize each scene", ncols=125)
            for i in range(len(bbox_params_t)):
                ii =  batch_idx * B + i
                # region) Eval for last step
                # Get the textured objects by retrieving the 3D models
                trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, obj_ids = get_textured_objects(
                    bbox_params_t[i,-1],
                    objects_dataset, classes,
                    objfeats[i,-1],
                    "openshape_vitg14",
                )

                if args.visualize_original:
                    trimesh_meshes_original, bbox_meshes_original, \
                    obj_classes_original, obj_sizes_original, obj_ids_original = get_textured_objects(
                        bbox_params_t[i,-1],
                        objects_dataset, classes,
                        objfeats_original[i],
                        "openshape_vitg14",
                        verbose=False
                    )

                # Compute the CLIP cosine similarity between the retrieved objects and the text
                style_texts = [
                    f"a {selected_styles[i].lower()} {obj_class.replace('_', ' ').lower()}"
                    for obj_class in obj_classes if obj_class != None
                ]
                class_texts = [
                    f"a {obj_class.replace('_', ' ').lower()}"
                    for obj_class in obj_classes if obj_class != None
                ]
                _, obj_clip_style_features = text_encoder(style_texts)  # (n_objs, D); already normalized
                _, obj_clip_class_features = text_encoder(class_texts)  # (n_objs, D); already normalized
                obj_clip_image_features = [
                    torch.from_numpy(np.load(
                        f"dataset/3D-FRONT/3D-FUTURE-model/{obj_id}/clip_vitb32.npy"
                    )).float().to(device)  # already normalized
                    for obj_id in obj_ids if obj_id != None
                ]
                obj_clip_image_features = torch.stack(obj_clip_image_features, dim=0)  # (n_objs, D)
                cossim = ((obj_clip_style_features - obj_clip_class_features) * obj_clip_image_features).sum(dim=-1)  # (n_objs,)
                clip_cossims.append(cossim.mean().item())

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "clip_cossim": "{:.6f}".format(clip_cossims[-1])
                })
                
                # Whether to visualize the scene by blender rendering
                if not args.visualize:
                    continue

                # To get the manually created floor plan, which includes vertices of all meshes in the scene
                all_vertices = np.concatenate([tr_mesh.vertices for tr_mesh in trimesh_meshes], axis=0)
                x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
                z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()

                tr_floor = floor_plan_from_scene(raw_dataset[0], config["data"]["path_to_floor_plan_textures"], 
                            room_size=[x_min, z_min, x_max, z_max])

                # Create a trimesh scene and export it to a temporary directory
                export_dir = os.path.join(save_dir, f"{ii:04d}@{batch['scene_uids'][i]}")


                scene_evaluator.render_scene(trimesh_meshes, bbox_meshes, export_dir, [tr_floor])

                if args.all_steps:
                    for step in range(bbox_params_t[i].shape[0]):
                        step_trimesh_meshes, step_bbox_meshes, _, _, _ = get_textured_objects(
                            bbox_params_t[i, step],
                            objects_dataset, classes,
                            objfeats[i, step],
                            "openshape_vitg14",
                        )
                        scene_evaluator.render_scene(step_trimesh_meshes, step_bbox_meshes, export_dir, [tr_floor], step)

                if args.visualize_original:
                    trimesh_meshes_original.append(tr_floor)
                    tmp_dir_original = os.path.join(export_dir, "tmp_original")
                    os.makedirs(tmp_dir_original, exist_ok=True)
                    export_scene(tmp_dir_original, trimesh_meshes_original, bbox_meshes_original)

                    blender_render_scene(
                        tmp_dir_original,
                        export_dir,
                        output_suffix="_original",
                        top_down_view=(not args.eight_views),
                        resolution_x=args.resolution,
                        resolution_y=args.resolution,
                        transparent=False
                    )

                with open(os.path.join(export_dir, "description.txt"), "w") as f:
                    f.write(texts[i])
                    
            syn_dir = os.path.join(save_dir, f"all_syns")
        os.makedirs(syn_dir, exist_ok=True)
        
        syn_images = []
        for scene_id in os.listdir(save_dir):
            try:
                scene_num = int(scene_id.split("@")[0])
                if 0 <= scene_num <= len(dataset) - 1:
                    topdown_path = os.path.join(save_dir, scene_id, "topdown.png")
                    if os.path.exists(topdown_path):
                        syn_images.append(topdown_path)
            except ValueError:
                continue
            
        for path in syn_images:
            name = os.path.basename(os.path.dirname(path)) + "_topdown.png"
            shutil.copyfile(path, os.path.join(syn_dir, name))

        num_syn_images = len(syn_images)
        print(f"Found [{num_syn_images}] synthesized images\n\n")

        configs = {"fdir1": real_dir,
                    "fdir2": syn_dir,
                    "device": device}
        
        fid_score = fid.compute_fid(verbose=False, **configs)

    # Save the evaluation results
    stat_file = os.path.join(save_dir, "stat.txt")
    eval_info = ""
    eval_info += f"CLIP cosine similarity: {np.mean(clip_cossims):.6f}\n"
    eval_info += f"FID score: {fid_score:.2f}\n"

    if eval_info != "":
        with open(stat_file, "w") as f:
            f.write(eval_info)
    
if __name__ == "__main__":
    main() 