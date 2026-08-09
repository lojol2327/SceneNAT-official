# ours_completion.py
import argparse
import random
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import shutil
from cleanfid import fid

from src.data import get_dataset_raw_and_encoded, filter_function
from src.data.utils_text import TextPreprocessor
from src.utils import *
from src.models import ObjectFeatureVQVAE, CLIPTextEncoder
from src.models.utils import process_model_output
from .utils import load_model

parser = argparse.ArgumentParser(description="SceneNAT layout-to-object")
parser.add_argument("--config_file", type=str, default=None, help="Path to the file that contains the experiment configuration")
parser.add_argument("--tag", type=str, default=None, help="Tag that refers to the current experiment")
parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
parser.add_argument("--checkpoint_epoch", type=int, default=None, help="The epoch to load the checkpoint from")
parser.add_argument("--use_best", action="store_true", help="Use best epoch")

parser.add_argument("--num_workers", type=int, default=1, help="The epoch to load the checkpoint from")
parser.add_argument("--visualize", action="store_true", help="Visualize the generated scenes")
parser.add_argument("--eight_views", action="store_true", help="Render 8 views of the scene")
parser.add_argument("--resolution", type=int, default=256,help="Resolution of the rendered image")
parser.add_argument("--timesteps", type=int, default=10, help="The number of timesteps for evaluation")
parser.add_argument("--temperature", type=float, default=None, help="The temperature for evaluation")
parser.add_argument("--all_steps", action="store_true", help="Save rendering of each iteration result")
parser.add_argument("--visualize_partial", action="store_true", help="Visualize the partial scenes with masking")
parser.add_argument("--model_version", type=str, default="0709", help="Model version (e.g., 0709, 0903v1)")
parser.add_argument("--result_dir", type=str, default="downstream", help="Result directory")
parser.add_argument("--exp_name", type=str, default="layouto", help="Experiment name")

args = parser.parse_args()

# seed = 250924
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
    
    # region) Load text encoder, tokenizer
    print(f"Load pretrained text encoder CLIP\n")
    max_rel_num = 4
    max_obj_num = dataset.max_length
    text_encoder = CLIPTextEncoder(config["model"]["text_encoder"], device=device)
    text_prep = TextPreprocessor(dataset.object_types, dataset.predicate_types, max_rel_num)
    # endregion
    
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
        max_obj_num=max_obj_num,
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
    
    save_dir = os.path.join(exp_dir, f"{args.result_dir}/epoch_{load_epoch:05d}/{args.exp_name}_seed2/{args.timesteps}_steps")
    if args.temperature is not None:
        save_dir += f"_temp_{args.temperature}"
    os.makedirs(save_dir, exist_ok=True)
    
    classes = np.array(dataset.object_types)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device)

            # Masking strategy: random masking of ~30% of objects per scene
            batch_selected_obj_ids = []
            BB = len(batch["lengths"])
            for b in range(BB):
                valid_len = batch["lengths"][b].long().item()
                num_to_mask = max(1, int(0.3 * valid_len))  # Mask at least one object
                # 추가적인 시드 설정으로 완전히 동일한 결과 보장
                torch.manual_seed(seed + batch_idx * 1000 + b)
                selected = torch.randperm(valid_len)[:num_to_mask].tolist()
                batch_selected_obj_ids.append(selected)

            # Completion
            model_output = getattr(model, args.exp_name)(
                batch=batch,
                mask_object_indices=batch_selected_obj_ids,
                max_length=config["data"]["max_length"],
                timesteps=args.timesteps,
                gsample=True if args.temperature is not None else False,
                temperature=args.temperature if args.temperature is not None else 1,
            )

            objfeats, bbox_params_t = process_model_output(
                model_output, dataset, vqvae_model,
                model.get_pad_ids()
            )

            objfeats = objfeats[:,-1]
            bbox_params_t = bbox_params_t[:,-1]

            progress_bar = tqdm(total=len(bbox_params_t), desc="Visualize each scene", ncols=125)
            for i in range(len(bbox_params_t)):
                # Get the textured objects by retrieving the 3D models
                trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, obj_ids = get_textured_objects(
                    bbox_params_t[i],
                    objects_dataset, classes,
                    objfeats[i],
                    "openshape_vitg14",
                    verbose=False,
                    get_bbox_meshes=True
                )
                
                trimesh_meshes_ori = [
                    trimesh_meshes[mesh_idx] for mesh_idx in range(len(trimesh_meshes))
                    if mesh_idx not in batch_selected_obj_ids[i]
                ]

                assert len(bbox_meshes) > 0, "bbox_meshes is empty"
                bbox_meshes_given = [
                    bbox_meshes[mesh_idx] for mesh_idx in range(len(bbox_meshes))
                    if mesh_idx in batch_selected_obj_ids[i]
                ]
                
                # Whether to visualize the scene by blender rendering
                if not args.visualize:
                    continue

                # To get the manually created floor plan, which includes vertices of all meshes in the scene
                all_vertices = np.concatenate([tr_mesh.vertices for tr_mesh in trimesh_meshes], axis=0)
                x_max, x_min = all_vertices[:, 0].max(), all_vertices[:, 0].min()
                z_max, z_min = all_vertices[:, 2].max(), all_vertices[:, 2].min()

                tr_floor = floor_plan_from_scene(raw_dataset[0], config["data"]["path_to_floor_plan_textures"],
                    room_size=[x_min, z_min, x_max, z_max])
                trimesh_meshes.append(tr_floor)
                trimesh_meshes_ori.append(tr_floor)

                # Create a trimesh scene and export it to a temporary directory
                ii =  batch_idx * B + i
                export_dir = os.path.join(save_dir, f"{ii:04d}@{batch['scene_uids'][i]}")
                tmp_dir = os.path.join(export_dir, "tmp")
                os.makedirs(export_dir, exist_ok=True)
                os.makedirs(tmp_dir, exist_ok=True)
                export_scene(tmp_dir, trimesh_meshes, None)

                # Render the exported scene by calling blender
                blender_render_scene(
                    tmp_dir,
                    export_dir,
                    top_down_view=(not args.eight_views),
                    resolution_x=args.resolution,
                    resolution_y=args.resolution,
                    transparent=False
                )
                if args.visualize_partial:
                    tmp_dir_partial = os.path.join(export_dir, "tmp_partial")
                    os.makedirs(tmp_dir_partial, exist_ok=True)
                    export_scene(tmp_dir_partial, trimesh_meshes_ori, bbox_meshes_given)

                    blender_render_scene(
                        tmp_dir_partial,
                        export_dir,
                        output_suffix="_partial",
                        top_down_view=(not args.eight_views),
                        resolution_x=args.resolution,
                        resolution_y=args.resolution,
                        transparent=False
                    )

                progress_bar.update(1)
                

        syn_dir = os.path.join(save_dir, f"all_syns")
        os.makedirs(syn_dir, exist_ok=True)
        
        syn_images = []
        for scene_id in os.listdir(save_dir):
            # Check if the folder name is in a numeric format and within the range of the current epoch
            try:
                scene_num = int(scene_id.split("@")[0])  # "0001@scene_id" -> extract 0001
                if 0 <= scene_num <= len(dataset) - 1:
                    topdown_path = os.path.join(save_dir, scene_id, "topdown.png")
                    if os.path.exists(topdown_path):
                        syn_images.append(topdown_path)
            except ValueError:
                continue  # Skip if the folder name is not in the expected format
            
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
    eval_info += f"FID score: {fid_score:.2f}\n"

    if eval_info != "":
        with open(stat_file, "w") as f:
            f.write(eval_info)

if __name__ == "__main__":
    main()
