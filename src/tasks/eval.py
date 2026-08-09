import argparse
import random
import os
import time
import pickle
from tqdm import tqdm
import shutil
import importlib
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data import get_dataset_raw_and_encoded, filter_function
from src.data.utils_text import TextPreprocessor

from src.utils import *

from src.models import ObjectFeatureVQVAE, CLIPTextEncoder
from src.models.utils import process_model_output
from src.tasks.utils import load_model

parser = argparse.ArgumentParser(description="SceneNAT Evaluation")
parser.add_argument("--config_file", type=str, default=None, help="Path to the file that contains the experiment configuration")
parser.add_argument("--tag", type=str, default=None, help="Tag that refers to the current experiment")
parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
parser.add_argument("--temperature", type=float, default=0, help="temperature")
parser.add_argument("--checkpoint_epoch", type=int, default=None, help="The epoch to load the checkpoint from")
parser.add_argument("--use_best", action="store_true", help="Use best epoch")
parser.add_argument("--num_workers", type=int, default=1, help="The epoch to load the checkpoint from")
parser.add_argument("--n_epochs", type=int, default=1, help="The number of epochs for evaluation")
parser.add_argument("--visualize", action="store_true", help="Visualize the generated scenes")
parser.add_argument("--eight_views", action="store_true", help="Render 8 views of the scene")
parser.add_argument("--resolution", type=int, default=256,help="Resolution of the rendered image")
parser.add_argument("--timesteps", type=int, default=10, help="The number of timesteps for evaluation")
parser.add_argument("--all_steps", action="store_true", help="Save rendering of each iteration result")
parser.add_argument("--given_lengths", action="store_true", help="Given lengths")
parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
parser.add_argument("--fix_prev", action="store_true", help="Fix previously survived tokens")
parser.add_argument("--scene_wise_schedule", action="store_true", help="Scene wise scheduling")
parser.add_argument("--model_version", type=str, default="0709", help="Model version (e.g., 0709, 0903v1)")

args = parser.parse_args()


seed = 250134
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
    if "eval" not in config["data"]["encoding_type"]: config["data"]["encoding_type"] += "_eval"
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
    SceneNAT_model = load_model(args.model_version, exp_dir)
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

    # save_dir = os.path.join(exp_dir, f"generated_scenes/epoch_{load_epoch:05d}/{args.timesteps}_steps")
    save_dir = os.path.join(exp_dir, f"visualize_all_steps2/")
    if args.temperature is not None:
        save_dir += f"_temp_{args.temperature}"
    if args.cfg_scale is not None and args.cfg_scale != 1.0:
        save_dir += f"_cfg_{args.cfg_scale}"
    if args.fix_prev:
        save_dir += f"_fix_prev"
    if args.scene_wise_schedule:
        save_dir += f"_scene_wise_schedule"
    os.makedirs(save_dir, exist_ok=True)
    
    # region) Create scene evaluator
    scene_evaluator = SceneEvaluator(
        raw_dataset, 
        dataset.object_types,
        dataset.predicate_types,
        max_rel_num,
        text_encoder,
        device,
        save_dir,
        args, config,
        visualize=args.visualize
    )
    scene_evaluator.transparent = True
    # scene_evaluator.visualize = False
    # endregion

    with torch.no_grad():
        for epoch in range(args.n_epochs):
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), ncols=125):
                # region) Prepare CLIP text embeddings
                descriptions = batch["descriptions"]
                texts = []
                batch_selected_relations, batch_selected_descs, batch_selected_descs_no_dup = [], [], []

                for desc_idx, desc in enumerate(descriptions):
                    text, selected_relations, selected_descs, selected_descs_no_dup, triples = \
                        text_prep.fill_templates(
                            desc, 
                            batch["object_descs"][desc_idx],
                            seed=epoch * len(dataset) + batch_idx * B + desc_idx,
                            return_triplets=True,
                            return_descs_no_dup=True,
                        )
                    texts.append(text)
                    batch_selected_relations.append(selected_relations)
                    batch_selected_descs.append(selected_descs)
                    batch_selected_descs_no_dup.append(selected_descs_no_dup)

                text_last_hidden_state, text_embeds = text_encoder(texts)
                # endregion
                
                start_time = time.perf_counter()
                
                model_output, token_masks = model.generate_samples(
                    max_length=config["data"]["max_length"],
                    text_last_hidden_state=text_last_hidden_state,
                    text_embeds=text_embeds,
                    obj_len=batch["lengths"].to(device) if args.given_lengths else None,
                    timesteps=args.timesteps,
                    temperature=args.temperature if args.temperature > 0 else 1.0,
                    gsample=True if args.temperature > 0 else False,
                    all_timesteps=args.all_steps,
                    fix_prev=args.fix_prev,
                    att_wise_schedule=False if args.scene_wise_schedule else True,
                    cfg_scale=args.cfg_scale,
                )
                
                end_time = time.perf_counter()
                scene_evaluator.inference_times.append(end_time - start_time)
                
                # Process model output
                objfeats, bbox_params_t = process_model_output(
                    model_output, dataset, vqvae_model,
                    model.get_pad_ids())
                
                if not args.all_steps:
                    objfeats = objfeats[:,-1]
                    bbox_params_t = bbox_params_t[:,-1]
                
                token_masks = torch.stack(token_masks, dim=1)
                
                progress_bar = tqdm(total=len(bbox_params_t), desc="Visualize each scene", ncols=125)
                for i in range(len(bbox_params_t)):
                    if int(epoch*len(dataset)+batch_idx*B+i) != 22:
                        continue
                    # scene id
                    scene_id = f"{epoch*len(dataset)+batch_idx*B+i:04d}@{batch['scene_uids'][i]}"

                    # Evaluate scene
                    scene_results = scene_evaluator.evaluate_scene(
                        bbox_params_t[i], 
                        objfeats[i], 
                        batch_selected_relations[i], 
                        batch_selected_descs_no_dup[i], 
                        texts[i], 
                        scene_id, 
                        args.all_steps
                    )
                    
                    # Update progress bar and metrics
                    metrics = scene_evaluator.update_metrics(scene_results)

                    ######
                    if args.all_steps:
                        mask_save_dir = os.path.join(scene_evaluator.save_dir, scene_id, f"masks")
                        os.makedirs(mask_save_dir, exist_ok=True)
                        token_mask_steps = token_masks[i] # This is step_n * L * D matrix.
                        # Convert token_mask to image
                        step_n, L, D = token_mask_steps.shape
                        
                        # Save image
                        for step_idx in range(step_n):
                            plt.figure(figsize=(10, 10))  # Increase vertical size
                            plt.imshow(token_mask_steps[step_idx].cpu().numpy(), cmap='RdBu', aspect='auto')
                            
                            # Draw grid for each cell
                            ax = plt.gca()
                            ax.set_xticks(np.arange(-0.5, D, 1), minor=True)
                            ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
                            ax.grid(True, which='minor', color='black', linewidth=1)
                            
                            # Remove horizontal and vertical ticks
                            ax.set_xticks([])
                            ax.set_yticks([])
                            
                            plt.title(f'Mask Visualization - Step {step_idx}', fontsize=16, fontweight='bold', pad=20)
                            
                            # Mark x, o, t, s, r regions on horizontal axis
                            plt.axvline(x=0.5, color='white', linewidth=2)  # x end
                            plt.axvline(x=4.5, color='white', linewidth=2)  # o end
                            plt.axvline(x=7.5, color='white', linewidth=2)  # t end
                            plt.axvline(x=10.5, color='white', linewidth=2)  # s end
                            
                            # Add region labels
                            plt.text(0., -0.5, 'x', ha='center', va='bottom', fontsize=12, fontweight='bold')
                            plt.text(2.5, -0.5, 'o', ha='center', va='bottom', fontsize=12, fontweight='bold')
                            plt.text(6, -0.5, 't', ha='center', va='bottom', fontsize=12, fontweight='bold')
                            plt.text(9, -0.5, 's', ha='center', va='bottom', fontsize=12, fontweight='bold')
                            plt.text(11., -0.5, 'r', ha='center', va='bottom', fontsize=12, fontweight='bold')
                            
                            # Display object numbers on vertical axis
                            for obj_idx in range(L):
                                plt.text(-0.8, obj_idx, f'Object #{obj_idx+1}', ha='right', va='center', fontsize=10, fontweight='bold')
                            
                            save_path = os.path.join(mask_save_dir, f"step_{step_idx:02d}.png")
                            plt.savefig(save_path, dpi=150, bbox_inches='tight')
                            plt.close()
                    ######
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "rel": "{:.4f}".format(metrics["rel"]),
                        "erel": "{:.4f}".format(metrics["erel"])
                    })

            ######
            if not args.all_steps:
                scene_id = f"{epoch*len(dataset)+batch_idx*B:04d}@{batch['scene_uids'][0]}"
                mask_save_dir = os.path.join(scene_evaluator.save_dir, scene_id, f"masks")
                os.makedirs(mask_save_dir, exist_ok=True)
                token_mask_steps = token_masks[0] # This is step_n * L * D matrix.
                # Convert token_mask to image
                step_n, L, D = token_mask_steps.shape
                
                # Save image
                for step_idx in range(step_n):
                    plt.figure(figsize=(10, 10))  # Increase vertical size
                    plt.imshow(token_mask_steps[step_idx].cpu().numpy(), cmap='RdBu', aspect='auto')
                    
                    # Draw grid for each cell
                    ax = plt.gca()
                    ax.set_xticks(np.arange(-0.5, D, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
                    ax.grid(True, which='minor', color='black', linewidth=1)
                    
                    # Remove horizontal and vertical ticks
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    plt.title(f'Mask Visualization - Step {step_idx}', fontsize=16, fontweight='bold', pad=20)
                    
                    # Mark x, o, t, s, r regions on horizontal axis
                    plt.axvline(x=0.5, color='white', linewidth=2)  # x end
                    plt.axvline(x=4.5, color='white', linewidth=2)  # o end
                    plt.axvline(x=7.5, color='white', linewidth=2)  # t end
                    plt.axvline(x=10.5, color='white', linewidth=2)  # s end
                    
                    # Add region labels
                    plt.text(0., -0.5, 'x', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    plt.text(2.5, -0.5, 'o', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    plt.text(6, -0.5, 't', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    plt.text(9, -0.5, 's', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    plt.text(11., -0.5, 'r', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
                    # Display object numbers on vertical axis
                    for obj_idx in range(L):
                        plt.text(-0.8, obj_idx, f'Object #{obj_idx+1}', ha='right', va='center', fontsize=10, fontweight='bold')
                    
                    save_path = os.path.join(mask_save_dir, f"step_{step_idx:02d}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
            ######
            

            # Gather and evaluate images for this epoch
            time.sleep(10)  # to make sure last rendering finished
            if args.visualize:
                scene_evaluator.eval_rendered_images(epoch)

            # Save evaluation results for this epoch
            scene_evaluator.save_epoch_metrics(epoch)
            
        # Save final statistics over all epochs
        scene_evaluator.save_final_statistics()



if __name__ == "__main__":
    main() 