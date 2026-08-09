import argparse
import random
import os
import time
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import get_dataset_raw_and_encoded, filter_function

from src.utils import *

from src.models import ObjectFeatureVQVAE
from src.models.utils import process_model_output
from src.tasks.utils import load_model


parser = argparse.ArgumentParser(description="SceneNAT unconditional generation")
parser.add_argument("--config_file", type=str, default=None, help="Path to the file that contains the experiment configuration")
parser.add_argument("--tag", type=str, default=None, help="Tag that refers to the current experiment")
parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
parser.add_argument("--checkpoint_epoch", type=int, default=None, help="The epoch to load the checkpoint from")
parser.add_argument("--use_best", action="store_true", help="Use best epoch")
parser.add_argument("--num_workers", type=int, default=1, help="The epoch to load the checkpoint from")
parser.add_argument("--n_epochs", type=int, default=1, help="The number of epochs for evaluation")
parser.add_argument("--visualize", action="store_true", help="Visualize the generated scenes")
parser.add_argument("--eight_views", action="store_true", help="Render 8 views of the scene")
parser.add_argument("--resolution", type=int, default=256,help="Resolution of the rendered image")
parser.add_argument("--timesteps", type=int, default=10, help="The number of timesteps for evaluation")
parser.add_argument("--temperature", type=float, default=None, help="The temperature for evaluation")
parser.add_argument("--model_version", type=str, default="0709", help="Model version (e.g., 0709, 0903v1)")
parser.add_argument("--result_dir", type=str, default="downstream", help="Result directory")
parser.add_argument("--exp_name", type=str, default="uncond", help="Experiment name")

args = parser.parse_args()

# seed = 250923
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

    # Check if `save_dir` exists and if it doesn't create it
    save_dir = os.path.join(exp_dir, f"{args.result_dir}/epoch_{load_epoch:05d}/{args.exp_name}/{args.timesteps}_steps")
    if args.temperature is not None:
        save_dir += f"_temp_{args.temperature}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create scene evaluator
    scene_evaluator = SceneEvaluator(
        raw_dataset, 
        dataset.object_types,
        dataset.predicate_types,
        4,
        None,
        device,
        save_dir,
        args, config,
        dfs=False,
        irecall=False
    )
    scene_evaluator.transparent = False

    with torch.no_grad():
        for epoch in range(args.n_epochs):
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), ncols=125):
                min_val = torch.min(batch["lengths"])
                max_val = torch.max(batch["lengths"])
                obj_len = torch.randint(low=min_val, high=max_val + 1, size=batch['lengths'].shape, device=device)
                model_output = getattr(model, args.exp_name)(
                    max_length=config["data"]["max_length"],
                    # obj_len=batch["lengths"].to(device),
                    obj_len=obj_len,
                    timesteps=args.timesteps,
                    gsample=True if args.temperature is not None else False,
                    temperature=args.temperature if args.temperature is not None else 1,
                )
                
                # Process model output
                objfeats, bbox_params_t = process_model_output(
                    model_output, dataset, vqvae_model,
                    model.get_pad_ids())
                
                progress_bar = tqdm(total=len(bbox_params_t), desc="Visualize each scene", ncols=125)
                for i in range(len(bbox_params_t)):
                    # scene id
                    scene_id = f"{epoch*len(dataset)+batch_idx*B+i:04d}@{batch['scene_uids'][i]}"

                    # Evaluate scene
                    scene_results = scene_evaluator.evaluate_scene(
                        bbox_params_t[i], 
                        objfeats[i], 
                        scene_id=scene_id, 
                    )
                    
                    # Update progress bar and metrics
                    scene_evaluator.update_metrics(scene_results)
                    progress_bar.update(1)

            # Gather and evaluate images for this epoch
            time.sleep(10)  # to make sure last rendering finished
            scene_evaluator.eval_rendered_images(epoch)

            # Save evaluation results for this epoch
            scene_evaluator.save_epoch_metrics(epoch)
            
        # Save final statistics over all epochs
        scene_evaluator.save_final_statistics()



if __name__ == "__main__":
    main() 