import argparse
import random
import os
import time
import gc
import shutil
import json
import yaml
import logging
from tqdm import tqdm
import pickle

import numpy as np
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils import *
from src.data import get_encoded_dataset, get_dataset_raw_and_encoded,filter_function
from src.data.utils_text import TextPreprocessor, TextPreprocessorOriginal

from src.models import optimizer_from_config, scheduler_from_config
from src.models import ObjectFeatureVQVAE, CLIPTextEncoder
from src.models import process_model_output, extract_partial_obj_list_and_remap


parser = argparse.ArgumentParser(description="SceneNAT training")
parser.add_argument("--config_file", type=str, default=None, help="Path to the file that contains the experiment configuration")
parser.add_argument("--tag", type=str, default=None, help="Tag that refers to the current experiment")
parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
parser.add_argument("--resume", action="store_true", help="resume training")
parser.add_argument("--checkpoint_epoch", type=int, default=None, help="The epoch to load the checkpoint from")
parser.add_argument("--timesteps", type=int, default=10, help="The number of decoding timesteps")
parser.add_argument("--num_workers", type=int, default=8, help="The epoch to load the checkpoint from")
parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--test", action="store_true", help="test during training")
parser.add_argument("--parse", action="store_true", help="parse during test")
parser.add_argument("--original_text_preprocessor", action="store_true", help="original text preprocessor")
parser.add_argument("--model_version", type=str, default="0903v2", help="SceneNAT model version (e.g., 0903v2, 0903v3)")
args = parser.parse_args()

def load_model(args, exp_dir):
    """Load SceneNAT model class"""
    model_version = args.model_version
    
    if args.resume and exp_dir:
        try:
            import sys
            models_script_dir = os.path.join(exp_dir, "models_script")
            if not os.path.exists(models_script_dir):
                print(f"Models script directory not found: {models_script_dir}")
                # Fallback to default import
                return load_model(argparse.Namespace(**{**vars(args), 'resume': False}), None)

            # Check if models_script directory is already in sys.path
            models_script_parent_dir = os.path.dirname(models_script_dir)
            if models_script_parent_dir not in sys.path:
                sys.path.insert(0, models_script_parent_dir)

            module_name = f"models_script.scene_nat_{model_version}"
            class_name = f"SceneNAT_{model_version}"
            
            module = __import__(module_name, fromlist=[class_name])
            model = getattr(module, class_name)
            print(f"Successfully loaded SceneNAT class: {class_name} from resume")
            return model
        except Exception as e:
            print(f"Error loading SceneNAT class for version {model_version} from resume: {e}")
            return None
    else:
        # Dynamic class loading for new training
        try:
            class_name = f"SceneNAT_{model_version}"
            module = __import__(f"src.models", fromlist=[class_name])
            model = getattr(module, class_name)
            print(f"Successfully loaded SceneNAT class: {class_name} from source")
            return model
        except (ImportError, AttributeError) as e:
            print(f"Error importing SceneNAT class for version {model_version} from source: {e}")
            return None


def setup_seed(config):
    """Set random seed"""
    seed = config.get("training", {}).get("seed", 250130)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def get_optimal_num_workers():
    """Calculate optimal num_workers based on CPU cores"""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    # Use half of the CPU cores, but limit it to a minimum of 1 and a maximum of 8
    optimal_workers = min(max(cpu_count // 2, 1), 8)
    return optimal_workers

def setup_ddp_and_print():
    """Initialize DDP and override the print function"""
    import builtins
    original_print = builtins.print
    
    def is_main_process():
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        return local_rank in [-1, 0]
    
    def ddp_print(*args, **kwargs):
        if is_main_process():
            original_print(*args, **kwargs)
    
    # Override the print function
    builtins.print = ddp_print
    
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    
    # Initialize DDP
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    
    print(f"Available GPUs: {torch.cuda.device_count()}, Run code on device [{device}]\n")
    
    return device, is_main_process

def setup_experiment_directory(args, is_main_process):
    """Set up experiment directory and copy files"""
    os.makedirs(args.output_dir, exist_ok=True)
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
        
    if args.resume:
        # Find json file in exp_dir
        json_files = [f for f in os.listdir(exp_dir) if f.endswith('.json')]
        assert len(json_files) > 0, f"No json file found in {exp_dir}"
        args.config_file = os.path.join(exp_dir, json_files[0])
        print(f"Using config file: {args.config_file}")

    else:
        # Save config files into experiment dir.
        if is_main_process():
            room_type = os.path.splitext(os.path.basename(args.config_file))[0]
            # save yaml with json format
            config_filename = room_type + ".json"
            with open(args.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            with open(os.path.join(exp_dir, config_filename), 'w') as f:
                json.dump(config_data, f, indent=4)
            
            # save shell script
            workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sh_script_path = os.path.join(workspace_dir, "scripts", "train_ddp.sh")
            if os.path.exists(sh_script_path):
                shutil.copy2(sh_script_path, os.path.join(exp_dir, "train.sh"))
            
            # Copy src/models folder to models_script (overwrite if exists)
            src_models_path = os.path.join(workspace_dir, "src", "models")
            dst_models_path = os.path.join(exp_dir, "models_script")
            if os.path.exists(src_models_path):
                if os.path.exists(dst_models_path):
                    shutil.rmtree(dst_models_path)
                shutil.copytree(src_models_path, dst_models_path, ignore=shutil.ignore_patterns('__pycache__'))
            
            # save train.py
            train_script_path = os.path.abspath(__file__)
            shutil.copy2(train_script_path, os.path.join(exp_dir, "train.py"))
    
    return exp_dir, ckpt_dir

def setup_datasets(config, args):
    """Set up datasets and DataLoader"""
    if args.tag == 'debug':
        batch_size = 8
    else: 
        batch_size = config["training"]["batch_size"]

    if dist.is_initialized() and dist.is_available():
        world_size = dist.get_world_size()
        batch_size = config["training"]["batch_size"] // world_size
    else:
        batch_size = config["training"]["batch_size"]
        
    print("\n--- Preparing Training Dataset ---")
    filter_function(config["data"], split=config["training"].get("splits", ["train", "val"]))
    train_dataset = get_encoded_dataset(config["data"],
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"]))

    print("\n--- Preparing Validation Dataset ---")
    config["data"]["encoding_type"] += "_eval"  # use the evaluation encoding for the validation dataset
    filter_function(config["data"], split=config["validation"].get("splits", ["test"]))
    
    raw_dataset, val_dataset = get_dataset_raw_and_encoded(
        config["data"],
        augmentations=None,
        split=config["validation"].get("splits", ["test"]))
    
    # Set up DistributedSampler for DDP
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    
    # Calculate optimal num_workers
    optimal_num_workers = get_optimal_num_workers()
    actual_num_workers = min(args.num_workers, optimal_num_workers)
    
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )
    print(f"Loaded [{len(train_dataset)}] training scenes with {train_dataset.n_object_types} object types")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=4,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
        sampler=val_sampler
    )
    print(f"Loaded [{len(val_dataset)}] training scenes with {val_dataset.n_object_types} object types")
    
    # Make sure that the `train_dataset` and the `val_dataset` have the same number of object categories
    assert train_dataset.object_types == val_dataset.object_types
    
    return train_dataset, val_dataset, raw_dataset, train_loader, val_loader, batch_size, train_sampler

def setup_text_encoder_and_preprocessor(config, train_dataset, args, device):
    """Set up text encoder and preprocessor"""
    max_num_rel = 4
    max_obj_num = train_dataset.max_length
    text_encoder = CLIPTextEncoder(config["model"]["text_encoder"], device=device)
    if args.original_text_preprocessor:
        text_prep = TextPreprocessorOriginal(train_dataset.object_types, train_dataset.predicate_types)
    else:
        text_prep = TextPreprocessor(train_dataset.object_types, train_dataset.predicate_types, max_num_rel)
    
    return text_encoder, text_prep, max_num_rel, max_obj_num

def setup_model_and_optimizer(config, train_dataset, max_obj_num, device, model_class):
    """Set up model and optimizer"""
    objfeat_dim = 64
    print(f"VQ-VAE model detected, using objfeat_dim: {objfeat_dim}")
    
    model = model_class(
                train_dataset.n_object_types,
                train_dataset.n_predicate_types,
                objfeat_dim=objfeat_dim,
                text_dim=512,
                bounds=train_dataset.bounds,
                t_disc_dim=train_dataset.t_disc_dim,
                r_disc_dim=train_dataset.r_disc_dim,
                s_disc_dim=train_dataset.s_disc_dim,
                loss_weights=config["training"]["loss_weights"],
                max_obj_num=max_obj_num,
                **config["model"]["transformer_config"],
                ).to(device)
    
    optimizer = optimizer_from_config(config["training"]["optimizer"],
        filter(lambda p: p.requires_grad, model.parameters()))
    
    # Initialize scheduler
    scheduler = None
    if "scheduler" in config["training"]:
        scheduler = scheduler_from_config(
            config["training"]["scheduler"], 
            optimizer, 
            config["training"]["epochs"]
        )
    
    return model, optimizer, scheduler

def safe_load_checkpoint(model, ckpt_dir, device, optimizer, scheduler, args):
    """Load checkpoint (with error handling)"""
    if args.resume:
        try:
            start_epoch, lowest_epoch, lowest_fid = load_checkpoints(
                model, ckpt_dir, optimizer=optimizer, scheduler=scheduler,
            epoch=args.checkpoint_epoch, 
                get_last=False if args.checkpoint_epoch is not None else True, 
                device=device
            )
            print(f"Successfully loaded checkpoint from {ckpt_dir}")
            return start_epoch, lowest_epoch, lowest_fid
        except FileNotFoundError:
            print(f"Checkpoint not found in {ckpt_dir}, starting from scratch")
            return 0, -1, float("inf")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch")
            return 0, -1, float("inf")
    else:
        print(f"Starting new training from scratch")
        return 0, -1, float("inf")

def setup_distributed_model(model, device, args):
    """Set up distributed model (DDP/DataParallel)"""
    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
        else:
            model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        model = model.to(device)
    
    # wait for all processes to reach this point
    if args.local_rank != -1:
        torch.distributed.barrier()
    
    return model

def setup_logging_and_writer(exp_dir, start_epoch, args):
    """Set up logging and TensorBoard"""
    # Control writer only in DDP
    if args.local_rank != -1 and args.local_rank != 0:
        writer = None
        StatsLogger.instance().add_output_file(None)
    else:
        writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"), purge_step=start_epoch)
        StatsLogger.instance().add_output_file(open(os.path.join(exp_dir, "logs.txt"), "a"))
    
    return writer

def get_training_parameters(config, batch_size):
    """Extract training parameters"""
    epochs = config["training"]["epochs"]
    
    # Set steps_per_epoch (from config file or auto-calculate)
    steps_per_epoch_config = config["training"].get("steps_per_epoch", "auto")
    if steps_per_epoch_config == "auto":
        steps_per_epoch = int(np.ceil(500*128/batch_size))  # 128=original batch size
    else:
        steps_per_epoch = steps_per_epoch_config
    
    loss_weights = config["training"]["loss_weights"]
    log_freq = config["training"]["log_frequency"]    # in iterations
    val_freq = config["validation"]["frequency"]      # in epochs
    save_frequency = config["training"]["save_frequency"]
    
    return epochs, steps_per_epoch, loss_weights, log_freq, val_freq, save_frequency

def setup_scene_evaluator(args, exp_dir, raw_dataset, train_dataset, max_num_rel, text_encoder, device, config):
    """Set up SceneEvaluator (for testing)"""
    scene_evaluator = None
    vqvae_model = None
    save_dir = None
    
    if args.test:
        print("Load pretrained VQ-VAE ...\n")
        with open("output/vqvae_openshape/objfeat_bounds.pkl", "rb") as f:
            kwargs = pickle.load(f)
        vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
        ckpt_path = f"output/vqvae_openshape/epoch_01999.pth"
        vqvae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
        vqvae_model = vqvae_model.to(device)
        vqvae_model.eval()

        save_dir = os.path.join(exp_dir, "_tmp_test")
        args.eight_views = False
        args.resolution = 256
        scene_evaluator = SceneEvaluator(
            raw_dataset, 
            train_dataset.object_types,
            train_dataset.predicate_types,
            max_num_rel,
            text_encoder,
            device,
            save_dir,
            args, config,
            # dfs=False
        )
        scene_evaluator.transparent = False
    
    return scene_evaluator, vqvae_model, save_dir


def setup_logging(exp_dir):
    """Set up structured logging system"""
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, 'logs.txt')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def validate_config(config):
    """Basic config file validation"""
    required_sections = ["training", "validation", "data", "model"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required section '{section}' not found in config file")
    
    training_required = ["batch_size", "epochs", "loss_weights", "log_frequency", "save_frequency"]
    for key in training_required:
        if key not in config["training"]:
            raise ValueError(f"Required key '{key}' not found in training section")
    
    if "frequency" not in config["validation"]:
        raise ValueError("Required key 'frequency' not found in validation section")
    
    if "max_length" not in config["data"]:
        raise ValueError("Required key 'max_length' not found in data section")
    
    if "transformer_config" not in config["model"]:
        raise ValueError("Required key 'transformer_config' not found in model section")

def main():
    # DDP initialization and print control setup
    device, is_main_process = setup_ddp_and_print()
    
    exp_dir, ckpt_dir = setup_experiment_directory(args, is_main_process)
    
    config: Dict[str, Dict[str, Any]] = load_config(args.config_file)
    
    # Validate config file
    validate_config(config)
    
    # Set random seed
    seed = setup_seed(config)
    
    # Set up datasets and DataLoader
    train_dataset, val_dataset, raw_dataset, train_loader, val_loader, batch_size, train_sampler = setup_datasets(config, args)
    
    # Set up text encoder and preprocessor
    text_encoder, text_prep, max_num_rel, max_obj_num = setup_text_encoder_and_preprocessor(config, train_dataset, args, device)
    
    # Load SceneNAT model class
    model_class = load_model(args, exp_dir)
    if model_class is None:
        raise ValueError(f"Failed to load SceneNAT model for version: {args.model_version}")
    
    # Set up model and optimizer
    model, optimizer, scheduler = setup_model_and_optimizer(config, train_dataset, max_obj_num, device, model_class)
    
    # Load checkpoint (with error handling)
    start_epoch, lowest_epoch, lowest_fid = safe_load_checkpoint(model, ckpt_dir, device, optimizer, scheduler, args)        
    highest_irecall = 0.0
    highest_irecall_epoch = -1
    
    # Set up distributed model
    model = setup_distributed_model(model, device, args)
    
    # Set up logging and TensorBoard
    writer = setup_logging_and_writer(exp_dir, start_epoch, args)
    logger = setup_logging(exp_dir)
    
    # Training start logs
    logger.info("=" * 50)
    logger.info("SceneNAT Training Started")
    logger.info(f"Experiment: {args.tag}")
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Model Version: {args.model_version}")
    logger.info(f"Output Directory: {exp_dir}")
    logger.info("=" * 50)

    # Extract training parameters
    epochs, steps_per_epoch, loss_weights, log_freq, val_freq, save_frequency = get_training_parameters(config, batch_size)

    # Set up SceneEvaluator (for testing)
    scene_evaluator, vqvae_model, save_dir = setup_scene_evaluator(args, exp_dir, raw_dataset, train_dataset, max_num_rel, text_encoder, device, config)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}\n")                    
    print("================================\n")
 
    # region) Start training
    logger.info(f"Training started with {epochs} epochs, {steps_per_epoch} steps per epoch")
    print(f"steps_per_epoch: {steps_per_epoch}")
    
    for i in range(start_epoch, epochs):
        model.train()
        if args.local_rank != -1:
            train_sampler.set_epoch(i)

        train_pbar = tqdm(
            zip(range(steps_per_epoch), yield_forever(train_loader)),
            total=steps_per_epoch,
            desc=f"Epoch {i+1}/{epochs}",
            ncols=100,
            disable=not is_main_process()
        )
        
        for b, batch in train_pbar:
            for k, v in batch.items():
                if not isinstance(v, list):
                    batch[k] = v.to(device)
            
            descriptions = batch["descriptions"]
            texts, spo_class_list, spo_slot_list = [], [], []

            for desc_idx, desc in enumerate(descriptions):
                text, spo_classes, descs, spo_slots = text_prep.fill_templates(
                    desc, batch["object_descs"][desc_idx], 
                    return_descs_no_dup=False,
                    return_triplets=True,
                )
                texts.append(text)
                spo_class_list.append(spo_classes)
                spo_slot_list.append(spo_slots)

            text_last_hidden_state, text_embeds = text_encoder(texts)
            
            optimizer.zero_grad()
            losses, acc = model(batch, text_last_hidden_state, text_embeds,
                        triple_list=spo_slot_list,
                        spo_class_list=spo_class_list,
                        predicate_types=train_dataset.predicate_types
            )
            total_loss = torch.zeros(1, device=device)

            for k, v in losses.items():
                total_loss += v

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            StatsLogger.instance().update_loss(total_loss.item() * batch["objs"].shape[0], batch["objs"].shape[0])
            
            if is_main_process():
                train_pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            if is_main_process() and (i * steps_per_epoch + b) % log_freq == 0:
                if writer is not None:
                    writer.add_scalar("training/loss", total_loss.item(), i * steps_per_epoch + b)
                    current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
                    writer.add_scalar("training/lr", current_lr, i * steps_per_epoch + b)
                    if len(losses) > 1:
                        for k, v in losses.items():
                            writer.add_scalar(f"training/{k}", v.item(), i * steps_per_epoch + b)
                    for k, v in acc.items():
                        writer.add_scalar(f"training/{k}", torch.mean(v), i * steps_per_epoch + b)
                  
            if b > 0 and b % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Update learning rate with scheduler
        if scheduler is not None:
            scheduler.step(i)
            if args.local_rank != -1:
                torch.distributed.barrier()

        if args.local_rank == -1 or args.local_rank == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint(model_to_save, optimizer, ckpt_dir, i, scheduler=scheduler, name = "last_model.pth")
            
            if (i + 1) % save_frequency == 0:
                save_checkpoint(model_to_save, optimizer, ckpt_dir, i, scheduler=scheduler)
                
        if is_main_process():
            StatsLogger.instance().clear()

        if (i+1) % val_freq == 0:
            model.eval()
            
            epoch_total_loss = 0
            cnt = 0

            val_pbar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Validation Epoch {i+1}",
                ncols=100,
                disable=not is_main_process()
            )

            with torch.no_grad():
                for val_b, val_batch in val_pbar:
                    for k, v in val_batch.items():
                        if not isinstance(v, list):
                            val_batch[k] = v.to(device)
                        
                    descriptions = val_batch["descriptions"]
                    texts, triple_list = [], []
                    mapped_triples_list, padded_objs_list = [], []
                    batch_selected_relations, batch_selected_descs_no_dup = [], []
                    
                    for desc_idx, desc in enumerate(descriptions):
                        text, selected_relations, selected_descs, selected_descs_no_dup, triples = \
                            text_prep.fill_templates(
                                desc, val_batch["object_descs"][desc_idx], 
                                return_descs_no_dup=True, 
                                return_triplets=True,
                                seed= val_b * len(val_loader) + desc_idx,
                                # seed=seed,
                            )
                        if args.test:
                            partial_obj_list, mapped_triples = extract_partial_obj_list_and_remap(val_batch["objs"][desc_idx], triples)
                            mapped_triples_list.append(mapped_triples)
                            padded_objs = partial_obj_list + [(model.module if hasattr(model, 'module') else model).x_mask_id] * (max_obj_num - len(partial_obj_list))
                            padded_objs_list.append(padded_objs)

                        texts.append(text)
                        batch_selected_relations.append(selected_relations)
                        batch_selected_descs_no_dup.append(selected_descs_no_dup)
                        triple_list.append(triples)
                        
                    text_last_hidden_state, text_embeds = text_encoder(texts)

                    val_losses, val_acc = model(val_batch, text_last_hidden_state, text_embeds,
                            triple_list=triple_list,
                            spo_class_list=batch_selected_relations,
                            predicate_types=train_dataset.predicate_types,
                            is_training=False,
                        )

                    val_total_loss = torch.zeros(1, device=device)
                    for k, v in val_losses.items():
                        if k in loss_weights:
                            val_total_loss += loss_weights[k] * torch.mean(v)
                        else:
                            val_total_loss += torch.mean(v)

                    StatsLogger.instance().update_loss(val_total_loss.item() * val_batch["objs"].shape[0], val_batch["objs"].shape[0])
                    
                    if is_main_process():
                        val_pbar.set_postfix({
                            'val_loss': f'{val_total_loss.item():.4f}'
                        })
                    
                    if args.local_rank == -1 or args.local_rank == 0:
                        StatsLogger.instance().print_progress(i, val_b, val=True)
                    
                    epoch_total_loss += val_total_loss.item()
                    cnt += 1

                    if args.test:
                        model_to_eval = model.module if hasattr(model, 'module') else model
                        model_output, _ = model_to_eval.generate_samples(
                            max_length=config["data"]["max_length"],
                            text_last_hidden_state=text_last_hidden_state,
                            text_embeds=text_embeds,
                            obj_len=val_batch["lengths"].to(device) if not config["model"]["transformer_config"]["predict_pad"] else None,
                            cfg_scale=config["model"]["transformer_config"]["cfg_scale"],
                            timesteps=args.timesteps
                        )
        
                        objfeats, bbox_params_t = process_model_output(
                            model_output, val_dataset, vqvae_model,
                            model_to_eval.get_pad_ids())
                        
                        test_pbar = tqdm(
                            range(len(bbox_params_t)), 
                            desc="Visualize each scene", 
                            ncols=125,
                            disable=not is_main_process()
                        )
                        for j in test_pbar:
                            scene_id = f"{j:04d}@{val_batch['scene_uids'][j]}"
                            
                            scene_results = scene_evaluator.evaluate_scene(
                                bbox_params_t[j], 
                                objfeats[j], 
                                rels=batch_selected_relations[j], 
                                descs=batch_selected_descs_no_dup[j],
                                scene_id=scene_id, 
                                verbose=False
                            )
                            
                            metrics = scene_evaluator.update_metrics(scene_results)
                            
                            if is_main_process():
                                test_pbar.set_postfix({
                                    "rel": "{:.4f}".format(metrics["rel"]),
                                    "erel": "{:.4f}".format(metrics["erel"])
                                })
                    
                    if val_b > 0 and val_b % 100 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                if args.test:
                    time.sleep(10)

                    if args.local_rank == -1 or args.local_rank == 0:
                        scene_evaluator.eval_rendered_images(0)
                        scene_evaluator.save_epoch_metrics(i+1, exp_dir)
                        shutil.rmtree(save_dir)
                    if args.local_rank != -1:
                        torch.distributed.barrier()

            if is_main_process():
                if writer is not None:
                    writer.add_scalar("validation/loss", StatsLogger.instance().loss, i * steps_per_epoch + b)
                    if len(val_losses) > 1:
                        for k, v in val_losses.items():
                            writer.add_scalar(f"validation/{k}", v.item(), i * steps_per_epoch + b)
                    for k, v in val_acc.items():
                        writer.add_scalar(f"validation/{k}", torch.mean(v), i * steps_per_epoch + b)

                avg_epoch_loss = epoch_total_loss / cnt if cnt > 0 else 0
                logger.info(f"Validation Epoch {i+1} Average Loss: {avg_epoch_loss:.7f}")
                print(f"Average Loss: {avg_epoch_loss:.7f}")
                if args.test:
                    irecall = scene_evaluator.epoch_metrics["relation_accs"][-1]
                    logger.info(f"Validation Epoch {i+1} FID: {scene_evaluator.fid:.2f}")
                    logger.info(f"Validation Epoch {i+1} iRecall: {irecall:.2f}")
                    print(f"FID: {scene_evaluator.fid:.2f}")
                    print(f"iRecall: {irecall:.2f}")

                    if scene_evaluator.fid < lowest_fid:
                        logger.info(f"New best FID found: {scene_evaluator.fid:.2f} at epoch {i+1} (previously {lowest_fid:.2f})")
                        print("BEST FID MODEL SO FAR")
                        lowest_fid = scene_evaluator.fid
                        lowest_epoch = i
                        save_checkpoint(model.module if hasattr(model, 'module') else model, optimizer, ckpt_dir, i, scheduler=scheduler, name="best_model.pth", loss=scene_evaluator.fid)

                    if irecall > highest_irecall:
                        logger.info(f"New best iRecall found: {irecall:.2f} at epoch {i+1} (previously {highest_irecall:.2f})")
                        print("BEST iRecall SO FAR")
                        highest_irecall = irecall
                        highest_irecall_epoch = i
                
                if writer is not None:
                    writer.add_scalar("Test/FID", scene_evaluator.fid, i)
                    writer.add_scalar("Test/FID_clip", scene_evaluator.fid_clip, i)
                    writer.add_scalar("Test/KID", scene_evaluator.kid, i)
                    writer.add_scalar("Test/iRecall", irecall, i)
                    writer.add_scalar("Test/DOS", scene_evaluator.epoch_metrics['dos'][-1]*1000, i)
                    writer.add_scalar("Test/DOS_fixed", scene_evaluator.epoch_metrics['dos_fixed'][-1]*1000, i)
                    writer.add_scalar("Test/DOS_recall", scene_evaluator.epoch_metrics['dos_recall'][-1]*1000, i)
                    scene_evaluator.reset_metrics()

                if args.test:
                    logger.info(f"Best scores so far -- FID: {lowest_fid:.2f} (epoch {lowest_epoch+1}), iRecall: {highest_irecall:.2f} (epoch {highest_irecall_epoch+1})")
                StatsLogger.instance().clear()
                
                if writer is not None:
                    writer.add_scalar("Loss/lowest_fid", lowest_fid, i)
                    writer.add_scalar("Loss/lowest_epoch", lowest_epoch, i)

if __name__ == "__main__":
    main()
