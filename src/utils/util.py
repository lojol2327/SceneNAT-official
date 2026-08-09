from typing import *
from argparse import Namespace
from torch.nn import Module
from torch.optim import Optimizer

import os
import json
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch

# Copied from https://github.com/huggingface/pytorch-image-models/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def yield_forever(iterator: Iterator[Any]):
    while True:
        for x in iterator:
            yield x

def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

def save_experiment_params(args: Namespace, experiment_tag: str, directory: str) -> None:
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

def save_model_architecture(model: Module, directory: str) -> None:
    """Save the model architecture to a `.txt` file."""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f'Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n' + str(model)

    with open(os.path.join(directory, 'model.txt'), 'w') as f:
        f.write(message)
        
def load_checkpoints_for_test(
    model: Module,
    ckpt_dir: str,
    epoch: Optional[int]=None,
    get_last=False,
    get_best=True,
    dist_to_mono=False,
    device=torch.device("cpu")
    ) -> int:
    """Load checkpoint from the given experiment directory and return the epoch of this checkpoint."""
    if epoch is None:
        if get_last:
            checkpoint_path = os.path.join(ckpt_dir, "last_model.pth")
        elif get_best:    
            checkpoint_path = os.path.join(ckpt_dir, "best_model.pth")
        else:
            model_files = [f.split(".")[0] for f in os.listdir(ckpt_dir)
                if f.startswith("epoch_") and f.endswith(".pth")]

            epoch = epoch or max([int(f[6:]) for f in model_files])  # load the latest checkpoint by default
            checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth")
    else:
        checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pth")
    
    assert(os.path.exists(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}")  # checkpoint file not found
    print(f"Load checkpoint from {checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = checkpoint["epoch"]
    
    if dist_to_mono:
        state_dict = checkpoint["model"]
        # Remove "module." prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # Remove "module."
            else:
                new_state_dict[k] = v
        checkpoint["model"] = new_state_dict
    
    model.load_state_dict(checkpoint["model"])

    return epoch

def load_checkpoints(
    model: Module,
    ckpt_dir: str,
    optimizer: Optional[Optimizer]=None,
    scheduler=None,
    epoch: Optional[int]=None,
    get_last=False,
    device=torch.device("cpu")
    ) -> int:
    """Load checkpoint from the given experiment directory and return the epoch of this checkpoint."""
    best_epoch = best_loss = None
    
    if get_last:
        checkpoint_path = os.path.join(ckpt_dir, "last_model.pth")
        assert(os.path.exists(checkpoint_path))  # checkpoint file not found
        
        print(f"Load checkpoint from {checkpoint_path}\n")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        epoch = checkpoint["epoch"]
        
    else:
        model_files = [f.split(".")[0] for f in os.listdir(ckpt_dir)
            if f.startswith("epoch_") and f.endswith(".pth")]

        assert len(model_files) > 0, f"No checkpoint found in {ckpt_dir}!!! Check for resume.\n"

        epoch = epoch or max([int(f[6:]) for f in model_files])  # load the latest checkpoint by default
        checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pth")
    
        print(f"Load checkpoint from {checkpoint_path}\n")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler = checkpoint["scheduler"]
    
    best_checkpoint_path = os.path.join(ckpt_dir, "best_model.pth")
    if os.path.exists(best_checkpoint_path):
        print(f"Load best checkpoint from {best_checkpoint_path}\n")
        best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        if "loss" in best_checkpoint:
            best_epoch = best_checkpoint["epoch"]
            best_loss = best_checkpoint["loss"]
    
    return epoch+1, best_epoch or -1, best_loss or float("inf")

def save_checkpoint(
    model: Module, 
    optimizer: Optimizer, 
    ckpt_dir: str, 
    epoch: int, 
    scheduler=None,
    name: Optional[str]=None,
    loss=None
    ) -> None:
    """Save checkpoint to the given experiment directory."""
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    if scheduler is not None:
        save_dict["scheduler"] = scheduler
        
    if name is None:
        name = f"epoch_{epoch:05d}.pth"
    
    if loss is not None:
        save_dict["loss"] = loss

    save_path = os.path.join(ckpt_dir, name)
    torch.save(save_dict, save_path)