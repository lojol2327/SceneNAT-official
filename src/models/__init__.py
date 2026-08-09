from typing import *
from torch import Tensor
from torch.optim import Optimizer
from functools import partial
from torch import optim
import math

from .objfeat_vqvae import ObjectFeatureVQVAE
from .clip_encoders import *

from .scene_nat_baseline import *
from .scene_nat_wo_triplet import *
from .scene_nat_full_edge import *
from .scene_nat_triplet import *

def optimizer_from_config(config: Dict[str, Any], params: Iterable[Tensor]) -> Optimizer:
    name = config["name"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.)
    kwargs = dict(lr=lr, weight_decay=weight_decay)
    betas = config.get("betas", (0.9, 0.999))

    optimizer = {"Adam": partial(optim.Adam, **kwargs),
                 "adamw": partial(optim.AdamW, betas=betas, **kwargs),}[name]

    return optimizer(params)

def scheduler_from_config(config: Dict[str, Any], optimizer: Optimizer, total_epochs: int):
    """Create scheduler from config"""
    name = config["name"]
    
    if name == "warmup_cosine":
        warmup_epochs = config.get("warmup_epochs", 10)
        min_lr = float(config.get("min_lr", 1e-6))
        return WarmupCosineScheduler(optimizer, warmup_epochs, total_epochs, min_lr)
    else:
        raise ValueError(f"Unknown scheduler: {name}")

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing Scheduler"""
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch: int):
        """Update learning rate for given epoch"""
        if epoch < self.warmup_epochs:
            # Warmup: linear increase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_last_lr(self):
        """Get current learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]

def adjust_learning_rate(lr_schedules, optimizer, epoch):
    if (type(lr_schedules)==list):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
    else:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules.get_learning_rate(epoch)

class StepLearningRateSchedule():
    def __init__(self, specs):
        print(specs)
        self.initial = specs['initial']
        self.interval = specs['interval']
        self.factor = specs['factor']
        assert self.initial != 0

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))