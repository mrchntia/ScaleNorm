import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Callable, Dict, List, Optional


act_funcs: Dict[str, Callable] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "mish": nn.Mish,
}


class Parameters:
    def __init__(self,
                 dataset: str = None,
                 batch_size: int = None,
                 act_func: str = "relu",
                 model_arch: str = None,
                 epochs: int = None,
                 target_epsilon: float = None,
                 grad_norm: float = None,
                 noise_mult: float = None,
                 privacy: bool = None,
                 scale_norm: bool = None,
                 norm_layer: str = None,
                 num_groups: int = None
                 ):
        self.dataset: str = dataset
        self.epochs: int = epochs
        self.batch_size: float = batch_size
        self.target_epsilon: float = target_epsilon
        self.grad_norm: float = grad_norm
        self.noise_mult: float = noise_mult
        self.model_arch: str = model_arch
        self.privacy: bool = privacy
        self.act_func: Callable = act_funcs[act_func]
        self.scale_norm: bool = scale_norm
        self.norm_layer: str = norm_layer
        self.num_groups: int = num_groups

        # Fixed
        self.max_batch_size = 16
        if self.dataset == "carvana":
            self.image_height = 160  # 1280 originally
            self.image_width = 240  # 1918 originally
            self.in_channels: int = 3
            self.out_channels: int = 1
        elif self.dataset == "pascal":
            self.image_height = 320
            self.image_width = 480
            self.in_channels: int = 3
            self.out_channels: int = 1
        elif self.dataset in "pancreas":
            self.image_height = 256  # 256, 512
            self.image_width = 256  # 256, 512
            self.in_channels: int = 1
            self.out_channels: int = 1
        elif self.dataset in "liver":
            self.image_height = 256
            self.image_width = 256
            self.in_channels: int = 1
            self.out_channels: int = 1

        self.learning_rate: float = 0.001
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_delta: float = 1e-5
        self.secure_rng: bool = False
        self.num_workers: int = 2
        self.seed: int = 1232
        self.optimizer: Callable = optim.NAdam
        self.final_epsilon: Optional[float] = None
        self.val_loss_list: Optional[List[float]] = None
        self.dice_score_list: Optional[List[float]] = None
