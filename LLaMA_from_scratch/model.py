from dataclasses import dataclass
from typing import Optional
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim:int=40
    n_layers:int=32
    n_kv_heads:Optional[int]=None
    vocab_size:int=-1
    multiple_of:int=256
    ffn_dim_multiplier:Optional[float]=None
    norm_eps:float=1e-5

    # For KV cache
    max_batch_size:int=32
    max_seq_len:int=2048
    
    device:str=None