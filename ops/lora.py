import math
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """A modular LoRA layer encapsulating low-rank adaptation weights."""
    def __init__(self, qkv: nn.Module, r: int):
        super().__init__()
        self.qkv = qkv
        self.r = r
        self.dim = qkv.in_features

        # LoRA weights
        self.w_a = nn.Linear(self.dim, r, bias=False)
        self.w_b = nn.Linear(r, self.dim, bias=False)

        # Initialize parameters
        self._reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute original QKV
        qkv_output = self.qkv(x)  # Shape: (B, N, 3 * dim)

        # Compute low-rank adaptations
        x_lora = x.to(dtype=self.w_a.weight.dtype)
        lora_output = self.w_b(self.w_a(x_lora)).to(dtype=qkv_output.dtype)  # Shape: (B, N, dim)

        # Add the adaptation to the Q and V components of QKV
        qkv_output[:, :, :self.dim] += lora_output
        qkv_output[:, :, -self.dim:] += lora_output

        return qkv_output

    def _reset_parameters(self):
        """Initialize the parameters for the LoRA layer."""
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b.weight)

class AffineLoRALayer(LoRALayer):

    def __init__(self, qkv, r, dim):
        super().__init__(qkv, r)
        self.beta = nn.Linear(dim, r, bias=False)
        self.gamma = nn.Linear(dim, r, bias=False)
        self.rep = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute original QKV
        qkv_output = self.qkv(x)  # Shape: (B, N, 3 * dim)
        
        # Compute low-rank adaptations
        x_lora = x.to(dtype=self.w_a.weight.dtype)
        rep = self.rep.to(dtype=self.w_a.weight.dtype)
        lora_output = self.w_b(self.beta(rep) * self.w_a(x_lora) + self.gamma(rep)).to(dtype=qkv_output.dtype)  # Shape: (B, N, dim)
        # Add the adaptation to the Q and V components of QKV
        qkv_output[:, :, :self.dim] += lora_output
        qkv_output[:, :, -self.dim:] += lora_output

        return qkv_output

class LoRALayerQKV(nn.Module):
    """A modular LoRA layer encapsulating low-rank adaptation weights."""
    def __init__(self, qkv: nn.Module, r: int):
        super().__init__()
        self.qkv = qkv
        self.r = r
        self.dim = qkv.in_features

        # LoRA weights
        # Initialize parameters
        self.w_a_q = nn.Linear(self.dim, r, bias=False)
        self.w_b_q = nn.Linear(r, self.dim, bias=False)
        self._reset_parameters(self.w_a_q, self.w_b_q)

        self.w_a_k = nn.Linear(self.dim, r, bias=False)
        self.w_b_k = nn.Linear(r, self.dim, bias=False)
        self._reset_parameters(self.w_a_k, self.w_b_k)

        self.w_a_v = nn.Linear(self.dim, r, bias=False)
        self.w_b_v = nn.Linear(r, self.dim, bias=False)
        self._reset_parameters(self.w_a_v, self.w_b_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute original QKV
        qkv_output = self.qkv(x)  # Shape: (B, N, 3 * dim)

        # Compute low-rank adaptations
        x_lora = x.to(dtype=self.w_a_q.weight.dtype)
        lora_output_q = self.w_b_q(self.w_a_q(x_lora)).to(dtype=qkv_output.dtype)  # Shape: (B, N, dim)
        lora_output_k = self.w_b_k(self.w_a_k(x_lora)).to(dtype=qkv_output.dtype)  # Shape: (B, N, dim)
        lora_output_v = self.w_b_v(self.w_a_v(x_lora)).to(dtype=qkv_output.dtype)  # Shape: (B, N, dim)

        # Add the adaptation to the Q and V components of QKV
        qkv_output[:, :, :self.dim] += lora_output_q
        qkv_output[:, :, self.dim:-self.dim] += lora_output_k
        qkv_output[:, :, -self.dim:] += lora_output_v

        return qkv_output

    def _reset_parameters(self, w_a, w_b):
        """Initialize the parameters for the LoRA layer."""
        nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(w_b.weight)
