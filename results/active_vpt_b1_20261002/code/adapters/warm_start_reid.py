"""Isolated VPT warm-start experiment. Production VICP defaults are unchanged.

Use scripts/train_vpt_context.py to train/evaluate these checkpoints: the generic
VICP loader does not know this subclass or its shared-delta control.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import torch
from adapters.args_reid import ReIDTrainingArguments
from adapters.reid_model import ReIDModel
from ops.losses import HardTripletLoss


@dataclass
class WarmStartArguments(ReIDTrainingArguments):
    warm_checkpoint: str = 'experiments/s2_vpt_long/checkpoint-12000'
    warm_variant: str = 'context'
    warm_eval_draws: int = 3
    warm_smoke_only: bool = False


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def state_hash(items):
    h = hashlib.sha256()
    for n, t in sorted(items):
        h.update((n + str(t.shape) + str(t.dtype) + tensor_hash(t)).encode())
    return h.hexdigest()


def load_exact(module, state, device):
    """Restore values AND checkpoint dtypes, including frozen FP32 LoRA tensors."""
    actual = module.state_dict()
    if set(actual) != set(state):
        raise ValueError(('state keys differ', set(actual) ^ set(state)))
    for n, p in module.named_parameters():
        assert p.shape == state[n].shape, n
        p.data = state[n].to(device).clone()
    for n, b in module.named_buffers():
        if n not in state:
            # Non-persistent buffers (e.g. RoPE frequencies) are reconstructed
            # by the constructor; state_dict deliberately excludes them.
            continue
        assert b.shape == state[n].shape, n
        b.data = state[n].to(device).clone()
    assert state_hash(module.state_dict().items()) == state_hash(state.items())


class FP32Triplet(HardTripletLoss):
    def forward(self, x, y):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float(), y)


class WarmStartReIDModel(ReIDModel):
    def __init__(self, args):
        if args.warm_variant not in ('context', 'shared'):
            raise ValueError(args.warm_variant)
        assert args.prompt_mode == 'residual' and args.ctx_center == 'none'
        super().__init__(args)
        # Both arms instantiate identical parameters and consume identical RNG.
        self.shared_delta = torch.nn.Parameter(torch.zeros_like(self.base_prompt))
        self.prompt_mlp.weight.data.zero_()
        self.ctx_gate.data.fill_(0.1)
        self.ctx_gate.requires_grad_(False)
        self.base_prompt.requires_grad_(False)
        self.encoder.requires_grad_(False)
        if self.reid_head is not None:
            self.reid_head.requires_grad_(False)
        self.shared_delta.requires_grad_(args.warm_variant == 'shared')
        if args.warm_variant == 'shared':
            self.prompt_mlp.requires_grad_(False)
            self.delta_norm.requires_grad_(False)
            self.query_embeddings.requires_grad_(False)
        # Q-Former keeps the same ICL objective in both arms. In the shared arm
        # it cannot affect retrieval: the prompt ignores all hidden states.
        self.loss = FP32Triplet(margin=args.triplet_margin, hardest=True)

    def _prompts_from_hidden(self, h):
        if self.args.warm_variant == 'shared':
            return (self.base_prompt.float() + self.shared_delta.float()).expand(h.size(0), -1, -1)
        return super()._prompts_from_hidden(h)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.encoder_copy.eval()
        self.lm.eval()
        if self.reid_head is not None:
            self.reid_head.eval()
        return self

    def restore_vpt(self, state, device):
        load_exact(self.encoder, {k[8:]: v for k, v in state.items() if k.startswith('encoder.')}, device)
        if self.reid_head is not None:
            load_exact(self.reid_head, {k[10:]: v for k, v in state.items() if k.startswith('reid_head.')}, device)
        self.base_prompt.data = state['prompt'].reshape_as(self.base_prompt).to(device).clone()
        # DGReIDTrainer converts frozen parameters to FP16; restore fixed gate
        # and unused branch tensors as FP32 so the two arms start identically.
        for n, p in self.named_parameters():
            if n.startswith(('ctx_gate', 'shared_delta', 'prompt_mlp.', 'delta_norm.', 'query_embeddings')):
                p.data = p.data.float()
        self.ctx_gate.data.fill_(0.1)

    def frozen_hash(self):
        items = [('encoder.' + k, v) for k, v in self.encoder.state_dict().items()]
        items += [('base_prompt', self.base_prompt), ('ctx_gate', self.ctx_gate)]
        if self.reid_head is not None:
            items += [('reid_head.' + k, v) for k, v in self.reid_head.state_dict().items()]
        return state_hash(items)
