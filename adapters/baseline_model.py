"""Source-trained base models (the "DG base" the active module is plugged into).

  PlainReIDModel (--model_type plain): backbone + LoRA (or full fine-tuning), triplet (+ optional BNNeck/ID).
  VPTReIDModel   (--model_type vpt):   the same encoder with one learnable deep visual prompt shared by every
                                       image. Its prompt is the base prompt that the active module adapts
                                       to a target domain (adapters/active/prompt_tuning.py).

Options (adapters/args_reid.py): --backbone, --train_backbone, --lora_layers, --lora_rank, --triplet_margin,
--ce_loss_weight / --bnneck (BNNeck + ID loss head), --ot_loss_weight (WPA local alignment).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.backbones import load_backbone, backbone_name
from adapters.reid_head import build_head
from ops.adapt import adapt_encoder
from ops.deep_prompt import insert_deep_prompts, sample_pair
from ops.losses import HardTripletLoss


def _build_encoder(args):
    return adapt_encoder(load_backbone(backbone_name(args)).eval(), args)


def _head_outputs(model, raw, labels, zero):
    """Retrieval feature and CE loss (0 when no head / not training)."""
    if model.reid_head is None:
        return F.normalize(raw, dim=-1), zero
    head = model.reid_head(raw, labels, compute_ce=labels is not None and model.training
                           and getattr(model.args, "ce_loss_weight", 0.0) > 0)
    return head["feat"], head.get("ce", zero)


def _flatten(image_crops, labels, dtype):
    nview = 1
    if image_crops.ndim == 5:
        bs, nview, nc, h, w = image_crops.size()
        image_crops = image_crops.reshape(-1, nc, h, w)
    if labels is not None:
        labels = labels.unsqueeze(1).expand(-1, nview).reshape(-1)
    return image_crops.to(dtype=dtype), labels


class PlainReIDModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = _build_encoder(args)
        self.loss = HardTripletLoss(margin=getattr(args, "triplet_margin", 0.1), hardest=True)
        self.reid_head = build_head(args, self.encoder.embed_dim)

    def forward(self, image_crops, labels=None, prompts=None, domains=None):
        # prompts is accepted and ignored so evaluation code can call every model alike
        image_crops, labels = _flatten(image_crops, labels, self.encoder.patch_embed.proj.weight.dtype)
        raw = self.encoder.forward_features(image_crops)["x_norm_clstoken"]
        x = F.normalize(raw, dim=-1)
        zero = torch.tensor(0.0, device=x.device)
        id_loss = self.loss(x, labels) if labels is not None else zero
        features, ce_loss = _head_outputs(self, raw, labels, zero)
        return {"loss": id_loss + ce_loss * getattr(self.args, "ce_loss_weight", 0.0), "id_loss": id_loss,
                "ce_loss": ce_loss, "ot_loss": zero, "icl_loss": zero, "features": features, "prompts": None,
                "std": x.std(dim=0).mean()}


class VPTReIDModel(nn.Module):
    """One learnable deep visual prompt (1, L, V, D) shared by every image, zero-initialised.

    forward(x, prompts=P) uses P instead of the learned prompt; P may carry more tokens per layer than
    --num_vpt_tokens (base prompt + appended domain tokens)."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = _build_encoder(args)
        self.loss = HardTripletLoss(margin=getattr(args, "triplet_margin", 0.1), hardest=True)
        self.num_layers = len(self.encoder.blocks)
        self.hidden_size = self.encoder.embed_dim
        self.prompt = nn.Parameter(torch.zeros(1, self.num_layers, args.num_vpt_tokens, self.hidden_size))
        self.reid_head = build_head(args, self.hidden_size)

    def forward(self, image_crops, labels=None, prompts=None, domains=None):
        image_crops, labels = _flatten(image_crops, labels, self.encoder.patch_embed.proj.weight.dtype)
        raw, patch = insert_deep_prompts(self.encoder, image_crops, self.prompt if prompts is None else prompts,
                                         self.num_layers)
        x = F.normalize(raw, dim=-1)
        zero = torch.tensor(0.0, device=x.device)
        id_loss = ot_loss = zero
        if labels is not None:
            id_loss = self.loss(x, labels)
            if self.args.ot_loss_weight != 0:
                from ops.wpa import compute_wpa
                pairs, pair_labels = sample_pair(labels)
                ot_loss = compute_wpa(patch[pairs[:, 0]], patch[pairs[:, 1]], pair_labels.to(x.device))
        features, ce_loss = _head_outputs(self, raw, labels, zero)
        loss = id_loss + ot_loss * self.args.ot_loss_weight + ce_loss * getattr(self.args, "ce_loss_weight", 0.0)
        return {"loss": loss, "id_loss": id_loss, "ce_loss": ce_loss, "ot_loss": ot_loss, "icl_loss": zero,
                "features": features, "prompts": self.prompt if prompts is None else prompts,
                "std": x.std(dim=0).mean()}


def build_model(args):
    """Model of --model_type (vpt | plain | vicp), freshly initialised."""
    if args.model_type == "vpt":
        return VPTReIDModel(args)
    if args.model_type == "plain":
        return PlainReIDModel(args)
    if args.model_type == "vicp":
        from adapters.vicp_model import VICPModel
        return VICPModel(args)
    raise ValueError("--model_type must be vpt, plain or vicp, got {}".format(args.model_type))


def load_checkpoint_model(args, device, checkpoint):
    """Rebuild a checkpoint's own architecture (training_args.bin), load its weights strictly, freeze it and
    use the trainer's dtype layout (frozen weights in --fp16/--bf16, trainable ones in FP32)."""
    import os
    from adapters.backbones import apply_checkpoint_structure
    apply_checkpoint_structure(args, checkpoint)
    model = build_model(args)
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model.to(dtype=dtype, device=device)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.to(dtype=torch.float32)
    state = None
    for name in ("pytorch_model.bin", "model.safetensors"):
        path = os.path.join(checkpoint, name)
        if os.path.isfile(path):
            if name.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(path, device=str(device))
            else:
                state = torch.load(path, map_location=device, weights_only=True)
            break
    if state is None:
        raise FileNotFoundError("no pytorch_model.bin / model.safetensors in {}".format(checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError("checkpoint {} does not match --model_type {} (missing {}, unexpected {}; e.g. {})".format(
            checkpoint, args.model_type, len(missing), len(unexpected), (missing + unexpected)[:3]))
    for p in model.parameters():
        p.requires_grad_(False)
    print("[checkpoint] {} ({}, {})".format(checkpoint, args.model_type, backbone_name(args)))
    return model.eval()
