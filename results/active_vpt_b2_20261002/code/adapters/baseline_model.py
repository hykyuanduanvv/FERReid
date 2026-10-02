"""Baselines sharing VICP's encoder setup but not its LLM / context path.

  PlainReIDModel (--model_type plain): backbone + LoRA (or full fine-tuning), triplet only.
      Used to check how much of VICP's accuracy comes from prompts at all.
  VPTReIDModel   (--model_type vpt):   VICP with the LLM / context replaced by one learnable prompt.
      Used to check whether in-context prompting adds anything over a fixed learned prompt.

Both use the same options as models.Model (see adapters/args_reid.py): --backbone, --train_backbone,
--lora_layers, --lora_rank, --triplet_margin, --ce_loss_weight / --bnneck (BNNeck + ID loss head).
With the defaults they are the historical models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.reid_head import build_head
from adapters.reid_model import load_backbone, backbone_name
from ops.adapt import adapt_encoder
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


class PlainReIDModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = _build_encoder(args)
        self.loss = HardTripletLoss(margin=getattr(args, "triplet_margin", 0.1), hardest=True)
        self.reid_head = build_head(args, self.encoder.embed_dim)

    def forward(self, image_crops, labels=None, prompts=None):
        # prompts is accepted and ignored so evaluation code can call both models alike
        encoder_dtype = self.encoder.patch_embed.proj.weight.dtype
        if image_crops.ndim == 5:
            bs, nview, nc, h, w = image_crops.size()
            image_crops = image_crops.reshape(-1, nc, h, w)
        raw = self.encoder.forward_features(image_crops.to(dtype=encoder_dtype))["x_norm_clstoken"]
        x = F.normalize(raw, dim=-1)
        std = x.std(dim=0).mean()
        zero = torch.tensor(0.0, device=x.device)
        if labels is not None:
            labels = labels.unsqueeze(1).expand(-1, nview).reshape(-1)
            id_loss = self.loss(x, labels)
        else:
            id_loss = zero
        features, ce_loss = _head_outputs(self, raw, labels, zero)
        loss = id_loss + ce_loss * getattr(self.args, "ce_loss_weight", 0.0)
        return {
            "loss": loss,
            "id_loss": id_loss,
            "ce_loss": ce_loss,
            "ot_loss": zero,
            "icl_loss": zero,
            "features": features,
            "prompts": None,
            "std": std,
        }


class VPTReIDModel(nn.Module):
    """VICP without the LLM / context: one learnable visual prompt shared by every image.

    Identical to models.Model except where the prompt comes from: same backbone, same encoder
    training (LoRA r=128 on the qkv of the last 4 blocks by default), same deep-prompt insertion
    (num_vpt_tokens per layer, all layers), same losses (triplet + ot_loss_weight * WPA, plus the
    optional BNNeck/ID head). The prompt starts at zero, like the zero-initialised prompt_mlp
    output in VICP.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = _build_encoder(args)
        self.loss = HardTripletLoss(margin=getattr(args, "triplet_margin", 0.1), hardest=True)
        self.num_layers = len(self.encoder.blocks)
        self.hidden_size = self.encoder.embed_dim
        self.prompt = nn.Parameter(torch.zeros(1, self.num_layers, args.num_vpt_tokens, self.hidden_size))
        self.reid_head = build_head(args, self.hidden_size)

    def forward(self, image_crops, labels=None, prompts=None):
        encoder_dtype = self.encoder.patch_embed.proj.weight.dtype
        if image_crops.ndim == 5:
            bs, nview, nc, h, w = image_crops.size()
            image_crops = image_crops.reshape(-1, nc, h, w)
        image_crops = image_crops.to(dtype=encoder_dtype)
        if prompts is None:
            prompts = self.prompt  # the context (if any) is ignored
        # same insertion as models.Model.forward
        prompts = prompts.reshape(prompts.size(0), self.num_layers, self.args.num_vpt_tokens, -1)
        prompts = prompts[torch.randint(0, prompts.size(0), (image_crops.size(0),))]
        x = self.encoder.prepare_tokens_with_masks(image_crops, None)
        prompts = prompts.to(dtype=x.dtype)
        for i, blk in enumerate(self.encoder.blocks):
            prompts_ = prompts[:, i]
            if i == 0:
                x = torch.cat([x[:, 0].unsqueeze(1), prompts_, x[:, 1:]], dim=1)
            else:
                x = torch.cat([x[:, 0].unsqueeze(1), prompts_, x[:, 1 + prompts_.size(1):]], dim=1)
            x = blk(x)
        x = self.encoder.norm(x)
        patch_features = x[:, 1 + prompts_.size(1):]
        raw = x[:, 0]
        x = F.normalize(raw, dim=-1)
        std = x.std(dim=0).mean()
        zero = torch.tensor(0.0, device=x.device)
        ot_loss = zero
        if labels is not None:
            labels = labels.unsqueeze(1).expand(-1, nview).reshape(-1)
            id_loss = self.loss(x, labels)
            if self.args.ot_loss_weight != 0:
                from models import Model
                from ops.wpa import compute_wpa
                all_pairs, all_labels = Model.sample_pair(None, labels)  # uses no instance state
                ot_loss = compute_wpa(patch_features[all_pairs[:, 0]], patch_features[all_pairs[:, 1]],
                                      all_labels.to(x.device))
        else:
            id_loss = zero
        features, ce_loss = _head_outputs(self, raw, labels, zero)
        loss = id_loss + ot_loss * self.args.ot_loss_weight + ce_loss * getattr(self.args, "ce_loss_weight", 0.0)
        return {"loss": loss, "id_loss": id_loss, "ce_loss": ce_loss, "ot_loss": ot_loss, "icl_loss": zero,
                "features": features, "prompts": prompts, "std": std}
