"""Plain ViT ReID baseline: VICP's encoder setup without the LLM / context / prompts.

Same backbone (ViT-B/16, 256x128), same trainable part (LoRA r=128 on the qkv of the
last 4 blocks), same HardTripletLoss as VICP's id_loss. Used to check how much of
VICP's accuracy actually comes from in-context prompting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.reid_model import ReIDModel
from ops.lora import LoRALayerQKV
from ops.losses import HardTripletLoss


class PlainReIDModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        encoder = ReIDModel._load_dinov2(args.vision_model).eval()
        encoder.requires_grad_(False)
        for block in encoder.blocks[-4:]:  # same LoRA placement / rank as models.Model
            block.attn.qkv = LoRALayerQKV(block.attn.qkv, r=128)
        self.encoder = encoder
        self.loss = HardTripletLoss(margin=0.1, hardest=True)

    def forward(self, image_crops, labels=None, prompts=None):
        # prompts is accepted and ignored so evaluation code can call both models alike
        encoder_dtype = self.encoder.patch_embed.proj.weight.dtype
        if image_crops.ndim == 5:
            bs, nview, nc, h, w = image_crops.size()
            image_crops = image_crops.reshape(-1, nc, h, w)
        x = self.encoder.forward_features(image_crops.to(dtype=encoder_dtype))["x_norm_clstoken"]
        x = F.normalize(x, dim=-1)
        std = x.std(dim=0).mean()
        zero = torch.tensor(0.0, device=x.device)
        if labels is not None:
            labels = labels.unsqueeze(1).expand(-1, nview).reshape(-1)
            id_loss = self.loss(x, labels)
        else:
            id_loss = zero
        return {
            "loss": id_loss,
            "id_loss": id_loss,
            "ot_loss": zero,
            "icl_loss": zero,
            "features": x,
            "prompts": None,
            "std": std,
        }
