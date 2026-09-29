import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import torch
import torch.nn as nn
import timm
from models import Model
from adapters.config_reid import DOMAIN_CONFIG

_WEIGHTS_DIR = DOMAIN_CONFIG["weights_dir"]
_VIT_B16_LOCAL = os.path.join(_WEIGHTS_DIR, "vit_base_patch16_224.pth")


class TimMViTWrapper(nn.Module):
    """
    Wraps timm ViT-B/16 (patch=16) to expose the DINOv2 interface VICP expects.
    Uses @property for blocks/norm/patch_embed so they don't register as duplicate
    submodules in the state_dict.
    """

    def __init__(self, img_size=(256, 128), pretrained=True):
        super().__init__()
        if os.path.isfile(_VIT_B16_LOCAL):
            print(f"[ReIDModel] Loading ViT-B/16 from {_VIT_B16_LOCAL}")
            self._vit = timm.create_model(
                "vit_base_patch16_224", pretrained=False,
                img_size=img_size, num_classes=0,
            )
            state = torch.load(_VIT_B16_LOCAL, map_location="cpu")
            self._vit.load_state_dict(state, strict=True)  # fail loudly instead of silently random-init
        else:
            raise FileNotFoundError(
                f"Missing pretrained ViT weights: {_VIT_B16_LOCAL}. "
                "Run scripts/prepare_weights.py or set FERREID_WEIGHTS_DIR. "
                "See docs/DEPLOYMENT.md; random initialization is not the recorded experiment."
            )

    # Expose DINOv2-compatible attributes as properties (no duplicate submodule registration)
    @property
    def blocks(self):
        return self._vit.blocks

    @property
    def norm(self):
        return self._vit.norm

    @property
    def patch_embed(self):
        return self._vit.patch_embed

    @property
    def embed_dim(self):
        return self._vit.embed_dim

    def prepare_tokens_with_masks(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        x = self._vit.patch_embed(x)
        x = self._vit._pos_embed(x)
        x = self._vit.norm_pre(x)
        return x

    def forward_features(self, x: torch.Tensor) -> dict:
        out = self._vit.forward_features(x)   # (B, N+1, D), CLS at 0, normed
        return {"x_norm_clstoken": out[:, 0]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["x_norm_clstoken"]


class _TrainedEncoderView(nn.Module):
    """Stand-in for encoder_copy: question features from the trained encoder (LoRA, no prompts).

    The encoder is held in a plain list so it is not registered twice (no duplicate
    parameters in the state_dict). models.Model.forward calls it under torch.no_grad(),
    so icl_loss does not backpropagate into the encoder.
    """

    def __init__(self, encoder):
        super().__init__()
        self._encoder = [encoder]

    def forward_features(self, x):
        return self._encoder[0].forward_features(x)


class ReIDModel(Model):
    """VICP Model with DINOv2-vitb14 replaced by ViT-B/16 for pedestrian ReID (256x128)."""

    def __init__(self, args):
        super().__init__(args)
        if getattr(args, "icl_feature", "frozen") == "trained":
            self.encoder_copy = _TrainedEncoderView(self.encoder)  # frees the frozen copy
        print("[ReIDModel] ICL question features:", getattr(args, "icl_feature", "frozen"))

    @staticmethod
    def _load_dinov2(_model_name: str):
        pretrained = os.path.isfile(_VIT_B16_LOCAL)
        return TimMViTWrapper(img_size=(256, 128), pretrained=pretrained)
