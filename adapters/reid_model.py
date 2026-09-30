import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models import Model
from adapters.config_reid import BACKBONES, DEFAULT_BACKBONE

_VIT_B16_LOCAL = BACKBONES["vit_b16"]["weights"]


class TimMViTWrapper(nn.Module):
    """
    Wraps timm ViT-B/16 (patch=16) to expose the DINOv2 interface VICP expects.
    Uses @property for blocks/norm/patch_embed so they don't register as duplicate
    submodules in the state_dict.
    """

    def __init__(self, img_size=(256, 128), weights=_VIT_B16_LOCAL):
        super().__init__()
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Missing pretrained ViT weights: {weights}. "
                "Run scripts/prepare_weights.py --model vit_b16 or set FERREID_WEIGHTS_DIR. "
                "See docs/DEPLOYMENT.md; random initialization is not the recorded experiment."
            )
        print(f"[ReIDModel] Loading ViT-B/16 from {weights}")
        self._vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False,
            img_size=img_size, num_classes=0,
        )
        state = torch.load(weights, map_location="cpu")
        self._vit.load_state_dict(state, strict=True)  # fail loudly instead of silently random-init

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


class Dinov2Wrapper(nn.Module):
    """DINOv2 ViT-B/14 for 2:1 pedestrian crops.

    The data pipelines produce 256x128 images (ViT-B/16 convention); patch 14 needs multiples
    of 14, so inputs are resized to input_size (default 252x126 = 18x9 patches) here. DINOv2
    interpolates its 37x37 position embedding to the patch grid on every forward.
    Exposes the same interface as TimMViTWrapper / the hub model (blocks, norm, patch_embed,
    embed_dim, prepare_tokens_with_masks, forward_features).
    """

    def __init__(self, repo, hub_name, weights, input_size=(252, 126)):
        super().__init__()
        if not os.path.isdir(repo):
            raise FileNotFoundError(
                f"Missing DINOv2 source repository: {repo}. Clone facebookresearch/dinov2 there or set "
                "FERREID_DINOV2_REPO (docs/DEPLOYMENT.md).")
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Missing DINOv2 weights: {weights}. Run scripts/prepare_weights.py --model dinov2_b14 "
                "or set FERREID_WEIGHTS_DIR; random initialization is not the recorded experiment.")
        print(f"[ReIDModel] Loading DINOv2 {hub_name} from {weights} (code {repo}), input {input_size}")
        self._vit = torch.hub.load(repo, hub_name, source="local", pretrained=False)
        self._vit.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        self.input_size = tuple(input_size)

    def _resize(self, x):
        if tuple(x.shape[-2:]) == self.input_size:
            return x
        return F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False, antialias=True)

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

    def prepare_tokens_with_masks(self, x, masks=None):
        return self._vit.prepare_tokens_with_masks(self._resize(x), masks)

    def forward_features(self, x):
        return self._vit.forward_features(self._resize(x))

    def forward(self, x):
        return self.forward_features(x)["x_norm_clstoken"]


def backbone_name(args):
    return getattr(args, "backbone", "") or DEFAULT_BACKBONE


def load_backbone(name):
    """Build the visual encoder named in adapters/config_reid.py::BACKBONES."""
    cfg = BACKBONES[name]
    if cfg["kind"] == "timm":
        return TimMViTWrapper(img_size=tuple(cfg["input_size"]), weights=cfg["weights"])
    if cfg["kind"] == "dinov2":
        return Dinov2Wrapper(cfg["repo"], cfg["hub_name"], cfg["weights"], cfg["input_size"])
    raise ValueError("unknown backbone kind: {}".format(cfg["kind"]))


from ops.adapt import adapt_encoder  # noqa: E402,F401  (re-exported; defined in ops to avoid an import cycle)


# Arguments that change the model's parameters / forward. Evaluation code copies them from the
# checkpoint's saved training_args.bin so a checkpoint is always rebuilt with its own architecture.
STRUCTURAL_FIELDS = ["model_type", "backbone", "train_backbone", "lora_layers", "lora_rank",
                     "num_vpt_tokens", "num_id_tokens", "num_icl_bs", "icl_feature", "llm_model",
                     "bnneck", "ce_loss_weight", "label_smoothing", "num_train_ids"]


def apply_checkpoint_structure(args, checkpoint_dir):
    """Copy STRUCTURAL_FIELDS from <checkpoint>/training_args.bin into args (with a printed note for
    every value that changes). Checkpoints older than a field keep that field's default."""
    path = os.path.join(checkpoint_dir, "training_args.bin")
    if not os.path.isfile(path):
        print("[checkpoint] no training_args.bin in {}; using the command-line model options".format(checkpoint_dir))
        return args
    saved = torch.load(path, map_location="cpu", weights_only=False)
    defaults = type(args)  # dataclass defaults for fields the old checkpoint does not have
    for name in STRUCTURAL_FIELDS:
        value = getattr(saved, name, getattr(defaults, name, None))
        if name == "num_train_ids" and not value:
            value = peek_num_train_ids(checkpoint_dir)
        if getattr(args, name, None) != value:
            print("[checkpoint] {} = {!r} (command line had {!r})".format(name, value, getattr(args, name, None)))
            setattr(args, name, value)
    return args


def peek_num_train_ids(checkpoint_dir):
    """Number of training identities of a checkpoint with an ID-classification head (0 if none),
    so evaluation code can rebuild the head with the right shape without loading the source data."""
    path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in state.items():
        if key.endswith("reid_head.classifier.weight"):
            return int(value.shape[0])
    return 0


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
    """VICP Model with a configurable pedestrian backbone (ViT-B/16 256x128 or DINOv2 ViT-B/14 252x126)."""

    def __init__(self, args):
        super().__init__(args)
        if getattr(args, "icl_feature", "frozen") == "trained":
            self.encoder_copy = _TrainedEncoderView(self.encoder)  # frees the frozen copy
        print("[ReIDModel] backbone:", backbone_name(args),
              "| ICL question features:", getattr(args, "icl_feature", "frozen"))

    def _load_dinov2(self, _model_name: str):
        # called by models.Model.__init__ (after self.args is set) for encoder and encoder_copy;
        # the backbone comes from --backbone / config, not from args.vision_model
        return load_backbone(backbone_name(self.args))
