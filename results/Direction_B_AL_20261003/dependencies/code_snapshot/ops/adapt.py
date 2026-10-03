"""How the pretrained visual encoder is trained (shared by the VICP, VPT and plain models)."""
from ops.lora import LoRALayerQKV


def adapt_encoder(encoder, args):
    """Freeze the pretrained encoder, then make its trainable part follow --train_backbone:
      lora  (default, historical): LoRA of rank --lora_rank on the qkv of the last --lora_layers blocks
      full  : every encoder parameter is trained (no LoRA); lr scaled by --backbone_lr_mult
    The defaults (lora, 4 layers, rank 128) reproduce the historical models exactly."""
    encoder.requires_grad_(False)
    mode = getattr(args, "train_backbone", "lora")
    if mode == "full":
        encoder.requires_grad_(True)
        return encoder
    if mode != "lora":
        raise ValueError("--train_backbone must be lora or full, got {}".format(mode))
    n = getattr(args, "lora_layers", 4)
    for block in encoder.blocks[-n:]:
        block.attn.qkv = LoRALayerQKV(block.attn.qkv, r=getattr(args, "lora_rank", 128))
    return encoder
