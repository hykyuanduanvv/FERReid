from dataclasses import dataclass, field

import transformers


@dataclass
class ReIDTrainingArguments(transformers.TrainingArguments):
    # --- VICP model options (same names/defaults as VICP's train_vpt_lora.TrainingArguments)
    # the backbone comes from --backbone; this field is kept for models.Model's signature
    vision_model: str = field(default="vit_base_patch16_224")
    llm_model: str = field(default="Qwen/Qwen3-0.6B")
    num_id_tokens: int = field(default=32)     # Q-Former tokens per image pair
    num_vpt_tokens: int = field(default=32)    # visual prompt tokens per ViT layer
    num_icl_samples: int = field(default=64)   # ICL sequence length L (questions per sequence)
    num_icl_bs: int = field(default=1)         # ICL sequences (prompt sets) per forward
    icl_loss_weight: float = field(default=1.0)
    # features the ICL questions are built from: "frozen" = VICP's frozen encoder copy;
    # "trained" = the trained encoder (LoRA, no prompts), gradient stopped as in VICP
    icl_feature: str = field(default="frozen")
    ot_loss_weight: float = field(default=0.01)  # WPA; 0 disables it (and skips its computation)

    # "vicp" = VICP (LLM + in-context prompts); "plain" = same encoder, triplet only (baseline);
    # "vpt" = VICP with the LLM/context replaced by one learnable prompt (same encoder, prompts, losses)
    model_type: str = field(default="vicp")
    # visual backbone, a key of adapters/config_reid.py::BACKBONES (empty = DEFAULT_BACKBONE = vit_b16)
    backbone: str = field(default="")

    # --- how the encoder is trained (defaults = historical: LoRA r=128 on the last 4 blocks)
    train_backbone: str = field(default="lora")   # "lora" | "full" (every encoder weight trained, no LoRA)
    lora_layers: int = field(default=4)           # LoRA on the qkv of the last N blocks
    lora_rank: int = field(default=128)
    backbone_lr_mult: float = field(default=1.0)  # lr multiplier for encoder weights when train_backbone=full

    # --- losses (defaults = historical: hardest triplet on the normalized CLS, no ID loss)
    triplet_margin: float = field(default=0.1)
    ce_loss_weight: float = field(default=0.0)    # > 0: identity cross-entropy on source identities
    bnneck: bool = field(default=False)           # BNNeck: CE on BN(feature), retrieval with BN(feature)
    label_smoothing: float = field(default=0.1)
    num_train_ids: int = field(default=0)         # set by the trainer / read from the checkpoint

    # --- direction A: make the prompt depend on the context (defaults = original VICP)
    prompt_mode: str = field(default="vicp")      # "residual": prompt = base + gate * delta(context)
    ctx_center: str = field(default="none")       # "ema": delta sees h minus a running mean of h
    ctx_gate_init: float = field(default=0.1)     # initial per-layer gate of the context term
    delta_init_std: float = field(default=0.02)   # init std of prompt_mlp in residual mode
    episode_context_ids: int = field(default=0)   # > 0: first N identities of a batch = context only
    episode_context_ids_min: int = field(default=0)  # > 0: N drawn uniformly from [min, episode_context_ids]
    pseudo_domains: str = field(default="none")   # "camera_pair": a training "domain" = (dataset, camera pair)
    pseudo_min_ids: int = field(default=8)        # camera pairs with fewer identities are dropped

    # --- training sampler (defaults = historical)
    instances_per_id: int = field(default=2)      # K images per identity (even; batch = P ids x K images)
    cross_camera_instances: bool = field(default=False)  # draw K images spanning >= 2 cameras when possible
    unique_ids_per_batch: bool = field(default=False)    # historical sampler draws identities with replacement
    batch_domain_mode: str = field(default="single")     # "single": every batch from one source domain
    #                                                       "mixed": identities from all source domains

    # --- domains (comma-separated; empty = use adapters/config_reid.py; "none" = no validation)
    source_domains: str = field(default="")
    val_domains: str = field(default="")
    target_domains: str = field(default="")
    # train on train+query+gallery of each source domain (torchreid combineall). Protocol-2 uses False.
    source_all_images: bool = field(default=True)

    # --- context evaluation
    # context_k is the labeling budget. num_icl_samples stays equal to the training value at test time.
    context_k: int = field(default=16)
    context_method: str = field(default="random")
    # "image" (label-free): the selector sees only unlabeled images (paths, camera ids) and picks k
    #     anchor images; simulated annotation pairs each anchor with another image of the same person
    #     from a different camera (any other image in NO_CAMERA_DOMAINS). Failed / duplicate anchors
    #     still consume budget.
    # "identity" (historical): the selector picks k identities from the pid-grouped pool.
    selection_unit: str = field(default="image")
    eval_seeds: int = field(default=1)
    eval_splits: int = field(default=10)
    # validation only: evaluate on a fixed random subset of this many test identities (0 = all)
    val_max_ids: int = field(default=500)
    eval_num_workers: int = field(default=8)
