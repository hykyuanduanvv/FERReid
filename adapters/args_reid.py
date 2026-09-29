from dataclasses import dataclass, field

import transformers


@dataclass
class ReIDTrainingArguments(transformers.TrainingArguments):
    # --- VICP model options (same names/defaults as VICP's train_vpt_lora.TrainingArguments)
    # ReIDModel always builds ViT-B/16 at 256x128; this field is kept for models.Model's signature
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
    ot_loss_weight: float = field(default=0.01)

    # "vicp" = VICP (LLM + in-context prompts); "plain" = same ViT + LoRA, id_loss only (baseline)
    model_type: str = field(default="vicp")

    # --- domains (comma-separated; empty = use adapters/config_reid.py)
    source_domains: str = field(default="")
    val_domains: str = field(default="")
    # train on train+query+gallery of each source domain (torchreid combineall)
    source_all_images: bool = field(default=True)

    # --- context evaluation
    # context_k is the labeling budget: identities whose cross-camera pairs fill the L slots.
    # num_icl_samples stays equal to the training value at test time.
    context_k: int = field(default=16)
    context_method: str = field(default="random")
    eval_seeds: int = field(default=1)
    eval_splits: int = field(default=10)
    # validation only: evaluate on a fixed random subset of this many test identities (0 = all)
    val_max_ids: int = field(default=500)
    eval_num_workers: int = field(default=8)
