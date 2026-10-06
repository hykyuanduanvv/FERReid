import os


def _path(var, default):
    return os.path.expanduser(os.environ.get(var, default))


DOMAIN_CONFIG = {
    # default fold (override with --source_domains / --val_domains / --target_domains)
    'source_domains': ['market1501', 'msmt17', 'cuhksysu'],
    # held-out domain for model selection during training (never a target domain); none by default
    'val_domains': [],
    'target_domains': ['cuhk03'],
    'data_root': _path('FERREID_DATA_ROOT', '/root/autodl-tmp/reid-data'),
    'weights_dir': _path('FERREID_WEIGHTS_DIR', '/root/autodl-tmp/weights'),
}

# Leave-one-out protocol: each target domain is evaluated with a model trained on the *train splits* of
# the other three datasets; its own train split is only the unlabeled pool of the active module.
# CUHK-SYSU is always a source domain (no camera labels, so never a target here).
DATASETS = ['market1501', 'msmt17', 'cuhk03', 'cuhksysu']
TARGETS = ['market1501', 'msmt17', 'cuhk03']


def fold_sources(target):
    """Source domains of the fold whose target is `target`."""
    if target not in TARGETS:
        raise ValueError("target must be one of {}, got {}".format(TARGETS, target))
    return [d for d in DATASETS if d != target]


# Domains whose camera ids are not real cameras (CUHK-SYSU: street snaps / movie frames). There, pairs
# are any two images of the pool instead of two images from different cameras.
NO_CAMERA_DOMAINS = {'cuhksysu'}

# Visual backbones (select with --backbone; empty = DEFAULT_BACKBONE). Data pipelines always produce
# 256x128 images; a backbone whose input_size differs resizes inside the model.
DEFAULT_BACKBONE = 'vit_b16'
BACKBONES = {
    # timm ViT-B/16 (augreg2 in21k -> in1k), pos_embed interpolated to 16x8 patches
    # (scripts/prepare_weights.py --model vit_b16)
    'vit_b16': {'kind': 'timm', 'weights': os.path.join(DOMAIN_CONFIG['weights_dir'], 'vit_base_patch16_224.pth'),
                'input_size': (256, 128)},
    # DINOv2 ViT-B/14 (the original VICP backbone). Patch 14 -> 252x126 = 18x9 patches, the closest
    # multiple of 14 to the 2:1 pedestrian aspect; DINOv2 interpolates its 37x37 pos_embed itself.
    # Code: a local clone of facebookresearch/dinov2 (FERREID_DINOV2_REPO); weights: the official
    # dinov2_vitb14_pretrain.pth in the weights dir (scripts/prepare_weights.py --model dinov2_b14).
    'dinov2_b14': {'kind': 'dinov2', 'hub_name': 'dinov2_vitb14',
                   'repo': _path('FERREID_DINOV2_REPO', '/root/basic-models/dinov2'),
                   'weights': os.path.join(DOMAIN_CONFIG['weights_dir'], 'dinov2_vitb14_pretrain.pth'),
                   'input_size': (252, 126)},
    # CLIP ViT-B/16 image encoder (OpenAI, QuickGELU), pos_embed interpolated to 16x8 patches, 768-d CLS before the
    # projection; inputs re-normalised to the CLIP mean / std inside the wrapper (scripts/prepare_weights.py --model clip_b16)
    'clip_b16': {'kind': 'clip', 'arch': 'vit_base_patch16_clip_quickgelu_224',
                 'weights': os.path.join(DOMAIN_CONFIG['weights_dir'], 'vit_base_patch16_clip_quickgelu_openai.pth'),
                 'input_size': (256, 128)},
    # PASS (ECCV'22) LUPerson self-supervised ViT-S/16 and ViT-B/16 (CASIA-IVA-Lab/PASS-reID release, files renamed),
    # 256x128 = 16x8 patches; [PART] tokens dropped (CLS feature); inputs normalised with mean = std = 0.5 as in PASS
    'pass_vits': {'kind': 'clip', 'arch': 'vit_small_patch16_224', 'norm': 'half',
                  'weights': os.path.join(DOMAIN_CONFIG['weights_dir'], 'pass_vits_lup.pth'), 'input_size': (256, 128)},
    'pass_vitb': {'kind': 'clip', 'arch': 'vit_base_patch16_224', 'norm': 'half',
                  'weights': os.path.join(DOMAIN_CONFIG['weights_dir'], 'pass_vitb_lup.pth'), 'input_size': (256, 128)},
    # the same with the three pre-trained [PART] tokens kept (frozen) in the sequence
    'pass_vitb_p': {'kind': 'clip', 'arch': 'vit_base_patch16_224', 'norm': 'half', 'parts': True,
                    'weights': os.path.join(DOMAIN_CONFIG['weights_dir'], 'pass_vitb_lup.pth'), 'input_size': (256, 128)},
    # randomly initialised tiny ViT for the CPU tests (tests/*.py); never used for experiments
    'tiny_test': {'kind': 'timm_random', 'arch': 'vit_tiny_patch16_224', 'input_size': (256, 128)},
}
