import os


def _path(var, default):
    return os.path.expanduser(os.environ.get(var, default))


DOMAIN_CONFIG = {
    # training only
    'source_domains': ['market1501', 'msmt17'],
    # held-out source domain for model selection during training (never a target domain)
    'val_domains': ['cuhk03'],
    # test only; never used for training or model selection (override with --target_domains)
    'target_domains': ['viper', 'grid', 'ilids', 'prid2011'],
    'data_root': _path('FERREID_DATA_ROOT', '/root/autodl-tmp/reid-data'),
    'weights_dir': _path('FERREID_WEIGHTS_DIR', '/root/autodl-tmp/weights'),
}

# number of official random splits per domain; domains not listed have a single fixed split
NUM_SPLITS = {
    'viper': 10,  # splits.json holds 20: 10-19 are 0-9 with cam_a/cam_b swapped
    'grid': 10,
    'ilids': 10,
    'prid2011': 10,
}

# Domains whose camera ids are not real cameras. i-LIDS: torchreid uses the image index as camid;
# CUHK-SYSU: street snaps / movie frames, no camera labels. For these, simulated annotation pairs
# an anchor with *another image* of the same person instead of an image from another camera.
NO_CAMERA_DOMAINS = {'ilids', 'cuhksysu'}

# DG-ReID Protocol-2 (leave-one-out over four large datasets): train on the *train split* of three,
# test on the query/gallery of the fourth; the fourth's train split is only the unlabeled context pool.
PROTOCOL2_DOMAINS = ['market1501', 'msmt17', 'cuhksysu', 'cuhk03']

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
}
