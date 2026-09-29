import os


DOMAIN_CONFIG = {
    # training only
    'source_domains': ['market1501', 'msmt17'],
    # held-out source domain for model selection during training (never a target domain)
    'val_domains': ['cuhk03'],
    # test only; never used for training or model selection
    'target_domains': ['viper', 'grid', 'ilids', 'prid2011'],
    'data_root': os.path.expanduser(os.environ.get('FERREID_DATA_ROOT', '/root/autodl-tmp/reid-data')),
    'weights_dir': os.path.expanduser(os.environ.get('FERREID_WEIGHTS_DIR', '/root/autodl-tmp/weights')),
}

# number of official random splits per domain; domains not listed have a single fixed split
NUM_SPLITS = {
    'viper': 10,  # splits.json holds 20: 10-19 are 0-9 with cam_a/cam_b swapped
    'grid': 10,
    'ilids': 10,
    'prid2011': 10,
}
