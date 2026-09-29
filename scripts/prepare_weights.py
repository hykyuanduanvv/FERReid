"""Prepare the named timm ViT-B/16 pretrained model for 256x128 inputs."""
import argparse
import hashlib
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=os.environ.get('FERREID_WEIGHTS_DIR', 'weights'))
    args = parser.parse_args()
    out = Path(args.output_dir).expanduser() / 'vit_base_patch16_224.pth'
    if out.exists():
        raise SystemExit(f'Refusing to overwrite existing weights: {out}')
    import timm
    import torch
    model = timm.create_model(
        'vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True,
        img_size=(256, 128), num_classes=0,
    )
    # timm interpolates the pretrained position embedding to the 16x8 patch grid.
    verifier = timm.create_model('vit_base_patch16_224', pretrained=False,
                                 img_size=(256, 128), num_classes=0)
    verifier.load_state_dict(model.state_dict(), strict=True)
    assert tuple(model.pos_embed.shape) == (1, 129, 768)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    with out.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(f'{out}\nSHA256 {digest}')
    print('For historical checkpoint reproduction, prefer the recorded original weight file; '
          'download/cache revisions and serialization can change this file hash.')


if __name__ == '__main__':
    main()
