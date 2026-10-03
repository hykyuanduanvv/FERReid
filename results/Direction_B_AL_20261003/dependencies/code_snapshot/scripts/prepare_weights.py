"""Prepare pretrained backbone weights in $FERREID_WEIGHTS_DIR.

  --model vit_b16    : timm vit_base_patch16_224.augreg2_in21k_ft_in1k adapted to 256x128
                       -> vit_base_patch16_224.pth
  --model dinov2_b14 : official DINOv2 ViT-B/14 weights -> dinov2_vitb14_pretrain.pth
                       (downloaded from dl.fbaipublicfiles.com, or copied with --from-file, e.g. from
                       ~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth)
Refuses to overwrite existing files.
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_vit(out):
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
    torch.save(model.state_dict(), out)
    print('For historical checkpoint reproduction, prefer the recorded original weight file; '
          'download/cache revisions and serialization can change this file hash.')


def prepare_dinov2(out, from_file):
    import torch
    if from_file:
        shutil.copyfile(Path(from_file).expanduser(), out)
    else:
        torch.hub.download_url_to_file(DINOV2_URL, str(out))
    state = torch.load(out, map_location='cpu')
    assert 'pos_embed' in state and tuple(state['pos_embed'].shape) == (1, 1370, 768), \
        'unexpected DINOv2 ViT-B/14 state_dict'
    repo = os.environ.get('FERREID_DINOV2_REPO')
    if repo and os.path.isdir(repo):  # strict load check against the model code in use
        model = torch.hub.load(repo, 'dinov2_vitb14', source='local', pretrained=False)
        model.load_state_dict(state, strict=True)
        print('strict load into', repo, 'OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['vit_b16', 'dinov2_b14'], default='vit_b16')
    parser.add_argument('--output-dir', default=os.environ.get('FERREID_WEIGHTS_DIR', 'weights'))
    parser.add_argument('--from-file', default='', help='dinov2_b14 only: copy this local file instead of downloading')
    args = parser.parse_args()
    name = {'vit_b16': 'vit_base_patch16_224.pth', 'dinov2_b14': 'dinov2_vitb14_pretrain.pth'}[args.model]
    out = Path(args.output_dir).expanduser() / name
    if out.exists():
        raise SystemExit(f'Refusing to overwrite existing weights: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.model == 'vit_b16':
        prepare_vit(out)
    else:
        prepare_dinov2(out, args.from_file)
    print(f'{out}\nSHA256 {sha256(out)}')


if __name__ == '__main__':
    main()
