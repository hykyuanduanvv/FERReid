"""Materialize recorded dataset splits for a new data root; never silently overwrite."""
import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def materialize(name, splits, root):
    """VIPeR/GRID use path-pid-camera triples; i-LIDS stores image basenames."""
    root = root.resolve()
    for split in splits:
        for subset in ('train', 'query', 'gallery'):
            for row in split[subset]:
                if name == 'ilids':
                    image = root / 'ilids/i-LIDS_Pedestrian/Persons' / row
                else:
                    relative = Path(row[0])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError(f'Invalid manifest path: {relative}')
                    image = root / relative
                    row[0] = str(image)
                if not image.is_file():
                    raise FileNotFoundError(image)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.environ.get('FERREID_DATA_ROOT', 'data'))
    parser.add_argument('--domains', default='viper,grid,ilids')
    parser.add_argument('--apply', action='store_true', help='Write missing split files; default is check-only.')
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    for name in args.domains.split(','):
        if name not in ('viper', 'grid', 'ilids'):
            raise ValueError(f'Unsupported split manifest: {name}')
        src = REPO / 'data_manifests' / f'{name}_splits.json'
        value = materialize(name, json.loads(src.read_text(encoding='utf-8')), root)
        target = root / name / 'splits.json'
        if target.exists():
            if json.loads(target.read_text(encoding='utf-8')) != value:
                raise SystemExit(f'{target} differs from the recorded split. Preserve your existing '
                                 'file separately before deliberately installing this manifest.')
            print(f'{name}: existing split matches the recorded split')
        elif args.apply:
            with target.open('x', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
            print(f'{name}: installed {len(value)} splits at {target}')
        else:
            print(f'{name}: {len(value)} splits and all images checked; add --apply to write {target}')


if __name__ == '__main__':
    main()
