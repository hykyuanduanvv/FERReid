"""CPU-only validation of historical artifact hashes, source syntax and data manifests."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    provenance = json.loads((ROOT / 'results/archive_20260929/provenance.json').read_text(encoding='utf-8'))
    for relative, expected in provenance['artifact_sha256'].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f'Artifact hash mismatch: {relative}')
    sources = [ROOT / 'models.py', ROOT / 'custom_trainer.py']
    for folder in ('adapters', 'ops', 'scripts'):
        sources.extend((ROOT / folder).glob('*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for name, count in [('viper', 20), ('grid', 10), ('ilids', 10)]:
        splits = json.loads((ROOT / 'data_manifests' / f'{name}_splits.json').read_text(encoding='utf-8'))
        assert len(splits) == count, (name, len(splits))
        for split in splits:
            paths = {}
            for part in ('train', 'query', 'gallery'):
                assert split[part], (name, part)
                paths[part] = {r if name == 'ilids' else r[0] for r in split[part]}
                assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in paths[part])
            assert not paths['train'] & (paths['query'] | paths['gallery']), name
    from summarize_results import summarize
    base = ROOT / 'results/archive_20260929/baseline_plain'
    for folder in ('eval_plain', 'eval_vicp_fold1'):
        assert {r['domain'] for r in summarize(base / folder / 'context_eval.csv')} == {'viper', 'grid', 'ilids'}
    print(f"PASS: {len(provenance['artifact_sha256'])} historical hashes, "
          f"{len(sources)} Python syntax checks, 40 recorded splits, complete main result CSVs.")
    print('This does not load model weights or establish image-content disjointness.')


if __name__ == '__main__':
    main()
