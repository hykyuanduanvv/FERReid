"""Verify the published 168-trial Direction_B_AL archive without CUDA or datasets."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(root):
    inventory = read(root / 'archive_manifest.json')
    for item in inventory['files']:
        raw = (root / item['archive_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['archive_sha256'], item['archive_path']
        original = gzip.decompress(raw) if item['encoding'] == 'gzip' else raw
        assert hashlib.sha256(original).hexdigest() == item['source_sha256'], item['archive_path']
        assert len(original) == item['source_bytes']
    run = root / 'run'
    manifest, summary = read(run / 'manifest.json'), read(run / 'summary.json')
    config = manifest['configuration']
    assert config == summary['configuration']
    assert config['steps'] == config['main_step'] == 100
    assert config['draws'] == [0, 1, 2] and config['seeds'] == [42, 43]
    runner = run / 'code/runner.py'
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == manifest['runner_sha256']
    expected_paths = set()
    recomputed = {}
    for domain in config['domains']:
        verify = read(run / 'verify' / (domain + '.json'))
        assert verify['baseline_pass'] and verify['independent_process']
        assert len(verify['trials']) == 2 and all(t['passed'] for t in verify['trials'])
        fold = manifest['folds'][domain]
        assert fold['baseline'] == summary['domains'][domain]['baseline']
        recomputed[domain] = {}
        for method in config['methods']:
            scores = {'30': [], '100': []}
            for draw in config['draws']:
                for seed in config['seeds']:
                    trial = run / 'adapt' / domain / method / ('draw%d_seed%d' % (draw, seed))
                    expected_paths.add(trial / 'complete.json')
                    result, args = read(trial / 'complete.json'), read(trial / 'configuration.json')
                    assert args['configuration'] == config and args['seed'] == seed
                    assert args['runner_sha256'] == manifest['runner_sha256']
                    assert result['status'] == 'complete' and result['actual_updates'] == 100
                    assert result['saved_prompt_exact'] and result['feature_replay_exact']
                    assert result['frozen_hash'] == fold['frozen_hash']
                    assert result['relative_prompt_delta'] <= .100002
                    logs = [json.loads(line) for line in (trial / 'training.jsonl').read_text().splitlines()]
                    assert [row['step'] for row in logs] == list(range(1, 101))
                    assert all(row['relative_prompt_delta'] <= .100002 for row in logs)
                    assert [row['step'] for row in result['rows']] == [30, 100]
                    for row in result['rows']: scores[str(row['step'])].append(row['mAP'])
            recomputed[domain][method] = {}
            for step, values in scores.items():
                value = statistics.mean(values)
                expected = summary['domains'][domain]['methods'][method]['metrics'][step]['mAP_mean']
                assert abs(value - expected) < 1e-10
                recomputed[domain][method][step] = value
    assert len(expected_paths) == 168
    assert set((run / 'adapt').glob('*/*/draw*_seed*/complete.json')) == expected_paths
    macro = {}
    for method in config['methods']:
        macro[method] = {}
        for step in ['30', '100']:
            value = statistics.mean(recomputed[d][method][step] for d in config['domains'])
            assert abs(value - summary['four_fold_macro'][method][step]) < 1e-10
            macro[method][step] = value
    # Check every code file required by the historical provenance gate, including
    # shared diagnostic helpers that were untracked in the original checkout.
    dependencies = root / 'dependencies/p2b_20261002_v1'
    for domain in config['domains']:
        provenance = read(dependencies / 'preflight' / domain / 'provenance.json')
        for relative, expected in provenance['code_sha256'].items():
            actual = hashlib.sha256((root / 'dependencies/code_snapshot' / relative).read_bytes()).hexdigest()
            assert actual == expected, (domain, relative)
    return dict(passed=True, trials=168, training_log_rows=16800, independent_fold_checks=4,
                archive_files=len(inventory['files']), four_fold_macro=macro,
                baseline_macro=statistics.mean(manifest['folds'][d]['baseline']['mAP'] for d in config['domains']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[1] / 'results/Direction_B_AL_20261003')
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), indent=2))
