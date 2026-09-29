"""Recompute the README table from archived CSVs, without third-party dependencies."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(path):
    values = defaultdict(lambda: defaultdict(list))
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            values[(row['domain'], row['method'], int(row['k']))][int(row['seed'])].append(row)
    for (domain, method, k), seeds in sorted(values.items()):
        result = dict(domain=domain, method=method, k=k, seeds=len(seeds))
        for rows in seeds.values():
            splits = [int(row['split']) for row in rows]
            if sorted(splits) != list(range(10)):
                raise ValueError(f'{path}: expected exactly ten splits per seed: {splits}')
        for metric in ('mAP', 'rank1'):
            per_seed = [statistics.mean(float(r[metric]) for r in rows) for rows in seeds.values()]
            result[metric] = statistics.mean(per_seed)
            result[metric + '_seed_sd'] = statistics.pstdev(per_seed) if len(per_seed) > 1 else None
        yield result


def main():
    root = ROOT / 'results/archive_20260929/baseline_plain'
    for model, folder in [('plain', 'eval_plain'), ('vicp', 'eval_vicp_fold1')]:
        for row in summarize(root / folder / 'context_eval.csv'):
            sd = 'not estimated (one seed)' if row['mAP_seed_sd'] is None else f"{row['mAP_seed_sd']:.4f}"
            print(f"{model:5s} {row['domain']:6s} k={row['k']} splits=10 seeds={row['seeds']} "
                  f"mAP={row['mAP']:.4f} Rank-1={row['rank1']:.4f} support-seed mAP SD={sd}")
    print('SD is population SD over seed-wise split means, matching the legacy script; '
          'it is not split-to-split or training-seed uncertainty.')


if __name__ == '__main__':
    main()
