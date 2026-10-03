# Direction_B_AL: completed short adaptation experiment

Completed 2026-10-03 03:14 (Asia/Shanghai): 168 trials, four target folds, seven selectors, three support draws and two optimizer seeds.

This is supervised target Prompt adaptation (100 successful updates), separate from later P3 and training-free VICP experiments.

[Experiment design and reproduction notes](../../docs/DIRECTION_B_AL_20261003.md) | [Original report](run/REPORT.md) | [Complete results](run/summary.json) | [User summary and verification](user_exports/summary_and_verification.json)

## Fixed 100-step mAP

Each adapted cell averages three support draws and two optimization seeds. Fold mean weights each target domain equally.

| Method | Market | CUHK-SYSU | CUHK03 | MSMT17 | Four-fold mean |
|---|---:|---:|---:|---:|---:|
| Source Prompt (no adaptation) | 74.88 | 94.35 | 52.08 | 36.99 | 64.58 |
| Random | 74.33 | 94.33 | 51.52 | 38.57 | 64.69 |
| Dedup | 74.66 | 94.35 | 51.74 | 37.97 | 64.68 |
| K-center | 75.82 | 94.40 | 52.60 | 36.82 | 64.91 |
| Typical | 74.88 | 94.36 | 52.71 | 38.16 | 65.03 |
| Hard Negative | 75.68 | 94.32 | 52.86 | 38.83 | 65.42 |
| Facility | 76.45 | 94.27 | 52.18 | 37.77 | 65.17 |
| Facility Camera | 75.84 | 94.27 | 52.57 | 37.85 | 65.13 |

100 steps remains the prespecified primary result;30-step results are secondary and remain in summary.json. No per-trial test-selected checkpoint.

## Contents

- run/code/runner.py: byte-exact original training, verification and orchestration code.
- run/adapt/: all168 configurations,100-step training logs,30/100-step retrieval results and completeness records.
- run/preflight/ and run/verify/: fourfold baseline checks and independent-process replay evidence.
- dependencies/p2b_20261002_v1/: source recipes,source metadata,frozen selections,data manifests and accepted MSMT17 policy.
- dependencies/code_snapshot/: every source file referenced by the historical provenance gate.
- archive_manifest.json:1029 original artifact entries with SHA256 and856 server-only checkpoint entries.
- publication_audit.json: CPU-only reaggregation and integrity check performed before publication.
- user_exports/: copies of the two previously delivered JSON reports;full results are semantically identical to the server summary.

## Verify without datasets or GPU

```bash
python scripts/audit_direction_b_al_archive.py
```

## Binary artifacts and paths

Dataset images and checkpoint binaries are not in Git. Prompt checkpoints (~1GB) and four source checkpoints remain under /root/hyk/FERReid/experiments/;see archive_manifest.json for exact paths, sizes and SHA256.
Large JSON files are losslessly stored as .json.gz; decompress before restoring their original layout. Original .log files have .log.txt archive names. Historical absolute paths remain unchanged.
This is an artifact audit, not a new GPU rerun. MSMT17 retains the accepted214 train/test exact-content overlap groups. Only three independent support draws and one source training seed are available.
