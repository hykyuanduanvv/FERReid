# PASS (ECCV'22) Cluster Contrast UDA: FERReid plug-in patches

Files modified from CASIA-IVA-Lab/PASS-reID (`PASS_cluster_contrast_reid/`), copied here for version control.
Copy them over a clean checkout:

| file | destination | change |
|---|---|---|
| `cluster_contrast_train_usl.py` | `examples/` | `--answers` (VLM answers as must / cannot-links on every epoch's DBSCAN labels), `--answer-yes/-no`, `--dump-epoch0` |
| `vlm_constraints.py` | `examples/` | new: answer loading, constraint enforcement, epoch-0 dump |
| `vision_transformer.py` | `clustercontrast/models/` | `torch._six` removed in torch 2 -> `collections.abc` |
| `serialization.py` | `clustercontrast/utils/` | `torch.load(..., weights_only=False)` (torch 2.6 default changed) |

Run (Market -> MSMT17, as `msmt_uda_mean.sh`, data root with `market1501/` and `MSMT17/`):

    PYTHONPATH=. python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --data-dir <root> \
      --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp pass_vits_market_sup.pth \
      --logs-dir <out> --feat-fusion mean --multi-neck [--answers answers.csv]

Questions for the VLM: `--dump-epoch0 epoch0.npz`, then `scripts/pass_questions.py`, then `scripts/diag_vlm.py`.
