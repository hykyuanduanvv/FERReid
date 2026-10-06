"""VLM answers as clustering constraints for Cluster Contrast / PASS (FERReid plug-in).

answers: csv with path_a, path_b, vlm_score (log P(yes) - log P(no) of a frozen, source-fine-tuned MLLM).
Answers with score > yes become must-links, score < no cannot-links; the rest is ignored. They are applied to
the DBSCAN pseudo labels of every epoch:
  must-link   every pseudo cluster touched by a must-link group is merged (outliers in the group join it)
  cannot-link a cluster holding both ends of a cannot-link is split in two around them (nearest end by cosine)
Also: dump_epoch0() saves the epoch-0 features / labels so the questions can be generated outside.
"""
import csv

import numpy as np
import torch


def load_answers(path, paths, yes=2.2, no=-2.2):
    index = {p: k for k, p in enumerate(paths)}
    must, cannot = [], []
    for r in csv.DictReader(open(path)):
        if r.get("kind", "q_all") not in ("q_all",):
            continue
        a, b, s = index.get(r["path_a"]), index.get(r["path_b"]), float(r["vlm_score"])
        if a is None or b is None:
            continue
        if s > yes:
            must.append((a, b))
        elif s < no:
            cannot.append((a, b))
    print("VLM answers: {} must-links, {} cannot-links".format(len(must), len(cannot)))
    return must, cannot


def _find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def enforce(labels, features, must, cannot):
    labels = np.asarray(labels).copy()
    nxt = labels.max() + 1
    par = {}
    for a, b in must:
        par.setdefault(a, a); par.setdefault(b, b)
        ra, rb = _find(par, a), _find(par, b)
        if ra != rb:
            par[ra] = rb
    groups = {}
    for x in par:
        groups.setdefault(_find(par, x), []).append(x)
    for members in groups.values():
        lab = np.unique(labels[members])
        lab = lab[lab >= 0]
        if len(lab):
            labels[np.isin(labels, lab)] = lab[0]
            labels[members] = lab[0]
        else:
            labels[members] = nxt
            nxt += 1
    X = torch.nn.functional.normalize(features.float(), dim=1)
    for a, b in cannot:
        if labels[a] < 0 or labels[a] != labels[b]:
            continue
        m = np.flatnonzero(labels == labels[a])
        side = (X[m] @ X[b]) > (X[m] @ X[a])
        labels[m[side.numpy()]] = nxt
        nxt += 1
    # consecutive ids, -1 kept
    ok = labels >= 0
    _, inv = np.unique(labels[ok], return_inverse=True)
    out = np.full(len(labels), -1, np.int64)
    out[ok] = inv
    return out


def dump_epoch0(path, features, labels, dataset_train):
    items = sorted(dataset_train)
    np.savez(path, features=torch.nn.functional.normalize(features.float(), dim=1).numpy(), labels=np.asarray(labels),
             paths=np.array([f for f, _, _ in items]), pids=np.array([p for _, p, _ in items]),
             cams=np.array([c for _, _, c in items]))
    print("dumped epoch-0 clustering to", path)
