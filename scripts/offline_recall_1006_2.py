"""Frozen round-0 query recall. Selectors receive no true identities.
First batch only: structural baseline, not A/B/C/D or end-to-end training.
"""
import argparse, csv, hashlib, importlib.util, itertools, json, math, os, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    s = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m
repair = module("repair_offline", "adapters/active/repair.py")
candidates = module("candidates_offline", "adapters/active/candidates.py")

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

class Pool:
    def __init__(self, features, labels, cams):
        self.x = norm(np.asarray(features, np.float32))
        self.cams = np.asarray(cams, np.int64)
        labels = np.asarray(labels, np.int64)
        old = np.unique(labels[labels >= 0])
        self.c = len(old)
        self.labels = np.full(len(labels), -1, np.int64)
        for c, label in enumerate(old):
            self.labels[labels == label] = c
        self.gid = self.labels.copy()
        self.outliers = np.flatnonzero(labels < 0)
        self.gid[self.outliers] = self.c + np.arange(len(self.outliers))
        self.members = [np.flatnonzero(self.gid == c) for c in range(self.c)]
        self.members += [np.array([i]) for i in self.outliers]
        self.size = np.array([len(m) for m in self.members])
        self.cent = norm(np.stack([self.x[m].mean(0) for m in self.members]))
        self.rep = np.array([m[np.argmax(self.x[m] @ self.cent[c])] for c, m in enumerate(self.members)])
        self.n = len(self.members)

def dedup(pairs):
    seen, out = set(), []
    for a, b in pairs:
        key = tuple(sorted((int(a), int(b))))
        if key[0] != key[1] and key not in seen:
            seen.add(key); out.append(key)
    return np.array(out, dtype=np.int64).reshape(-1, 2)

def old_queries(pool, k, rule=False):
    C = pool.c
    S = pool.cent[:C] @ pool.cent[:C].T
    mask = np.zeros((C, len(np.unique(pool.cams))), np.int32)
    inv = np.unique(pool.cams, return_inverse=True)[1]
    for c, m in enumerate(pool.members[:C]):
        mask[c, inv[m]] = 1
    S[(mask @ mask.T) > 0] = -np.inf
    np.fill_diagonal(S, -np.inf)
    nb = np.argsort(-S, axis=1, kind="stable")[:, :min(k, max(0,C-1))]
    cluster_order = np.arange(C)
    if rule:
        cluster_order = np.lexsort((-S.max(axis=1), mask.sum(axis=1)))
    pairs = [(pool.rep[a], pool.rep[b]) for a in cluster_order for b in nb[a] if np.isfinite(S[a,b])]
    # Original all_k traverses cluster ids; rule prioritizes few-camera clusters.
    return dedup(pairs)

def random_queries(pool, count, seed):
    rng = np.random.RandomState(seed)
    total = pool.n * (pool.n-1) // 2
    count = min(count, total)
    selected, pairs = set(), []
    while len(pairs) < count:
        a,b = rng.randint(pool.n, size=2)
        key = tuple(sorted((int(a),int(b))))
        if a != b and key not in selected:
            selected.add(key); pairs.append((pool.rep[a],pool.rep[b]))
    return dedup(pairs)

def camera_queries(pool, k=10):
    members, parents, cams = [], [], []
    for a,m in enumerate(pool.members):
        for cam in np.unique(pool.cams[m]):
            members.append(m[pool.cams[m] == cam]); parents.append(a); cams.append(cam)
    parents, cams = np.array(parents), np.array(cams)
    cent = norm(np.stack([pool.x[m].mean(0) for m in members]))
    rep = np.array([m[np.argmax(pool.x[m] @ cent[a])] for a,m in enumerate(members)])
    xt = torch.as_tensor(cent,device="cuda")
    pa,ca = torch.as_tensor(parents,device="cuda"), torch.as_tensor(cams,device="cuda")
    scored = []
    for start in range(0,len(cent),512):
        end = min(start+512,len(cent))
        S = xt[start:end] @ xt.T
        S.masked_fill_((pa[start:end,None] == pa[None,:]) | (ca[start:end,None] == ca[None,:]), -torch.inf)
        val,ind = S.topk(min(k,len(cent)),dim=1)
        for a,(vv,jj) in enumerate(zip(val.cpu().numpy(),ind.cpu().numpy()),start):
            for v,b in zip(vv,jj):
                if np.isfinite(v):
                    scored.append((float(v),int(rep[a]),int(rep[b])))
    scored.sort(key=lambda t:(-t[0],t[1],t[2]))
    return dedup((a,b) for _,a,b in scored)

def selectors(pool):
    # This API cannot access pids: label-free methods are fully materialized first.
    result = {}
    for seed in (42,43,44):
        result["random_nodes_seed"+str(seed)] = random_queries(pool,max(4000,2*pool.c),seed)
    for k in (3,5,10):
        result["old_K"+str(k)+"_original_order"] = old_queries(pool,k)
    result["shortlist_rule_K1_unique_port"] = old_queries(pool,1,rule=True)
    print("Selecting exact cross-camera kNN repair2",flush=True)
    knn = candidates.candidate_pairs(torch.as_tensor(pool.x,device="cuda"),pool.cams,k=10,chunk=512)
    cand = repair.repair_candidates_knn(pool.x,pool.labels,knn)
    tau = candidates.random_pair_quantile(pool.x,seed=42)
    prob = repair.Calibrator([],[],tau)(cand["sim"])
    order = repair.rank_repair(cand,prob,np.random.RandomState(42),per_cluster=1)
    result["repair2_frozen_round0"] = dedup(zip(cand["i"][order],cand["j"][order]))
    print("Selecting camera subcluster cosine baseline",flush=True)
    result["camera_subcluster_cosine_K10"] = camera_queries(pool)
    return result, {"repair2_tau":tau}

def gold_pairs(pool,pids):
    """Cluster majority ties: lowest numeric pid. Outliers match every containing cluster."""
    pids = np.asarray(pids)
    majority, ties = [], []
    contains = {}
    for c,m in enumerate(pool.members[:pool.c]):
        u,n = np.unique(pids[m],return_counts=True)
        majority.append(u[np.argmax(n)].item())
        if (n == n.max()).sum() > 1: ties.append(c)
        for p in u: contains.setdefault(p.item(),[]).append(c)
    by_pid = {}
    for c,p in enumerate(majority): by_pid.setdefault(p,[]).append(c)
    M = {}
    for cs in by_pid.values():
        for a,b in itertools.combinations(cs,2): M[(a,b)] = int(min(pool.size[a],pool.size[b]))
    for a,i in enumerate(pool.outliers,pool.c):
        for b in contains.get(pids[i].item(),[]): M[(b,a)] = 1
    counts = np.unique(pids[pool.outliers],return_counts=True)[1]
    info = {"majority_tie_clusters":ties,
            "outlier_outlier_pairs_excluded_from_primary":int(np.sum(counts*(counts-1)//2)),
            "outliers_without_identity_in_any_cluster":sum(pids[i].item() not in contains for i in pool.outliers)}
    return M,info

def metrics(pool,pids,M,pairs):
    covered, effective, unique_groups = set(),set(),set()
    yes = 0
    for i,j in pairs:
        g = tuple(sorted((int(pool.gid[i]),int(pool.gid[j]))))
        unique_groups.add(g)
        answer = bool(pids[i] == pids[j])
        yes += answer
        if g in M:
            covered.add(g)
            if answer: effective.add(g)
    den = sum(M.values())
    row = {"queries":len(pairs),"unique_group_pairs":len(unique_groups),
           "covered_pairs":len(covered),"effective_pairs":len(effective),
           "recall":len(covered)/len(M) if M else None,
           "weighted_recall":sum(M[g] for g in covered)/den if den else None,
           "effective_weighted_recall":sum(M[g] for g in effective)/den if den else None,
           "M_pair_yield":len(covered)/len(unique_groups) if unique_groups else None,
           "covered_pairs_per_question":len(covered)/len(pairs) if len(pairs) else None,
           "oracle_yes_count":yes,"oracle_yes_rate":yes/len(pairs) if len(pairs) else None}
    return row,covered,effective

def write_csv(path, rows):
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def similarity_bins(pool,M):
    # Quantiles use unlabeled random parent pairs, never true split-pair quantiles.
    rng = np.random.RandomState(105)
    a,b = rng.randint(pool.n,size=(2,200000))
    keep = a != b
    sims = np.einsum("ij,ij->i",pool.cent[a[keep]],pool.cent[b[keep]])
    edges = np.quantile(sims,[.2,.4,.6,.8])
    bins = {g:int(np.searchsorted(edges,float(pool.cent[g[0]] @ pool.cent[g[1]]),side="right")) for g in M}
    return edges,bins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",required=True); parser.add_argument("--output",required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True,exist_ok=False)
    start = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    print("START",datetime.now().astimezone().isoformat(),args.input,flush=True)
    with np.load(args.input,allow_pickle=False) as z:
        pool = Pool(z["features"],z["labels"],z["cams"])
        selections,selector_info = selectors(pool)
        pids = z["pids"].copy() # identities accessed only after questions have been generated
    M,gold_info = gold_pairs(pool,pids)
    edges,bins = similarity_bins(pool,M)
    budget_specs = [("ratio_"+str(r),int(math.ceil(r*pool.c))) for r in (.25,.5,1,2)]
    budget_specs += [("fixed_"+str(n),n) for n in (250,500,1000,2000,4000)]
    rows,bin_rows = [],[]
    for method, pairs in selections.items():
        np.savez_compressed(out/(method+".npz"),image_pairs=pairs)
        is_random = method.startswith("random")
        specs = budget_specs + ([] if is_random else [("all",len(pairs))])
        if is_random:
            specs += [("sample_limit",len(pairs))]
        for budget,B in specs:
            selected = pairs[:B]
            row,covered,effective = metrics(pool,pids,M,selected)
            row = dict(method=method,budget=budget,requested_queries=B,
                       candidate_pool_size=len(pairs),candidate_pool_exhaustive=not is_random,**row)
            rows.append(row)
            for b in range(5):
                bg = [g for g in M if bins[g] == b]; den = sum(M[g] for g in bg)
                bin_rows.append(dict(method=method,budget=budget,bin=b+1,queries=len(selected),
                    gold_pairs=len(bg),gold_weight=den,
                    weighted_recall=sum(M[g] for g in bg if g in covered)/den if den else None,
                    effective_weighted_recall=sum(M[g] for g in bg if g in effective)/den if den else None))
        print(method, "questions",len(pairs),flush=True)
    write_csv(out/"metrics.csv",rows); write_csv(out/"similarity_bins.csv",bin_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key in ("weighted_recall","effective_weighted_recall"):
        fig,ax = plt.subplots(figsize=(9,6))
        for method in selections:
            if method.startswith("random") and not method.endswith("42"): continue
            vals = [r for r in rows if r["method"]==method and r["budget"].startswith("fixed")]
            ax.plot([v["queries"] for v in vals],[v[key] for v in vals],marker="o",markersize=3,label=method)
        ax.set(xlabel="Paid image-pair questions",ylabel=key.replace("_"," "),ylim=(0,1))
        ax.grid(alpha=.2); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(out/(key+".png"),dpi=180); fig.savefig(out/(key+".svg")); plt.close(fig)
    provenance = dict(stage="2_initial_baselines",started_at=datetime.fromtimestamp(start).astimezone().isoformat(),
        completed_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.time()-start,
        source=args.input,input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_training_lr=2e-4,images=len(pool.x),clusters=pool.c,outliers=len(pool.outliers),
        M_pairs=len(M),M_weight=sum(M.values()),bin_edges=edges.tolist(),
        gold_definition="cluster pairs with same majority pid plus outlier to EVERY cluster containing its pid; no outlier-outlier pairs in primary",
        weight_interpretation="min node size pair-weight proxy; not exact count of corrected images",
        query_cost="unique image pairs; repeated parent pairs consume budget but coverage counted once",
        fairness_note="old K/shortlist exclude outliers; random/repair2/camera baseline include them; coverage denominator is shared",
        old_K_order="original ascending cluster-id traversal; no target-label sorting",
        shortlist_scope="camera-count ascending, best eligible cosine descending, one nearest per cluster; deduplicate repeated image pairs before budgets",
        repair2_scope="existing kNN and ranking with initial label-free calibration and soft quota; no online updates or inferred-answer skipping",
        camera_scope="cross-parent cross-camera subclusters; cosine K10, descending score; NO A/B/C/D or hard quota yet",
        random_scope="uniform distinct node pairs incl singletons; finite sample, sample_limit is NOT all pairs",
        upper_bound=dict(recall=1,weighted_recall=1,oracle_only=True),
        pending=["UCAL MPPS port","SPAL DUS port","A3S verified port","components A/B/C/D and A-prime","five-round oracle training","negative constraint ablations","PASS integration"],
        **gold_info,**selector_info)
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2))
    (out/"DONE.json").write_text(json.dumps({"status":"done","completed_at":provenance["completed_at"]}))
    print("DONE",provenance["elapsed_seconds"],"seconds",flush=True)

if __name__ == "__main__": main()
