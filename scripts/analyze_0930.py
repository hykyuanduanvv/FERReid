"""Figures + summary tables for the 2026-09-30 experiments (oracle prompt, label value,
3000-step controls, DINOv2 backbone). Missing inputs are skipped.

  python scripts/analyze_0930.py   ->  experiments/report_0930/{*.png, summary.md}
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(ROOT, "experiments")
OUT = os.path.join(E, "report_0930")
os.makedirs(OUT, exist_ok=True)
DOMS = ["viper", "grid", "ilids"]
md = []


def table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 1. oracle prompt
ORACLES = [("oracle_prompt/lr1e-3", "ViT-B/16", "1e-3", "vit_lr1e-3"), ("oracle_prompt/lr1e-2", "ViT-B/16", "1e-2", "vit_lr1e-2"),
           ("oracle_prompt_dinov2/lr1e-3", "DINOv2-B/14", "1e-3", "dinov2_lr1e-3")]
for sub, bb, lr, tag in ORACLES:
    p = os.path.join(E, sub, "oracle_prompt.csv")
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    own = df[df.condition != "cross"]
    s = own.groupby(["domain", "condition"])[["rank1", "mAP"]].mean().reset_index()
    cr = df[df.condition == "cross"].groupby(["domain", "train_domain"])[["rank1", "mAP"]].mean().reset_index()
    md.append("## Oracle prompt, {} lr={}\n\nmean over splits (n={})\n".format(bb, lr, own.split.nunique()))
    piv = s.pivot(index="domain", columns="condition", values="mAP").reindex(DOMS)
    md.append(table(piv.reset_index()))
    md.append("\ncross-domain (prompt tuned on train_domain, evaluated on domain), mAP\n")
    md.append(table(cr))
    lo = own[own.condition.str.startswith("opt")].groupby("condition")[["loss_start", "loss_end", "delta_norm"]].mean()
    md.append("\nprompt tuning: triplet loss start/end, relative prompt change\n")
    md.append(table(lo.reset_index(), "{:.4f}"))
    hp = os.path.join(E, sub, "h_prompt_cosine.json")
    if os.path.exists(hp):
        md.append("\nh (LLM hidden state before prompt_mlp) vs prompt: " + open(hp).read().replace("\n", " "))

    conds = ["zero", "vicp", "opt_k4", "opt_k16", "opt_all", "cross"]
    labels = ["zero prompt", "VICP (k=16)", "tuned, 4 ids", "tuned, 16 ids", "tuned, all pool", "tuned on other domain"]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.13
    for i, c in enumerate(conds):
        if c == "cross":
            vals = [cr[cr.domain == d].mAP.mean() for d in DOMS]
        else:
            vals = [s[(s.domain == d) & (s.condition == c)].mAP.mean() for d in DOMS]
        ax.bar(np.arange(3) + (i - 2.5) * w, vals, w, label=labels[i])
    ax.set_xticks(range(3)); ax.set_xticklabels(DOMS); ax.set_ylabel("mAP (%)")
    base = s[s.condition == "vicp"].mAP.min()
    ax.set_ylim(max(0, s[s.condition == "zero"].mAP.min() - 5), s.mAP.max() + 3)
    ax.set_title("Domain-specific prompt upper bound (VICP {}, 3000 steps; prompt-only tuning, lr={})".format(bb, lr), fontsize=9)
    ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_oracle_prompt_{}.png".format(tag)), dpi=150); plt.close(fig)

# --------------------------------------------------------------------------- 2. label value
p = os.path.join(E, "label_value", "label_value.csv")
if os.path.exists(p):
    df = pd.read_csv(p)
    gammas = sorted(g for g in df.gamma.unique() if g >= 0)
    md.append("\n## Label value (frozen plain ViT features + KISSME)\n")
    fixed = df[df.condition.isin(["raw", "center", "unlab", "kiss_all"])]
    ft = fixed.groupby(["domain", "condition", "gamma"]).mAP.mean().reset_index()
    md.append(table(ft))
    kk = df[df.condition == "kiss_k"]
    per = kk.groupby(["domain", "gamma", "k", "split"]).mAP.agg(["mean", "std", "min", "max"]).reset_index()
    agg = per.groupby(["domain", "gamma", "k"])[["mean", "std", "min", "max"]].mean().reset_index()
    agg["oracle-mean"] = agg["max"] - agg["mean"]
    agg["mean-worst"] = agg["mean"] - agg["min"]
    md.append("\nkiss_k: per split over {} draws, then mean over splits (mAP)\n".format(kk.draw.nunique()))
    md.append(table(agg))

    fig, axes = plt.subplots(len(gammas), 3, figsize=(12, 3.6 * len(gammas)), squeeze=False)
    for gi, g in enumerate(gammas):
        for di, d in enumerate(DOMS):
            ax = axes[gi, di]
            a = agg[(agg.domain == d) & (agg.gamma == g)].sort_values("k")
            if a.empty:
                continue
            ax.fill_between(a.k, a["min"], a["max"], alpha=0.2, label="worst..best of draws")
            ax.plot(a.k, a["mean"], "o-", label="k labelled ids (mean)")
            ax.plot(a.k, a["max"], "^--", ms=4, label="oracle@N (best draw)")
            for c, st, lab in [("raw", ":", "raw"), ("center", "-.", "centered (no labels)"),
                               ("unlab", "--", "whitened (no labels)"), ("kiss_all", "-", "all pool labels")]:
                sub = ft[(ft.domain == d) & (ft.condition == c) & ((ft.gamma == g) | (ft.gamma < 0))]
                if not sub.empty:
                    ax.axhline(sub.mAP.iloc[0], ls=st, lw=1, color="gray" if c != "kiss_all" else "k", label=lab)
            ax.set_xscale("log", base=2); ax.set_xticks(a.k); ax.set_xticklabels(a.k)
            ax.set_title("{}  (gamma={})".format(d, g), fontsize=9); ax.set_xlabel("k"); ax.set_ylabel("mAP (%)")
            if gi == 0 and di == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Value of k annotated identities (frozen features + KISSME metric)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_label_value.png"), dpi=150); plt.close(fig)

# --------------------------------------------------------------------------- 3. training curves
runs = [("cv_fold1_3k", "VICP ViT-B/16"), ("vpt_fold1_3k", "VPT (fixed prompt) ViT-B/16"),
        ("plain_fold1_3k", "plain ViT-B/16 (LoRA only)"), ("dinov2_fold1_3k", "VICP DINOv2-B/14"),
        ("vpt_dinov2_fold1_3k", "VPT DINOv2-B/14"), ("plain_dinov2_fold1_3k", "plain DINOv2-B/14"),
        ("dinov2_fold1_3k_s1", "VICP DINOv2-B/14 seed1"), ("vpt_dinov2_fold1_3k_s1", "VPT DINOv2-B/14 seed1")]
hist = {}
for r, lab in runs:
    p = os.path.join(E, r, "val_cuhk03", "trainer_state.json")
    if os.path.exists(p):
        hist[lab] = json.load(open(p))["log_history"]
if hist:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    rows = []
    for lab, h in hist.items():
        v = [(x["step"], x["val_cuhk03_mAP"]) for x in h if "val_cuhk03_mAP" in x]
        if v:
            st, m = zip(*v)
            axes[0].plot(st, m, label=lab)
            rows.append(dict(model=lab, best_mAP=max(m), best_step=st[int(np.argmax(m))], final_mAP=m[-1],
                             mean_last5=float(np.mean(m[-5:]))))
        tr = [x for x in h if "id_loss" in x and "step" in x]
        stp = [x["step"] for x in tr]
        sm = lambda a: np.convolve(a, np.ones(10) / 10, mode="valid")
        if tr:
            axes[1].plot(stp[9:], sm([x["id_loss"] for x in tr]), label=lab)
            if any(x.get("icl_loss", 0) > 0 for x in tr):
                axes[2].plot(stp[9:], sm([x["icl_loss"] for x in tr]), label=lab)
    axes[0].set_title("CUHK03 validation mAP (500 ids)", fontsize=9); axes[0].set_xlabel("step")
    axes[1].set_title("id_loss (triplet), 100-step moving avg", fontsize=9); axes[1].set_xlabel("step")
    axes[2].set_title("icl_loss (LLM yes/no), 100-step moving avg", fontsize=9); axes[2].set_xlabel("step")
    axes[2].axhline(np.log(2), color="gray", ls=":", lw=1); axes[2].set_ylim(0.3, 0.8)
    for a in axes:
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_training_curves.png"), dpi=150); plt.close(fig)
    md.append("\n## Training (fold1 Market+MSMT -> val CUHK03, 3000 steps)\n")
    md.append(table(pd.DataFrame(rows)))
    for lab, h in hist.items():
        tr = [x for x in h if "id_loss" in x]
        last = tr[-20:]
        md.append("\n{}: last 200 steps id_loss {:.4f} icl_loss {:.4f} ot_loss {:.4f} std {:.4f}".format(
            lab, np.mean([x["id_loss"] for x in last]), np.mean([x.get("icl_loss", 0) for x in last]),
            np.mean([x["ot_loss"] for x in last]), np.mean([x["std"] for x in last])))

# --------------------------------------------------------------------------- 5. selection sensitivity under prompt tuning
SELECTS = [("prompt_select", "ViT-B/16, k=16, random selections", "vit_k16"),
           ("prompt_select_noise_vit", "ViT-B/16, k=16, FIXED selection (generator noise)", "vit_k16_noise"),
           ("prompt_select_dinov2/k16", "DINOv2, k=16, random selections", "dinov2_k16"),
           ("prompt_select_noise_dinov2", "DINOv2, k=16, FIXED selection (generator noise)", "dinov2_k16_noise"),
           ("prompt_select_dinov2/k4", "DINOv2, k=4, random selections", "dinov2_k4")]
sel_rows = []
for sub, title, tag in SELECTS:
    p = os.path.join(E, sub, "prompt_select.csv")
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    md.append("\n## Selection sensitivity: {} (VICP in-context vs prompt tuned on the same pairs)\n".format(title))
    st = []
    for col, lab in (("vicp_mAP", "VICP in-context"), ("tuned_mAP", "prompt tuned on k pairs")):
        # spread over draws within each split, then averaged over splits
        g = df.groupby(["domain", "split"])[col].agg(["mean", "std", "min", "max"])
        g["best-mean"] = g["max"] - g["mean"]; g["mean-worst"] = g["mean"] - g["min"]
        g = g.groupby("domain").mean().reindex(DOMS).dropna(how="all").reset_index()
        g.insert(1, "generator", lab); st.append(g)
        for _, r in g.iterrows():
            sel_rows.append(dict(run=tag, domain=r.domain, generator=lab, std=r["std"], best_mean=r["best-mean"]))
    md.append(table(pd.concat(st)))
    md.append("\n(n_splits={}, draws per split={})".format(df.split.nunique(), df.draw.nunique()))
    corr = {d: float(np.corrcoef(x.vicp_mAP, x.tuned_mAP)[0, 1]) for d, x in df.groupby("domain")}
    md.append("\ncorrelation over draws (vicp vs tuned): " + ", ".join("{} {:.2f}".format(d, v) for d, v in corr.items()))
    x0 = df[df.split == 0]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, d in zip(axes, DOMS):
        x = x0[x0.domain == d]
        if x.empty:
            continue
        for _, r in x.iterrows():
            ax.plot([0, 1], [r.vicp_mAP, r.tuned_mAP], "-o", color="tab:blue", alpha=0.5, ms=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["VICP in-context", "prompt tuned"]); ax.set_xlim(-0.3, 1.3)
        ax.set_title("{} split 0: {} draws".format(d, len(x)), fontsize=9); ax.set_ylabel("mAP (%)")
    fig.suptitle(title + " (lines connect the same annotated identities)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_selection_{}.png".format(tag)), dpi=150); plt.close(fig)
if sel_rows:
    S = pd.DataFrame(sel_rows)
    S = S[S.generator == "prompt tuned on k pairs"].pivot_table(index=["domain"], columns="run", values="std")
    md.append("\n### std of tuned mAP over draws: random selections vs fixed selection (noise)\n")
    md.append(table(S.reset_index()))

# lr chosen on the validation domain (CUHK03)
lr_rows = []
for lr in ["3e-4", "1e-3", "3e-3"]:
    for k in [4, 16]:
        p = os.path.join(E, "lr_select_cuhk03", "lr{}_k{}".format(lr, k), "prompt_select.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            lr_rows.append(dict(lr=lr, k=k, vicp_mAP=df.vicp_mAP.mean(), tuned_mAP=df.tuned_mAP.mean(),
                                gain=(df.tuned_mAP - df.vicp_mAP).mean(), tuned_std=df.tuned_mAP.std()))
if lr_rows:
    md.append("\n## Few-shot prompt tuning lr, chosen on the validation domain CUHK03 (DINOv2, 5 draws)\n")
    md.append(table(pd.DataFrame(lr_rows)))

# --------------------------------------------------------------------------- 4. target-domain evaluation
evals = [("eval3k/plain_vit", "plain ViT-B/16"), ("eval3k/vpt_vit", "VPT ViT-B/16"), ("eval3k/vicp_vit", "VICP ViT-B/16"),
         ("eval3k/plain_dinov2", "plain DINOv2"), ("eval3k/vpt_dinov2", "VPT DINOv2"), ("eval3k/vicp_dinov2", "VICP DINOv2"),
         ("eval3k/vpt_dinov2_s1", "VPT DINOv2 seed1"), ("eval3k/vicp_dinov2_s1", "VICP DINOv2 seed1"),
         ("eval_final/vpt_dinov2", "FINAL VPT DINOv2 (3 sources)"), ("eval_final/vicp_dinov2", "FINAL VICP DINOv2 (3 sources)")]
res = []
for e, lab in evals:
    p = os.path.join(E, e, "context_eval.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        per_seed = df.groupby(["domain", "seed"])[["rank1", "mAP"]].mean().reset_index()
        g = per_seed.groupby("domain").agg(rank1=("rank1", "mean"), mAP=("mAP", "mean"), mAP_seed_std=("mAP", "std")).reset_index()
        g.insert(0, "model", lab)
        res.append(g)
if res:
    R = pd.concat(res)
    md.append("\n## Target domains, 10 splits, random k=16 context, 3 seeds (3000-step models; fold1 unless FINAL)\n")
    md.append(table(R))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    models = list(R.model.unique())
    w = 0.8 / len(models)
    for mi, m in enumerate(models):
        sub = R[R.model == m].set_index("domain").reindex(DOMS)
        for ai, met in enumerate(["rank1", "mAP"]):
            bars = axes[ai].bar(np.arange(3) + (mi - (len(models) - 1) / 2) * w, sub[met], w, label=m)
            for b, v in zip(bars, sub[met]):
                axes[ai].text(b.get_x() + b.get_width() / 2, v + 0.5, "{:.1f}".format(v), ha="center", fontsize=6)
    for ai, met in enumerate(["Rank-1 (%)", "mAP (%)"]):
        axes[ai].set_xticks(range(3)); axes[ai].set_xticklabels(DOMS); axes[ai].set_ylabel(met)
    axes[0].legend(fontsize=7)
    fig.suptitle("Target domains (fold1 models, 3000 steps; 10 splits x 3 context seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_target_eval.png"), dpi=150); plt.close(fig)

open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("figures:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
