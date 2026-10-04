"""CPU smoke test of the model variants (tiny batch; needs backbone weights, Qwen for VICP).

  python tests/test_models.py [--skip_vicp]

For each variant: forward + backward in training mode, the expected trainable parameters receive
gradients, eval-mode features are L2-normalized, and the defaults keep the historical structure
(no ID head, LoRA r=128 on the last 4 blocks, 2.36M LoRA parameters).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import transformers

from adapters.args_reid import ReIDTrainingArguments


def make_args(**kw):
    argv = ["--output_dir", "/tmp/ferreid_test", "--report_to", "none", "--num_icl_samples", "4"]
    for k, v in kw.items():
        argv += ["--" + k, str(v)]
    return transformers.HfArgumentParser(ReIDTrainingArguments).parse_args_into_dataclasses(argv)[0]


def build(args):
    if args.model_type == "vpt":
        from adapters.baseline_model import VPTReIDModel
        return VPTReIDModel(args)
    if args.model_type == "plain":
        from adapters.baseline_model import PlainReIDModel
        return PlainReIDModel(args)
    from adapters.reid_model import ReIDModel
    return ReIDModel(args)


def check(name, K=2, **kw):
    torch.manual_seed(0)
    args = make_args(num_train_ids=10, **kw)
    m = build(args)
    m.train()
    x = torch.randn(3, K, 3, 256, 128)
    y = torch.tensor([1, 4, 7])
    out = m(x, y)
    out["loss"].backward()
    trainable = {n: p for n, p in m.named_parameters() if p.requires_grad}
    no_grad = [n for n, p in trainable.items() if p.grad is None]
    lora = sum(p.numel() for n, p in trainable.items() if ".w_a" in n or ".w_b" in n)
    m.eval()
    with torch.no_grad():
        f = m(x[:, 0], prompts=out["prompts"].detach() if out["prompts"] is not None else None)["features"]
    assert torch.allclose(f.norm(dim=1), torch.ones(len(f)), atol=1e-4)
    print("{:<34} loss {:.3f} (triplet {:.3f}, ce {:.3f}) | trainable {:.2f}M, LoRA {:.2f}M, head {} | "
          "no-grad trainable: {}".format(name, out["loss"].item(), out["id_loss"].item(),
                                         float(out.get("ce_loss", torch.tensor(0.0))), sum(p.numel() for p in trainable.values()) / 1e6,
                                         lora / 1e6, m.reid_head is not None, no_grad[:3] or "none"))
    return m, no_grad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_vicp", action="store_true")
    a = p.parse_args()
    m, bad = check("plain vit (historical defaults)", model_type="plain")
    assert m.reid_head is None and not bad
    m, bad = check("vpt dinov2 + BNNeck/CE, K=4", K=4, model_type="vpt", backbone="dinov2_b14",
                   ce_loss_weight=1.0, bnneck=True)
    assert m.reid_head is not None and not [n for n in bad if not n.endswith("bn.bias")]
    m, bad = check("plain dinov2 full FT + BNNeck/CE", model_type="plain", backbone="dinov2_b14",
                   train_backbone="full", ce_loss_weight=1.0, bnneck=True)
    m, bad = check("vpt dinov2 lora 8 layers r32, no WPA", model_type="vpt", backbone="dinov2_b14",
                   lora_layers=8, lora_rank=32, ot_loss_weight=0)
    if not a.skip_vicp:
        m, bad = check("vicp vit (historical defaults)", model_type="vicp")
        assert m.reid_head is None
        m, bad = check("vicp dinov2 + BNNeck/CE, K=4", K=4, model_type="vicp", backbone="dinov2_b14",
                       ce_loss_weight=1.0, bnneck=True)
        # direction A: residual prompt + EMA centring + episodic split (context = first identity of 3)
        m, bad = check("vicp dinov2 A: residual+ema+episodic", model_type="vicp", backbone="dinov2_b14",
                       prompt_mode="residual", ctx_center="ema", episode_context_ids=1)
        assert not bad and m.h_mean_count.item() == 1, (bad, m.h_mean_count)
        assert tuple(m.base_prompts().shape) == (1, m.num_layers, m.args.num_vpt_tokens, m.hidden_size)
        m.eval()
        with torch.no_grad():  # two different contexts must give different prompts
            torch.manual_seed(1)
            p1 = m(torch.randn(3, 2, 3, 256, 128), torch.arange(3))["prompts"][0].flatten()
            torch.manual_seed(1)
            p2 = m(torch.randn(3, 2, 3, 256, 128) * 3, torch.arange(3))["prompts"][0].flatten()
        c = torch.nn.functional.cosine_similarity(p1, p2, dim=0).item()
        print("   residual prompt cosine between two contexts: {:.4f} (must be < 1)".format(c))
        assert c < 0.9999
        check_contrast()
        check_distill()
    print("PASS")


def check_distill():
    """direction A, --prompt_kd_weight: features under the generated prompt are pulled towards the features
    under the batch domain's teacher prompt; the gradient reaches the context branch, the teacher path has none."""
    torch.manual_seed(0)
    m = build(make_args(num_train_ids=10, model_type="vicp", backbone="dinov2_b14", prompt_mode="residual",
                        prompt_kd_weight=1.0, episode_context_ids=1))
    base = m.base_prompt.detach()[0]
    m._prompt_teachers = torch.stack([base + 0.5 * torch.randn_like(base), base - 0.5 * torch.randn_like(base)])
    m.train()
    x, y = torch.randn(3, 2, 3, 256, 128), torch.tensor([1, 4, 7])
    for mode in ("rel", "feat"):
        m.args.prompt_kd_mode = mode
        m.zero_grad(set_to_none=True)
        out = m(x, y, domains=torch.ones(3, dtype=torch.long))
        out["loss"].backward()
        got = {n.split(".")[0] for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        assert out["prompt_kd"].item() > 0 and "prompt_mlp" in got, (out["prompt_kd"].item(), got)
        print("{:<34} prompt_kd {:.5f} | grads in {}".format("vicp A: prompt distill ({})".format(mode),
                                                              out["prompt_kd"].item(), sorted(got)))


def check_contrast():
    """direction A, --ctx_contrast_weight: the first batch of a domain only fills the context bank, a batch of
    another domain then adds the contrastive term; --init_from a VPT checkpoint + --train_context_only."""
    import tempfile
    from adapters.trainer_reid import load_init_checkpoint, freeze_all_but_context
    from ops.losses import per_anchor_gap
    torch.manual_seed(0)
    g = per_anchor_gap(torch.nn.functional.normalize(torch.randn(6, 8), dim=1), torch.tensor([0, 0, 1, 1, 2, 2]))
    assert g.shape == (6,)
    with tempfile.TemporaryDirectory() as tmp:
        vpt, _ = check("vpt dinov2 (for --init_from)", model_type="vpt", backbone="dinov2_b14")
        torch.save(vpt.state_dict(), os.path.join(tmp, "pytorch_model.bin"))
        kw = dict(model_type="vicp", backbone="dinov2_b14", prompt_mode="residual", ctx_center="ema",
                  delta_init_std=0, ctx_contrast_weight=1.0, ctx_contrast_margin=0.05, train_context_only=True)
        torch.manual_seed(0)
        m = build(make_args(num_train_ids=10, **kw))
        loaded = load_init_checkpoint(m, tmp)
        assert "base_prompt" in loaded and torch.equal(
            m.base_prompt.detach(), vpt.prompt.detach().reshape(m.base_prompt.shape))
        freeze_all_but_context(m)
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        assert trainable and not any(n.startswith(("encoder.", "base_prompt", "lm.")) for n in trainable), trainable
    m.train()
    x, y = torch.randn(3, 2, 3, 256, 128), torch.tensor([1, 4, 7])
    out0 = m(x, y, domains=torch.zeros(3, dtype=torch.long))  # domain 0: bank empty -> no contrast term
    assert out0["ctx_contrast"].item() == 0 and set(m._ctx_bank) == {0}
    m.zero_grad()
    out1 = m(torch.randn(3, 2, 3, 256, 128), y, domains=torch.ones(3, dtype=torch.long))
    out1["loss"].backward()
    assert set(m._ctx_bank) == {0, 1}
    assert out1["ctx_contrast"].item() > 0, "margin > 0 and gap_own == gap_cross at delta 0 -> loss = margin"
    got = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert any(n.startswith("prompt_mlp.") for n in got), got
    print("{:<34} ctx_contrast {:.4f} (gap own {:.4f}, cross {:.4f}) | grads in {}".format(
        "vicp A: ctx contrast + init_from", out1["ctx_contrast"].item(), out1["ctx_gap_own"].item(),
        out1["ctx_gap_cross"].item(), sorted({n.split(".")[0] for n in got})))
    try:
        m(x, y, domains=torch.tensor([0, 1, 0]))
        raise AssertionError("mixed-domain batch must be rejected")
    except ValueError:
        pass
    del m
    # --ctx_contrast_detach_encoder: the cross-domain path reaches the context branch, never the encoder / LoRA
    torch.manual_seed(0)
    m = build(make_args(num_train_ids=10, model_type="vicp", backbone="dinov2_b14", prompt_mode="residual",
                        ctx_contrast_weight=1.0))
    m.train()
    f, lab = torch.randn(6, m.hidden_size), torch.tensor([0, 0, 1, 1, 2, 2])
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    for detach in (False, True):
        m.args.ctx_contrast_detach_encoder = detach
        m.zero_grad(set_to_none=True)
        (m._cross_domain_features(f, lab, torch.randn(4, 3, 256, 128)) ** 3).sum().backward()
        got = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        assert [n for n, p in m.named_parameters() if p.requires_grad] == trainable  # restored
        assert any(n.startswith("prompt_mlp.") for n in got) and any(n.startswith("mm_projector.") for n in got), got
        enc = sorted(n for n in got if n.startswith(("encoder.", "base_prompt", "lm.")))
        assert bool(enc) != detach, (detach, enc[:3])
        print("   cross path, detach_encoder={}: grads outside the context branch: {}".format(
            detach, len(enc)))


if __name__ == "__main__":
    main()
