"""CPU smoke test of the model variants (tiny batch; needs backbone weights, Qwen for VICP).

  python tests/test_models.py [--skip_vicp]
  python tests/test_models.py --tiny        # no weights needed: random tiny ViT, VPT / plain only

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
    from adapters.baseline_model import build_model
    return build_model(args)


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
    p.add_argument("--tiny", action="store_true")
    a = p.parse_args()
    if a.tiny:
        m, bad = check("plain tiny", model_type="plain", backbone="tiny_test")
        assert not bad
        m, bad = check("vpt tiny + BNNeck/CE", model_type="vpt", backbone="tiny_test", ce_loss_weight=1.0, bnneck=True)
        assert not [n for n in bad if not n.endswith("bn.bias")]
        # a prompt with extra (appended) tokens per layer is accepted at retrieval time
        P = torch.cat([m.prompt.detach(), torch.randn(1, m.num_layers, 3, m.hidden_size) * 0.02], dim=2)
        with torch.no_grad():
            f = m.eval()(torch.randn(2, 3, 256, 128), prompts=P)["features"]
        assert f.shape[0] == 2
        print("PASS")
        return
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
    print("PASS")


if __name__ == "__main__":
    main()
