"""CPU check of base-model training with multi-domain tokens, end to end through the HF Trainer (synthetic
images, random tiny ViT; about a minute; no data or weights needed).

  python tests/test_trainer.py

Three synthetic "source domains" -> DGReIDTrainer with --source_domain_tokens 2: batches are single-domain,
the per-sample domain index reaches the model, only the trained domains' tokens move, the checkpoint
records num_source_domains, and load_checkpoint_model rebuilds the same model from it.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import transformers

import adapters.trainer_reid as tr
from adapters.args_reid import ReIDTrainingArguments
from adapters.baseline_model import load_checkpoint_model
from test_active import make_images


class _FakeDataset:
    def __init__(self, train):
        self.train, self.query, self.gallery = train, [], []
        self.num_train_pids = len({x[1] for x in train})


def main():
    with tempfile.TemporaryDirectory() as tmp:
        data = {name: _FakeDataset(make_images(tmp, 6, 2, 2, seed=n, start_pid=0))
                for n, name in enumerate(["dom_a", "dom_b", "dom_c"])}
        tr._get_dataset_cls = lambda name: (lambda root, combineall=False, verbose=False: data[name])
        out = os.path.join(tmp, "run")
        args = transformers.HfArgumentParser(ReIDTrainingArguments).parse_args_into_dataclasses([
            "--output_dir", out, "--report_to", "none", "--model_type", "vpt", "--backbone", "tiny_test",
            "--num_vpt_tokens", "4", "--lora_layers", "2", "--lora_rank", "8", "--source_domain_tokens", "2",
            "--source_domains", "dom_a,dom_b,dom_c", "--val_domains", "none", "--source_all_images", "False",
            "--per_device_train_batch_size", "4", "--max_steps", "4", "--save_steps", "4", "--save_strategy", "steps",
            "--learning_rate", "1e-2", "--logging_steps", "2", "--dataloader_num_workers", "0",
            "--ot_loss_weight", "0", "--save_safetensors", "False", "--use_cpu", "True"])[0]
        transformers.set_seed(0)
        trainer = tr.DGReIDTrainer(args=args, device="cpu")
        assert args.num_source_domains == 3
        before = trainer.model.domain_prompts.detach().clone()
        seen = []

        def spy(module, a, kw):  # records the domains of every training batch
            if module.training and kw.get("domains") is not None:
                seen.append(sorted(set(kw["domains"].tolist())))
        trainer.model.register_forward_pre_hook(spy, with_kwargs=True)
        grads = []  # per step: which domains' tokens received a gradient
        trainer.model.domain_prompts.register_hook(
            lambda g: grads.append((g.abs().flatten(1).sum(1) > 0).nonzero().flatten().tolist()))
        trainer.train()
        moved = (trainer.model.domain_prompts.detach() - before).abs().flatten(1).sum(1) > 0
        assert seen and all(len(d) == 1 for d in seen), seen   # single-domain batches
        assert len(grads) == len(seen) and all(set(g) <= set(d) for g, d in zip(grads, seen)), (grads, seen)
        trained = sorted({x for g in grads for x in g})  # (a batch whose triplet loss is 0 sends no gradient)
        assert trained and moved.nonzero().flatten().tolist() == trained, (moved, trained)
        ckpt = os.path.join(out, "checkpoint-4")
        eval_args = transformers.HfArgumentParser(ReIDTrainingArguments).parse_args_into_dataclasses(
            ["--output_dir", tmp, "--report_to", "none"])[0]
        model = load_checkpoint_model(eval_args, "cpu", ckpt)
        assert eval_args.source_domain_tokens == 2 and eval_args.num_source_domains == 3
        assert torch.equal(model.domain_prompts, trainer.model.domain_prompts.detach())
        print("trainer: {} steps, batch domains {}, gradient to domains {}; checkpoint reloads: ok".format(
            len(seen), seen, grads))
    print("PASS")


if __name__ == "__main__":
    main()
