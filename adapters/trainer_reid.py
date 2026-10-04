import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from custom_trainer import CustomTrainer
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS, NO_CAMERA_DOMAINS
from adapters.reid_dataset import (DomainPersonTrainDataset, CameraPairTrainDataset, CameraGroupTrainDataset,
                                   ContextPairDataset, DomainReIDEvalDataset)
from adapters.context_selection import ContextSampler, needs_features
from torchreid.metrics import evaluate_rank


_CLS_MAP = {
    'market1501': 'Market1501', 'msmt17': 'MSMT17',
    'viper': 'VIPeR', 'grid': 'GRID', 'ilids': 'iLIDS',
    'prid2011': 'PRID', 'cuhk02': 'CUHK02', 'cuhk03': 'CUHK03',
}


def _get_dataset_cls(name, cls_map=_CLS_MAP):
    import torchreid
    if name == "cuhk03":
        from adapters.cuhk03_np import CUHK03NP
        return CUHK03NP
    if name == "cuhksysu":
        from adapters.cuhksysu import CUHKSYSU
        return CUHKSYSU
    return getattr(torchreid.data.datasets.image, cls_map[name])


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _EvalView:
    """train / query / gallery lists, as used by evaluation."""

    def __init__(self, train, query, gallery):
        self.train, self.query, self.gallery = train, query, gallery


def _subsample_ids(ds, max_ids, seed=0):
    """Keep query/gallery images of a fixed random subset of test identities (validation speed-up)."""
    pids = sorted({x[1] for x in ds.query} & {x[1] for x in ds.gallery})
    if len(pids) <= max_ids:
        return ds
    keep = set(np.random.RandomState(seed).choice(pids, size=max_ids, replace=False).tolist())
    return _EvalView(ds.train,
                     [x for x in ds.query if x[1] in keep],
                     [x for x in ds.gallery if x[1] in keep])


class _RNGGuard:
    """Restore global RNG state on exit so evaluation never perturbs training."""

    def __enter__(self):
        self.state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                      torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

    def __exit__(self, *exc):
        py, npy, cpu, cuda = self.state
        random.setstate(py)
        np.random.set_state(npy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _domains(arg_value, config_key):
    if arg_value == "none":  # e.g. final model trained on every source domain, no validation
        return []
    return [d for d in arg_value.split(",") if d] if arg_value else list(DOMAIN_CONFIG[config_key])


def cosine_distmat(qf, gf, device, chunk=4096):
    """1 - q.g for L2-normalized features, computed in query chunks (on the GPU when available)
    so that large galleries (e.g. MSMT17: 11,659 x 82,161) fit in memory."""
    out = np.empty((qf.size(0), gf.size(0)), dtype=np.float32)
    g = gf.to(device)
    for i in range(0, qf.size(0), chunk):
        out[i:i + chunk] = (1 - qf[i:i + chunk].to(device) @ g.T).float().cpu().numpy()
    return out


from models import CONTEXT_BRANCH


def _read_state_dict(checkpoint_dir):
    for name in ("pytorch_model.bin", "model.safetensors"):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            if name.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(path)
            return torch.load(path, map_location="cpu", weights_only=True)
    raise FileNotFoundError("no pytorch_model.bin / model.safetensors in {}".format(checkpoint_dir))


def load_init_checkpoint(model, checkpoint_dir):
    """--init_from: copy matching weights of a checkpoint into a freshly built model before training.
    A VPT checkpoint's `prompt` (1, L, V, D) becomes the residual `base_prompt` (1, L*V, D), so a residual
    VICP with --delta_init_std 0 starts exactly at the VPT model. Everything that does not match keeps its
    initialisation and is listed; at least the encoder must load."""
    state = _read_state_dict(checkpoint_dir)
    own = model.state_dict()
    if "prompt" in state and "base_prompt" in own and "base_prompt" not in state:
        state["base_prompt"] = state.pop("prompt").reshape(own["base_prompt"].shape)
    loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    mismatched = sorted(k for k in state if k in own and own[k].shape != state[k].shape)
    unused = sorted(k for k in state if k not in own)
    missing = sorted(k for k in own if k not in loadable)
    if not any(k.startswith("encoder.") for k in loadable):
        raise ValueError("--init_from {}: no encoder weight matches this model".format(checkpoint_dir))
    model.load_state_dict(loadable, strict=False)
    n_enc = sum(k.startswith("encoder.") for k in loadable)
    print("[init_from] {}: loaded {} tensors ({} encoder{}); shape mismatch {}; unused {}".format(
        checkpoint_dir, len(loadable), n_enc, ", base_prompt" if "base_prompt" in loadable else "",
        mismatched, unused))
    fresh = sorted({k.split(".")[0] for k in missing if not k.startswith(("lm.", "encoder_copy."))})
    print("[init_from] freshly initialised (besides the pretrained LLM / frozen encoder copy):", fresh)
    return loadable


def load_prompt_teachers(path, domain_names, shape):
    """--prompt_teacher: teachers.pt of scripts/group_prompts.py -> (num_domains, L*V, D) FP32, in the order of
    the training dataset's domain_names. Every training domain needs a teacher."""
    if not path:
        raise ValueError("--prompt_kd_weight > 0 needs --prompt_teacher <teachers.pt>")
    prompts = torch.load(path, map_location="cpu", weights_only=False)["prompts"]
    missing = [d for d in domain_names if d not in prompts]
    if missing:
        raise ValueError("--prompt_teacher {} has no prompt for training domains {} (has {})".format(
            path, missing, sorted(prompts)))
    return torch.stack([prompts[d].float().reshape(shape) for d in domain_names])


def freeze_all_but_context(model):
    """--train_context_only: only the context branch of VICP stays trainable."""
    kept = []
    for n, p in model.named_parameters():
        train = n.startswith(CONTEXT_BRANCH)
        if p.requires_grad and not train:
            p.requires_grad_(False)
        if p.requires_grad:
            kept.append(n)
    if not kept:
        raise ValueError("--train_context_only left no trainable parameter (is this a VICP model?)")
    print("[train_context_only] trainable: {:.2f}M in {}".format(
        sum(p.numel() for n, p in model.named_parameters() if p.requires_grad) / 1e6,
        sorted({n.split(".")[0] for n in kept})))


class DGReIDTrainer(CustomTrainer):

    def __init__(self, args, device="cpu"):
        data_root = DOMAIN_CONFIG["data_root"]
        self.source_domains = _domains(args.source_domains, "source_domains")
        self.val_domains = _domains(args.val_domains, "val_domains")
        self.target_domains = _domains(args.target_domains, "target_domains")
        overlap = set(self.source_domains) & set(self.val_domains + self.target_domains)
        assert not overlap, "source domains reused for validation/test: {}".format(overlap)
        print("source domains:", self.source_domains, "(all images)" if args.source_all_images else "(train split)")
        print("val domains:", self.val_domains, "| target domains:", self.target_domains)

        source_datasets = []
        for name in self.source_domains:
            cls = _get_dataset_cls(name)
            # combineall: query/gallery identities are relabeled and appended to train
            ds = cls(root=data_root, combineall=args.source_all_images, verbose=False)
            print("  {}: {} ids / {} images".format(name, ds.num_train_pids, len(ds.train)))
            source_datasets.append((name, ds))

        if args.pseudo_domains == "camera_pair":
            train_dataset = CameraPairTrainDataset(source_datasets, instances_per_id=args.instances_per_id,
                                                   min_ids=args.pseudo_min_ids)
        elif args.pseudo_domains == "camera_group":
            if not args.camera_groups:
                raise ValueError("--pseudo_domains camera_group needs --camera_groups <groups.json>")
            train_dataset = CameraGroupTrainDataset(source_datasets, args.camera_groups,
                                                    instances_per_id=args.instances_per_id, min_ids=args.pseudo_min_ids)
        elif args.pseudo_domains == "none":
            train_dataset = DomainPersonTrainDataset(source_datasets, instances_per_id=args.instances_per_id,
                                                     cross_camera=args.cross_camera_instances)
        else:
            raise ValueError("--pseudo_domains must be none, camera_pair or camera_group")
        print("num identities: ", train_dataset.num_ids)
        # size of the ID-classification head (if any). A value given on the command line (e.g. read from
        # a checkpoint for evaluation) must agree with the data that is loaded now.
        if args.num_train_ids and args.num_train_ids != train_dataset.num_ids:
            raise ValueError("--num_train_ids={} but the source data has {} identities; use the source "
                             "domains / source_all_images / pseudo_domains of the checkpoint".format(
                                 args.num_train_ids, train_dataset.num_ids))
        args.num_train_ids = train_dataset.num_ids

        if args.model_type == "plain":
            from adapters.baseline_model import PlainReIDModel
            model = PlainReIDModel(args)
        elif args.model_type == "vpt":
            from adapters.baseline_model import VPTReIDModel
            model = VPTReIDModel(args)
        else:
            from adapters.reid_model import ReIDModel
            model = ReIDModel(args)
        if args.ctx_contrast_weight > 0:
            if args.model_type != "vicp":
                raise ValueError("--ctx_contrast_weight needs --model_type vicp")
            if args.batch_domain_mode != "single":
                raise ValueError("--ctx_contrast_weight needs --batch_domain_mode single")
            if len(train_dataset.domain_names) < 2:
                raise ValueError("--ctx_contrast_weight needs >= 2 training domains, got {}".format(
                    train_dataset.domain_names))
            print("contrastive context loss: weight {} margin {} over {} training domains{}".format(
                args.ctx_contrast_weight, args.ctx_contrast_margin, len(train_dataset.domain_names),
                ", cross path trains the context branch only" if args.ctx_contrast_detach_encoder else ""))
        if args.prompt_kd_weight > 0:
            if args.model_type != "vicp" or args.batch_domain_mode != "single":
                raise ValueError("--prompt_kd_weight needs --model_type vicp and --batch_domain_mode single")
            model._prompt_teachers = load_prompt_teachers(args.prompt_teacher, train_dataset.domain_names,
                                                          model.base_prompt.shape[1:])
            print("prompt distillation: weight {} mode {} over {} training domains (teachers {})".format(
                args.prompt_kd_weight, args.prompt_kd_mode, len(train_dataset.domain_names), args.prompt_teacher))
        if args.init_from:
            load_init_checkpoint(model, args.init_from)
        print("model type:", args.model_type, "| train_backbone:", args.train_backbone,
              "| trainable params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
        weight_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
        model.to(dtype=weight_dtype, device=device)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.to(dtype=torch.float32)
        # after the cast, so frozen LoRA / base prompt keep their FP32 values (not rounded to fp16)
        if args.train_context_only:
            freeze_all_but_context(model)

        # encoder weights trained in full fine-tuning get lr * backbone_lr_mult (LoRA / heads / prompts: lr)
        backbone, other = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = n.startswith("encoder.") and ".w_a" not in n and ".w_b" not in n
            (backbone if is_backbone else other).append(p)
        groups = [{"params": other}]
        if backbone:
            groups.append({"params": backbone, "lr": args.learning_rate * args.backbone_lr_mult})
            print("encoder parameters trained with lr x {}: {:.2f}M".format(
                args.backbone_lr_mult, sum(p.numel() for p in backbone) / 1e6))
        optimizer = torch.optim.Adam(
            groups,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        extra_losses = ["ot_loss", "id_loss", "icl_loss", "ce_loss", "std"]
        if args.ctx_contrast_weight > 0:
            extra_losses += ["ctx_contrast", "ctx_gap_own", "ctx_gap_cross"]
        if args.prompt_kd_weight > 0:
            extra_losses += ["prompt_kd"]

        self._device = device
        self._data_root = data_root
        self._dataset_cache = {}

        super().__init__(
            model=model,
            args=args,
            extra_losses=extra_losses,
            optimizers=[optimizer, None],
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )

    def _get_train_sampler(self, dataset=None):
        """Index stream of 100,000 batches of `batch_size` identities.

        batch_domain_mode=single (historical): each batch comes from one source domain (chosen
        uniformly), identities drawn with replacement (without with --unique_ids_per_batch);
        batch_domain_mode=mixed: identities drawn from all source domains together."""
        dataset = self.train_dataset
        pids = np.array(list(dataset.label2images.keys()))
        labels = np.array([x.split("/")[0] for x in pids])
        if self.args.batch_domain_mode == "mixed":
            label_to_indices = {"all": np.arange(len(pids))}
        elif self.args.batch_domain_mode == "single":
            label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        else:
            raise ValueError("--batch_domain_mode must be single or mixed")
        num_batches = 100000
        batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.accelerator.num_processes
        )
        print("batch_size:", batch_size, "num_batches:", num_batches)
        all_batches = []
        while len(all_batches) < num_batches:
            current_label = np.random.choice(list(label_to_indices.keys()))
            indices = label_to_indices[current_label]
            replace = not (self.args.unique_ids_per_batch and len(indices) >= batch_size)
            batch_indices = np.random.choice(indices, batch_size, replace=replace)
            all_batches.append(batch_indices)
        indices = np.array(all_batches).flatten()

        class OrderedSampler(Sampler):
            def __init__(self, indices):
                self.indices = indices

            def __iter__(self):
                return iter(self.indices)

            def __len__(self):
                return len(self.indices)

        return OrderedSampler(indices)

    # ------------------------------------------------------------------ evaluation

    def _load_split(self, name, split_id, max_ids=0):
        key = (name, split_id, max_ids)
        if key not in self._dataset_cache:
            cls = _get_dataset_cls(name)
            kwargs = {"split_id": split_id} if name in NUM_SPLITS else {}
            ds = cls(root=self._data_root, verbose=False, **kwargs)
            pool_paths = set(p for p, *_ in ds.train)
            eval_paths = set(p for p, *_ in ds.query + ds.gallery)
            assert not (pool_paths & eval_paths), (
                "Domain leak in {} split {}: context pool overlaps query/gallery".format(name, split_id)
            )
            if max_ids:
                ds = _subsample_ids(ds, max_ids)
            self._dataset_cache[key] = (ds, ContextSampler(ds.train, has_cameras=name not in NO_CAMERA_DOMAINS))
        return self._dataset_cache[key]

    @torch.no_grad()
    def _attach_selector_features(self, ds, sampler):
        """Label-free features of the context pool (ds.train order) for feature-based selectors."""
        from adapters.selectors import extract_selector_features
        loader = DataLoader(DomainReIDEvalDataset(ds.train), batch_size=256, shuffle=False,
                            num_workers=self.args.eval_num_workers)
        feats, style = extract_selector_features(self.model, loader, self._device)
        sampler.attach_features(feats, style)

    @torch.no_grad()
    def _extract(self, split_data, prompts, seed):
        _seed_all(seed)
        loader = DataLoader(DomainReIDEvalDataset(split_data), batch_size=256, shuffle=False,
                            num_workers=self.args.eval_num_workers)
        feats, pids, camids = [], [], []
        for images, batch_pids, batch_camids in loader:
            images = images.to(self._device)
            f = self.model(images, prompts=prompts)["features"]
            f = f + self.model(torch.flip(images, dims=(3,)), prompts=prompts)["features"]
            feats.append(F.normalize(f, p=2, dim=1).float().cpu())
            pids.append(batch_pids)
            camids.append(batch_camids)
        return torch.cat(feats), torch.cat(pids), torch.cat(camids)

    @torch.no_grad()
    def eval_context(self, ds, pairs, seed):
        """Build shared prompts from the annotated pairs, then rank query vs gallery."""
        _seed_all(seed)
        ctx = ContextPairDataset(pairs)
        batch = next(iter(DataLoader(ctx, batch_size=len(ctx), shuffle=False, num_workers=0)))
        batch = {k: v.to(self._device) for k, v in batch.items()}
        # num_icl_samples = sequence length L, filled from the annotated identities
        prompts = self.model(**batch)["prompts"]

        qf, q_pids, q_camids = self._extract(ds.query, prompts, seed)
        gf, g_pids, g_camids = self._extract(ds.gallery, prompts, seed)
        # features are L2-normalized, so cosine distance = 1 - q.g
        distmat = cosine_distmat(qf, gf, self._device)
        cmc, mAP = evaluate_rank(
            distmat, q_pids.numpy(), g_pids.numpy(), q_camids.numpy(), g_camids.numpy(),
            max_rank=10,
        )
        return {
            "rank1": float(cmc[0]) * 100.0,
            "rank5": float(cmc[4]) * 100.0,
            "rank10": float(cmc[9]) * 100.0,
            "mAP": float(mAP) * 100.0,
        }

    @torch.no_grad()
    def run_eval(self, domains, methods, ks, seeds, num_splits=None, max_ids=0):
        """Returns one row per (domain, split, method, k, seed). With --selection_unit image the row
        also records how many annotated pairs the k anchors produced (n_pairs / n_fail / n_dup)."""
        self.model.eval()
        unit = self.args.selection_unit
        rows = []
        with _RNGGuard():
            for name in domains:
                n_splits = min(num_splits or NUM_SPLITS.get(name, 1), NUM_SPLITS.get(name, 1))
                for split_id in range(n_splits):
                    try:
                        ds, sampler = self._load_split(name, split_id, max_ids)
                    except Exception as e:
                        print("Skipping {} split {}: {}".format(name, split_id, e))
                        break
                    if unit == "image" and any(needs_features(m) for m in methods) and not sampler.has_features:
                        self._attach_selector_features(ds, sampler)
                    for method in methods:
                        for k in ks:
                            for seed in seeds:
                                s = seed + 1000 * split_id
                                rng = np.random.RandomState(s)
                                pairs, info = sampler.draw(unit, method, k, rng)
                                if not pairs:
                                    print("WARNING: {} split {} {} k={} seed {}: no annotated pair, skipped"
                                          .format(name, split_id, method, k, seed))
                                    continue
                                res = self.eval_context(ds, pairs, s)
                                props = {key: v for key, v in info.items() if key.startswith("p_")}
                                rows.append(dict(domain=name, split=split_id, method=method, k=k, seed=seed,
                                                 unit=unit, n_pairs=info["n_pairs"], n_fail=info["n_fail"],
                                                 n_dup=info["n_dup"], **res, **props))
        return rows

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """In-training evaluation: held-out source (validation) domains only."""
        rows = self.run_eval(
            self.val_domains,
            methods=[self.args.context_method],
            ks=[self.args.context_k],
            seeds=range(self.args.eval_seeds),
            max_ids=self.args.val_max_ids,
        )
        results = {}
        for name in self.val_domains:
            sub = [r for r in rows if r["domain"] == name]
            for m in ("rank1", "mAP"):
                if sub:
                    results["val_{}_{}".format(name, m)] = float(np.mean([r[m] for r in sub]))
        if results:
            results["val_mean_rank1"] = float(np.mean([v for k, v in results.items() if k.endswith("_rank1")]))
            results["val_mean_mAP"] = float(np.mean([v for k, v in results.items() if k.endswith("_mAP")]))
            self.log(results)
        return results
