import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from custom_trainer import CustomTrainer
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS
from adapters.reid_dataset import DomainPersonTrainDataset, ContextPairDataset, DomainReIDEvalDataset
from adapters.context_selection import CandidatePool, select, make_pairs
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
    return [d for d in arg_value.split(",") if d] if arg_value else list(DOMAIN_CONFIG[config_key])


class DGReIDTrainer(CustomTrainer):

    def __init__(self, args, device="cpu"):
        data_root = DOMAIN_CONFIG["data_root"]
        self.source_domains = _domains(args.source_domains, "source_domains")
        self.val_domains = _domains(args.val_domains, "val_domains")
        self.target_domains = list(DOMAIN_CONFIG["target_domains"])
        overlap = set(self.source_domains) & set(self.val_domains + self.target_domains)
        assert not overlap, "source domains reused for validation/test: {}".format(overlap)
        print("source domains:", self.source_domains, "(all images)" if args.source_all_images else "(train split)")
        print("val domains:", self.val_domains)

        source_datasets = []
        for name in self.source_domains:
            cls = _get_dataset_cls(name)
            # combineall: query/gallery identities are relabeled and appended to train
            ds = cls(root=data_root, combineall=args.source_all_images, verbose=False)
            print("  {}: {} ids / {} images".format(name, ds.num_train_pids, len(ds.train)))
            source_datasets.append((name, ds))

        train_dataset = DomainPersonTrainDataset(source_datasets)
        print("num identities: ", len(train_dataset))

        if args.model_type == "plain":
            from adapters.baseline_model import PlainReIDModel
            model = PlainReIDModel(args)
        else:
            from adapters.reid_model import ReIDModel
            model = ReIDModel(args)
        print("model type:", args.model_type,
              "| trainable params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
        weight_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
        model.to(dtype=weight_dtype, device=device)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.to(dtype=torch.float32)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        extra_losses = ["ot_loss", "id_loss", "icl_loss", "std"]

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
        dataset = self.train_dataset
        pids = np.array(list(dataset.label2images.keys()))
        labels = np.array([x.split("/")[0] for x in pids])
        label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
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
            batch_indices = np.random.choice(indices, batch_size, replace=True)
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
            self._dataset_cache[key] = (ds, CandidatePool(ds.train))
        return self._dataset_cache[key]

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
        # num_icl_samples = sequence length L, filled from the k annotated identities
        prompts = self.model(**batch)["prompts"]

        qf, q_pids, q_camids = self._extract(ds.query, prompts, seed)
        gf, g_pids, g_camids = self._extract(ds.gallery, prompts, seed)
        # features are L2-normalized, so cosine distance = 1 - q.g
        distmat = (1 - qf @ gf.T).numpy()
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
        """Returns one row per (domain, split, method, k, seed)."""
        self.model.eval()
        rows = []
        with _RNGGuard():
            for name in domains:
                n_splits = min(num_splits or NUM_SPLITS.get(name, 1), NUM_SPLITS.get(name, 1))
                for split_id in range(n_splits):
                    try:
                        ds, pool = self._load_split(name, split_id, max_ids)
                    except Exception as e:
                        print("Skipping {} split {}: {}".format(name, split_id, e))
                        break
                    for method in methods:
                        for k in ks:
                            for seed in seeds:
                                s = seed + 1000 * split_id
                                rng = np.random.RandomState(s)
                                pids = select(method, pool, k, rng)
                                pairs = make_pairs(pool, pids, rng)
                                res = self.eval_context(ds, pairs, s)
                                rows.append(dict(domain=name, split=split_id, method=method,
                                                 k=k, seed=seed, **res))
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
