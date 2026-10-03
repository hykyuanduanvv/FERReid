"""Step 4: frozen trained VPT + zero-initialized context or shared correction.

Two matched seeds, 300 updates. Initial VPT equivalence, gradient connectivity,
frozen-weight hashes, and final own/cross-context retrieval are mandatory gates.
"""
import os, sys, json, copy, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from adapters.warm_start_reid import (WarmStartArguments, WarmStartReIDModel, load_exact,
                                       state_hash, tensor_hash)
from adapters.trainer_reid import DGReIDTrainer, _RNGGuard, _seed_all, cosine_distmat
from adapters.reid_dataset import ContextPairDataset
from torchreid.metrics import evaluate_rank


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


class WarmTrainer(DGReIDTrainer):
    def __init__(self, args, device):
        import adapters.reid_model as module
        original = module.ReIDModel
        module.ReIDModel = WarmStartReIDModel
        try:
            super().__init__(args, device)
        finally:
            module.ReIDModel = original
        state = torch.load(Path(args.warm_checkpoint)/'pytorch_model.bin', map_location='cpu', weights_only=True)
        self.model.restore_vpt(state, device)
        self.initial_frozen_hash = self.model.frozen_hash()
        self.logfile = (Path(args.output_dir)/'audit.jsonl').open('w', buffering=1)
        self.emit('initialization', all_parameter_hash=state_hash(self.model.state_dict().items()),
                  frozen_hash=self.initial_frozen_hash,
                  trainable=[n for n,p in self.model.named_parameters() if p.requires_grad])
        optimized = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        assert all(id(p) not in optimized for p in self.model.encoder.parameters())
        assert id(self.model.base_prompt) not in optimized
        scaler = self.accelerator.scaler
        s = scaler.state_dict(); s['scale'] = 128.0; scaler.load_state_dict(s)
        self.emit('scaler', state=scaler.state_dict())

    def emit(self, event, **kw):
        self.logfile.write(json.dumps(dict(event=event, step=self.state.global_step, **kw), allow_nan=False)+'\n')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        training = model.training
        if training:
            y = inputs['labels']; cut = self.args.episode_context_ids
            assert len(set(y[:cut].tolist()) & set(y[cut:].tolist())) == 0
            assert y.unique().numel() == len(y)
            self.emit('batch', images=tensor_hash(inputs['image_crops']), labels=tensor_hash(y),
                      cpu_rng=tensor_hash(torch.get_rng_state()), cuda_rng=tensor_hash(torch.cuda.get_rng_state()),
                      context_ids=cut, retrieval_ids=len(y)-cut)
        loss, output = super().compute_loss(model, inputs, True, num_items_in_batch)
        if training and self.state.global_step in (0, 1):
            names = (['prompt_mlp.weight', 'query_embeddings', 'mm_projector.visual_proj.weight']
                     if self.args.warm_variant == 'context' else ['shared_delta'])
            params = dict(model.named_parameters())
            gradients = torch.autograd.grad(output['id_loss'], [params[n] for n in names],
                                            retain_graph=True, allow_unused=True)
            for name, g in zip(names, gradients):
                self.emit('retrieval_gradient', parameter=name, connected=g is not None,
                          norm=float(g.float().norm()) if g is not None else None,
                          finite=bool(torch.isfinite(g).all()) if g is not None else None)
            if self.state.global_step == 0:
                g = gradients[0]
                assert g is not None and torch.isfinite(g).all() and g.float().norm()>0, 'dead correction branch'
        return (loss, output) if return_outputs else loss


class AuditCallback(TrainerCallback):
    def __init__(self, trainer): self.t = trainer
    def on_step_end(self, args, state, control, **kw):
        self.t.emit('optimizer', skipped=bool(self.t.accelerator.optimizer_step_was_skipped),
                    scale=float(self.t.accelerator.scaler.get_scale()))
    def on_save(self, args, state, control, **kw):
        frozen = self.t.model.frozen_hash()
        assert frozen == self.t.initial_frozen_hash, 'frozen VPT changed'
        save(Path(args.output_dir)/f'checkpoint-{state.global_step}'/'warm_start_manifest.json',
             dict(warm_checkpoint=args.warm_checkpoint, variant=args.warm_variant,
                  frozen_hash=frozen, step=state.global_step,
                  loader='adapters.warm_start_reid.WarmStartReIDModel',
                  warning='Generic VICP evaluation cannot load this subclass; use this experiment runner.'))


@torch.no_grad()
def make_prompt(t, source, draw):
    ds, sampler = t._load_split(source, 0, 0)
    seed = 1600 + draw
    pairs, info = sampler.draw('image', 'random', 16, np.random.RandomState(seed))
    assert pairs
    dsctx = ContextPairDataset(pairs)
    batch = next(iter(DataLoader(dsctx, batch_size=len(dsctx), shuffle=False, num_workers=0)))
    batch = {k: v.to(t._device) for k,v in batch.items()}
    _seed_all(seed)
    p = t.model(**batch)['prompts'][:1].clone()
    return p, dict(source=source, draw=draw, seed=seed, pairs=pairs, info=info)


@torch.no_grad()
def initial_gate(t):
    from adapters.baseline_model import VPTReIDModel
    out = Path(t.args.output_dir)
    with _RNGGuard():
        _seed_all(20261001)
        refargs = torch.load(Path(t.args.warm_checkpoint)/'training_args.bin', map_location='cpu', weights_only=False)
        reference = VPTReIDModel(refargs).to(t._device)
        state = torch.load(Path(t.args.warm_checkpoint)/'pytorch_model.bin', map_location='cpu', weights_only=True)
        load_exact(reference, state, t._device)
        reference.requires_grad_(False).eval()
        t.model.eval()
        p, context = make_prompt(t, 'cuhk03', 0)
        expected = reference.prompt.to(dtype=p.dtype)
        prompt_diff = float((p-expected).abs().max())
        ds,_ = t._load_split('cuhk03',0,t.args.val_max_ids)
        from adapters.reid_dataset import DomainReIDEvalDataset
        x = next(iter(DataLoader(DomainReIDEvalDataset(ds.query[:16]), batch_size=16)))[0].to(t._device)
        checks = {}
        for amp in (False, True):
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                a = t.model(x, prompts=p)['features']
                b = reference(x)['features']
            checks['amp' if amp else 'evaluation'] = float((a-b).abs().max())
        assert prompt_diff == 0 and max(checks.values()) == 0, (prompt_diff, checks)
        zero_rows = t.run_eval(['cuhk03'], ['random'], [16], [0], max_ids=t.args.val_max_ids)
        warm = t.model
        try:
            t.model = reference
            ref_rows = t.run_eval(['cuhk03'], ['random'], [16], [0], max_ids=t.args.val_max_ids)
        finally:
            t.model = warm
        assert len(ref_rows)==len(zero_rows)==1
        assert ref_rows[0]['mAP'] == zero_rows[0]['mAP']
        assert ref_rows[0]['rank1'] == zero_rows[0]['rank1']
        result = dict(prompt_max_difference=prompt_diff, feature_max_difference=checks,
                      reference=ref_rows, zero_correction=zero_rows, context=context,
                      frozen_hash=t.initial_frozen_hash)
        save(out/'initial_equivalence.json',result)
        print('INITIAL_EQUIVALENCE',json.dumps(result),flush=True)
        del reference, state
        torch.cuda.empty_cache()


@torch.no_grad()
def final_evaluation(t):
    """Matched CUHK03 query/gallery, three fixed own and cross draws; no tuning."""
    from scripts.context_sensitivity import load_images
    out = Path(t.args.output_dir)
    t.model.eval()
    assert t.model.frozen_hash() == t.initial_frozen_hash
    with _RNGGuard():
        ds,_ = t._load_split('cuhk03',0,t.args.val_max_ids)
        qi=load_images([r[0] for r in ds.query],t._device)
        gi=load_images([r[0] for r in ds.gallery],t._device)
        rows=[]; prompt_bank={}; contexts=[]
        qp=np.array([r[1] for r in ds.query]); gp=np.array([r[1] for r in ds.gallery])
        qc=np.array([r[2] for r in ds.query]); gc=np.array([r[2] for r in ds.gallery])
        # Mirror DGReIDTrainer's FP16 flip addition/normalization, then FP32 distances.
        def extract(images,p):
            feats=[]
            for i in range(0,len(images),256):
                x=images[i:i+256]
                f=t.model(x,prompts=p)['features']+t.model(x.flip(3),prompts=p)['features']
                feats.append(F.normalize(f,p=2,dim=1).float())
            return torch.cat(feats)
        base=t.model.base_prompts()
        ref_q=extract(qi,base); ref_g=extract(gi,base)
        ref_dist=(1-ref_q@ref_g.T).cpu().numpy()
        valid=~((qp[:,None]==gp[None,:])&(qc[:,None]==gc[None,:]))
        ref_order=np.argsort(np.where(valid,ref_dist,np.inf),axis=1)
        same=(qp[:,None]==gp[None,:])&valid; different=(qp[:,None]!=gp[None,:])&valid
        def add(condition,p,source='',draw=-1,baseline=False):
            if baseline: q,g,d=ref_q,ref_g,ref_dist
            else:
                q=extract(qi,p); g=extract(gi,p); d=(1-q@g.T).cpu().numpy()
            cmc,ap=evaluate_rank(d,qp,gp,qc,gc,max_rank=10)
            order=np.argsort(np.where(valid,d,np.inf),axis=1)
            row=dict(condition=condition,source=source,draw=draw,mAP=float(ap)*100,rank1=float(cmc[0])*100,
                prompt_delta_rms=float((p.float()-base.float()).square().mean().sqrt()),
                query_feature_change=float((q-ref_q).norm(dim=1).mean()),
                positive_distance=float(d[same].mean()),negative_distance=float(d[different].mean()),
                top1_changed=float((order[:,0]!=ref_order[:,0]).mean()))
            rows.append(row); save(out/'retrieval.partial.json',rows)
            print('FINAL_RETRIEVAL',json.dumps(row),flush=True)
        add('frozen_vpt',base,baseline=True)
        initial=json.loads((out/'initial_equivalence.json').read_text())['reference'][0]['mAP']
        assert abs(rows[0]['mAP']-initial)<1e-6, (rows[0]['mAP'],initial)
        for source in ('cuhk03','market1501','msmt17'):
            for draw in range(t.args.warm_eval_draws):
                p,ctx=make_prompt(t,source,draw); contexts.append(ctx)
                key=f'{source}_{draw}'; prompt_bank[key]=p.cpu()
                # All shared-arm prompts must be exactly context invariant.
                if t.args.warm_variant=='shared' and len(prompt_bank)>1:
                    assert torch.equal(p.cpu(),next(iter(prompt_bank.values())))
                    continue
                add('own' if source=='cuhk03' else 'cross',p,source,draw)
        own=[v for k,v in prompt_bank.items() if k.startswith('cuhk03_')]
        cross=[v for k,v in prompt_bank.items() if not k.startswith('cuhk03_')]
        mean_cross=torch.stack(cross).mean(0).to(t._device)
        if t.args.warm_variant=='context':
            add('mean_source_prompt',mean_cross,'market1501+msmt17')
            repeat,_=make_prompt(t,'cuhk03',0)
            assert torch.equal(repeat.cpu(),own[0]), 'non-deterministic prompt generation'
        flat=torch.stack([v.flatten().float() for v in prompt_bank.values()])
        geom=dict(context_repeat_max=0.0,
                  variation_rms=float((flat-flat.mean(0)).square().mean().sqrt()),
                  own_cross_cosine=float(torch.stack([F.cosine_similarity(a.flatten().float(),b.flatten().float(),dim=0)
                                                     for a in own for b in cross]).mean()),
                  frozen_hash=t.model.frozen_hash())
        save(out/'prompt_geometry.json',geom); save(out/'contexts.json',contexts)
        torch.save(prompt_bank,out/'evaluation_prompts.pt')
        save(out/'retrieval.json',rows)


def main():
    args=transformers.HfArgumentParser(WarmStartArguments).parse_args_into_dataclasses()[0]
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.jsonl').exists(): raise FileExistsError(out/'audit.jsonl')
    assert args.max_steps==300 and args.seed in (42,43) and not args.resume_from_checkpoint
    assert args.episode_context_ids==16 and args.episode_context_ids_min==0
    assert args.unique_ids_per_batch and args.instances_per_id==2 and args.num_icl_bs==1
    saved=torch.load(Path(args.warm_checkpoint)/'training_args.bin',map_location='cpu',weights_only=False)
    assert saved.model_type=='vpt'
    for name in ['backbone','train_backbone','lora_layers','lora_rank','num_vpt_tokens','bnneck',
                 'ce_loss_weight','source_domains','source_all_images']:
        assert getattr(args,name)==getattr(saved,name), (name,getattr(args,name),getattr(saved,name))
    transformers.set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    t=WarmTrainer(args,'cuda')
    t.add_callback(AuditCallback(t))
    save(out/'configuration.json',dict(arguments=args.to_dict(),
         checkpoint_sha256=hashlib.file_digest((Path(args.warm_checkpoint)/'pytorch_model.bin').open('rb'),'sha256').hexdigest(),
         sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                  ['scripts/train_vpt_context.py','adapters/warm_start_reid.py']}))
    initial_gate(t)
    if args.warm_smoke_only:
        t.emit('smoke_complete'); return
    t.train(); t.save_state()
    assert t.model.frozen_hash()==t.initial_frozen_hash
    # Trainer leaves an autocast + FP32-output wrapper on forward after train().
    # Remove it to match the explicit mixed-dtype initial/reference evaluation.
    t.model=t.accelerator.unwrap_model(t.model,keep_fp32_wrapper=False)
    final_evaluation(t)
    t.emit('complete'); t.logfile.close()


if __name__=='__main__': main()
