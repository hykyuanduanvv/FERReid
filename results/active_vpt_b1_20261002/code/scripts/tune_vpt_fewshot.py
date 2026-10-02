"""Step 5: tune only an external prompt on k annotated target-pool pairs.

Explicit mixed-dtype VPT loading; FP32 Triplet; no Trainer/LLM. Image selection
is label-free, annotation is simulated, and evaluation identities are disjoint.
Only selected support images are cached; query/gallery evaluation is streamed.
"""
import sys,os,json,hashlib,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adapters.baseline_model import VPTReIDModel
from adapters.warm_start_reid import load_exact,state_hash,tensor_hash
from adapters.trainer_reid import _get_dataset_cls,_subsample_ids,cosine_distmat
from adapters.config_reid import DOMAIN_CONFIG,NUM_SPLITS,NO_CAMERA_DOMAINS
from adapters.context_selection import ContextSampler
from adapters.reid_dataset import DomainReIDEvalDataset
from scripts.context_sensitivity import load_images
from scripts.oracle_prompt import augment
from ops.losses import HardTripletLoss
from torchreid.metrics import evaluate_rank


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))


def raw_identity(domain,path):
    p=Path(path)
    if domain=='cuhk03': return p.parent.name
    if domain=='ilids': return str(int(p.name[:4]))
    if domain in ('viper','grid'): return str(int(p.stem.split('_')[0]))
    raise ValueError('Add verified raw identity parser for '+domain)


def load_data(domain):
    kw={'split_id':0} if domain in NUM_SPLITS else {}
    ds=_get_dataset_cls(domain)(root=DOMAIN_CONFIG['data_root'],verbose=False,**kw)
    train_paths={r[0] for r in ds.train}; test_paths={r[0] for r in ds.query+ds.gallery}
    train_ids={raw_identity(domain,p) for p in train_paths}
    test_ids={raw_identity(domain,p) for p in test_paths}
    assert not train_paths&test_paths,'image leakage'
    assert not train_ids&test_ids,'original-identity leakage (not relabeled numeric IDs)'
    manifest=dict(domain=domain,split=0,pool_images=len(train_paths),pool_identities=len(train_ids),
                  original_test_identities=len(test_ids),path_overlap=0,raw_identity_overlap=0,
                  original_splits={key:hashlib.sha256(json.dumps(getattr(ds,key)).encode()).hexdigest()
                                   for key in ['train','query','gallery']})
    if domain=='cuhk03': ds=_subsample_ids(ds,500)
    manifest.update(query_images=len(ds.query),gallery_images=len(ds.gallery),
                    evaluated_query_ids=len({r[1] for r in ds.query}),
                    query_manifest=ds.query,gallery_manifest=ds.gallery)
    sampler=ContextSampler(ds.train,has_cameras=domain not in NO_CAMERA_DOMAINS)
    draws=[]
    by_path={r[0]:r for r in ds.train}
    for draw in range(3):
        pairs,info=sampler.draw('image','random',16,np.random.RandomState(1600+draw))
        assert len(pairs)>=2
        ids=[]
        for a,b in pairs:
            assert a in train_paths and b in train_paths and a!=b
            assert by_path[a][1]==by_path[b][1]
            assert raw_identity(domain,a)==raw_identity(domain,b)
            if domain not in NO_CAMERA_DOMAINS: assert by_path[a][2]!=by_path[b][2]
            ids.append(raw_identity(domain,a))
        assert len(ids)==len(set(ids))
        draws.append(dict(draw=draw,selection_seed=1600+draw,pairs=pairs,
                          annotation=info,effective_identities=len(ids),original_ids=ids,
                          support_sha256=hashlib.sha256(json.dumps(pairs).encode()).hexdigest()))
    return ds,manifest,draws


@torch.no_grad()
def extract(model,records,prompt):
    loader=DataLoader(DomainReIDEvalDataset(records),batch_size=256,shuffle=False,num_workers=4)
    values=[]; pids=[]; cams=[]
    for x,yp,yc in loader:
        x=x.cuda()
        f=model(x,prompts=prompt)['features']+model(x.flip(3),prompts=prompt)['features']
        values.append(F.normalize(f,p=2,dim=1).float().cpu())
        pids.append(yp); cams.append(yc)
    return torch.cat(values),torch.cat(pids),torch.cat(cams)


@torch.no_grad()
def evaluate(model,ds,prompt):
    q,qp,qc=extract(model,ds.query,prompt)
    g,gp,gc=extract(model,ds.gallery,prompt)
    cmc,ap=evaluate_rank(cosine_distmat(q,g,'cuda'),qp.numpy(),gp.numpy(),qc.numpy(),gc.numpy(),max_rank=10)
    return dict(mAP=float(ap)*100,rank1=float(cmc[0])*100,rank5=float(cmc[4])*100)


def fp32_loss(model,images,labels,prompt,criterion):
    feats=model(images,prompts=prompt)['features']
    with torch.autocast('cuda',enabled=False):
        return criterion(feats.float(),labels)


def run_trial(model,ds,support,init,selection,lr,seed,out,base,frozen_hash,steps=300):
    directory=out/f"draw{selection['draw']}_seed{seed}_lr{lr:g}"
    directory.mkdir(exist_ok=True)
    if (directory/'complete.json').exists():
        return json.loads((directory/'complete.json').read_text())
    if (directory/'training.jsonl').exists(): raise FileExistsError('Partial trial needs an explicitly new directory: '+str(directory))
    log=(directory/'training.jsonl').open('w',buffering=1)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    gen=torch.Generator(device='cuda').manual_seed(seed)
    rng=np.random.RandomState(seed)
    p=torch.nn.Parameter(init.clone().float())
    optimizer=torch.optim.Adam([p],lr=lr)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.0)
    criterion=HardTripletLoss(margin=0.1,hardest=True)
    labels=torch.arange(len(support)//2,device='cuda').repeat_interleave(2)
    with torch.no_grad(): before=float(fp32_loss(model,support,labels,p,criterion))
    assert torch.equal(p.detach(),init)
    rows=[];losses=[];skips=[];nonzero=0;t0=time.time()
    save(directory/'configuration.json',dict(domain=out.name,selection=selection,seed=seed,lr=lr,
         steps=steps,initial_prompt_hash=tensor_hash(init),frozen_model_hash=frozen_hash,
         objective='FP32 hardest Triplet margin 0.1 only',optimizer='Adam; only external prompt',
         augmentation='flip, pad/crop +-10, random erasing; no colour jitter',
         note='No evaluation feedback, checkpoint selection, or support-set replacement.'))
    for step in range(1,steps+1):
        ids=rng.permutation(len(support)//2)
        idx=[]
        for pid in ids: idx.extend((2*pid+rng.permutation(2)).tolist())
        ii=torch.tensor(idx,device='cuda')
        images=augment(support[ii],gen); y=labels[ii]
        batch_hash=tensor_hash(images); label_hash=tensor_hash(y)
        loss=fp32_loss(model,images,y,p,criterion)
        assert torch.isfinite(loss),('nonfinite loss',step)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad=p.grad.detach().float()
        finite=bool(torch.isfinite(grad).all()); gn=float(grad.norm()) if finite else None
        if finite and gn>0: nonzero+=1
        scale_before=scaler.get_scale()
        scaler.step(optimizer);scaler.update()
        skipped=scaler.get_scale()<scale_before
        if skipped: skips.append(step)
        row=dict(step=step,loss=float(loss),gradient_norm=gn,gradient_finite=finite,skipped=skipped,
                 scale=float(scaler.get_scale()),images_sha256=batch_hash,labels_sha256=label_hash,
                 delta_rms=float((p.detach()-init).square().mean().sqrt()))
        log.write(json.dumps(row,allow_nan=False)+'\n');losses.append(float(loss))
        if step in (100,300):
            score=evaluate(model,ds,p.detach())
            r=dict(step=step,**score,gain=score['mAP']-base['mAP'],draw=selection['draw'],
                   seed=seed,lr=lr,effective_identities=selection['effective_identities'])
            rows.append(r);save(directory/'retrieval.partial.json',rows)
            torch.save(p.detach().cpu(),directory/f'prompt-{step}.pt')
            print('EVAL',json.dumps(r),flush=True)
        if step%50==0: print('PROGRESS',directory.name,step,round(float(loss),6),flush=True)
    with torch.no_grad(): after=float(fp32_loss(model,support,labels,p,criterion))
    adam_updates=int(optimizer.state[p]['step'])
    assert adam_updates==steps-len(skips)
    assert state_hash(model.state_dict().items())==frozen_hash,'base model modified'
    # Replay the saved prompt from disk through the full evaluation path.
    restored=torch.load(directory/'prompt-300.pt',map_location='cuda',weights_only=True)
    assert torch.equal(restored,p.detach())
    result=dict(rows=rows,support_loss_before=before,support_loss_after=after,
                augmented_loss_first20=float(np.mean(losses[:20])),augmented_loss_last20=float(np.mean(losses[-20:])),
                successful_updates=adam_updates,skipped_steps=skips,nonzero_gradient_steps=nonzero,
                delta_rms=float((p.detach()-init).square().mean().sqrt()),frozen_hash=frozen_hash,
                saved_prompt_exact=True,elapsed_seconds=time.time()-t0)
    save(directory/'complete.json',result);log.close()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--domain',choices=['cuhk03','grid','viper','ilids'],required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--checkpoint',default='experiments/s2_vpt_long/checkpoint-12000')
    parser.add_argument('--preflight-only',action='store_true')
    a=parser.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    ds,manifest,selections=load_data(a.domain)
    if a.preflight_only:
        print(json.dumps(dict(domain=a.domain,manifest={k:v for k,v in manifest.items() if 'manifest' not in k},
                              annotations=[r['annotation'] for r in selections])));return
    if (out/'complete.json').exists():raise FileExistsError(out/'complete.json')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args=torch.load(Path(a.checkpoint)/'training_args.bin',map_location='cpu',weights_only=False)
    assert args.model_type=='vpt' and not args.source_all_images
    assert a.domain not in args.source_domains.split(',')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=VPTReIDModel(args).to(device='cuda',dtype=torch.float16)
    state=torch.load(Path(a.checkpoint)/'pytorch_model.bin',map_location='cpu',weights_only=True)
    load_exact(model,state,'cuda');del state
    model.requires_grad_(False).eval()
    init=model.prompt.detach().clone()
    frozen_hash=state_hash(model.state_dict().items())
    save(out/'data_manifest.json',manifest);save(out/'selections.json',selections)
    save(out/'provenance.json',dict(checkpoint=a.checkpoint,
         checkpoint_sha256=hashlib.sha256((Path(a.checkpoint)/'pytorch_model.bin').read_bytes()).hexdigest(),
         frozen_hash=frozen_hash,initial_prompt_hash=tensor_hash(init),
         source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                       ['scripts/tune_vpt_fewshot.py','scripts/oracle_prompt.py','adapters/warm_start_reid.py']},
         torch=torch.__version__,predeclared=dict(selections=3,optimization_seeds=[42,43],
                       learning_rates=[1e-4,1e-3],primary_lr=1e-4,steps=300,eval_steps=[100,300],
                       k=16,unit='image',method='random',split=0)))
    baseline=evaluate(model,ds,init)
    if a.domain=='cuhk03':assert abs(baseline['mAP']-57.0725679397583)<1e-6,baseline
    save(out/'baseline.json',baseline);print('BASELINE',a.domain,json.dumps(baseline),flush=True)
    results=[]
    for selection in selections:
        paths=[p for pair in selection['pairs'] for p in pair]
        support=load_images(paths,'cuda')
        for lr in [1e-4,1e-3]:
            for seed in [42,43]:
                result=run_trial(model,ds,support,init,selection,lr,seed,out,baseline,frozen_hash)
                results.append(result);save(out/'all_trials.partial.json',results)
        del support
    # Evaluation of the immutable base at the end must reproduce the start.
    final_base=evaluate(model,ds,init)
    assert final_base==baseline,(baseline,final_base)
    save(out/'complete.json',dict(domain=a.domain,baseline=baseline,final_base=final_base,trials=results,
         scope='Three random image-budget selections, two optimization seeds, two fixed LRs; split 0 only.'))
    print('DOMAIN_COMPLETE',a.domain,flush=True)


if __name__=='__main__':main()
