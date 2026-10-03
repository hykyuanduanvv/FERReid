"""Observational context-path audit; held-out target scores are diagnostics, not tuning.

Identity-disjoint context buckets define probe folds. Context tokens are pooled over
question slots (not treated as aligned images); query slots remain aligned.
"""
import os, sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from adapters.args_reid import ReIDTrainingArguments
from adapters.config_reid import DOMAIN_CONFIG, NUM_SPLITS, NO_CAMERA_DOMAINS
from adapters.trainer_reid import _get_dataset_cls, _seed_all, _subsample_ids
from scripts.context_sensitivity import build_model, load_images, features
from torchreid.metrics import evaluate_rank

@dataclass
class TraceArguments:
    checkpoint: str = ''
    domains: str = 'viper,grid,ilids,cuhk03'
    contexts_per_bucket: int = 10
    trace_k: int = 16
    trace_seed: int = 7101
    retrieval_max_ids: int = 100
    skip_retrieval: bool = False

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))

class Moments:
    def __init__(self):
        self.data = {}
    def add(self, name, domain, x):
        x = x.detach().double().cpu().numpy()
        key = (name, domain)
        if key not in self.data:
            self.data[key] = [0, np.zeros_like(x), np.zeros(x.shape[-1]), 0.]
        z = self.data[key]
        z[0] += 1; z[1] += x
        z[2] += (x*x).reshape(-1,x.shape[-1]).sum(0)
        z[3] = max(z[3], float(np.abs(x).max()))
    def summary(self):
        out = {}
        for name in sorted({x[0] for x in self.data}):
            vals = [z for (k,d),z in self.data.items() if k == name]
            n = sum(z[0] for z in vals)
            mean = sum(z[1] for z in vals)/n
            energy = sum(z[2].sum() for z in vals)/n
            within = sum(z[2].sum() - (z[1]*z[1]).sum()/z[0] for z in vals)/n
            between = sum(z[0]*((z[1]/z[0]-mean)**2).sum() for z in vals)/n
            dims = sum(z[2] for z in vals)
            top = np.argsort(dims)[-4:][::-1]
            out[name] = dict(n=n, total_energy=float(energy), absolute_variation=float(within+between),
                within_variation=float(within), between_variation=float(between),
                rho=float((within+between)/max(energy,1e-30)),
                between_fraction_of_variation=float(between/max(within+between,1e-30)),
                top4_energy_fraction=float(dims[top].sum()/max(dims.sum(),1e-30)),
                top4_dimensions=top.tolist(), max_abs=max(z[3] for z in vals))
        return out

def dataset(name):
    return _get_dataset_cls(name)(root=DOMAIN_CONFIG['data_root'], verbose=False,
                                 **({'split_id':0} if name in NUM_SPLITS else {}))

def pools(ds):
    result = defaultdict(list)
    for row in ds.train:
        result[int(row[1])].append(row)
    result = {k:v for k,v in result.items() if len({r[0] for r in v}) >= 2}
    assert not ({r[0] for r in ds.train} & {r[0] for r in ds.query+ds.gallery})
    return result

def pairs_for(pool, ids, rng, has_cameras):
    pairs=[]
    for pid in ids:
        rows=pool[int(pid)]; a=rows[rng.randint(len(rows))]
        other=[r for r in rows if r[0]!=a[0] and (not has_cameras or r[2]!=a[2])]
        if not other:
            other=[r for r in rows if r[0]!=a[0]]
        b=other[rng.randint(len(other))]; pairs.append([a[0],b[0]])
    return pairs

def schedule(k, n, seed):
    # Match the production RNG stream exactly; same pair slots and yes/no sequence for all contexts.
    _seed_all(seed); pairs=[]; truth=[]
    for _ in range(n):
        positive=torch.randint(0,2,(1,)).item()
        if positive:
            i=torch.randint(0,k,(1,)).item(); a,b=2*i,2*i+1
        else:
            a=torch.randint(0,2*k,(1,)).item(); b=torch.randint(0,2*k,(1,)).item()
        pairs.append([a,b]); truth.append(int(a//2==b//2))
    return pairs,truth

@torch.no_grad()
def encode(model, imgs, pair_ids, truth):
    f=model.encoder_copy.forward_features(imgs.to(model.encoder.patch_embed.proj.weight.dtype))['x_norm_clstoken']
    f=f[torch.tensor(pair_ids,device=f.device)].reshape(-1,2*model.hidden_size)
    projected=model.mm_projector(f.to(model.mm_projector.visual_proj.weight.dtype))
    yes=model.tokenizer.convert_tokens_to_ids('yes'); no=model.tokenizer.convert_tokens_to_ids('no')
    s=[]
    for a in truth: s.extend([0]*model.args.num_id_tokens+[yes if a else no])
    ids=torch.tensor([s],device=f.device)
    e=model.lm.get_input_embeddings()(ids).clone()
    mask=(torch.arange(len(s),device=f.device)%(model.args.num_id_tokens+1))<model.args.num_id_tokens
    e[:,mask]=projected.reshape(1,-1,model.lm.config.hidden_size).to(e.dtype)
    e=torch.cat([e, model.query_embeddings[None].to(e.dtype)],1)
    # Calling the decoder avoids materializing unnecessary vocabulary logits.
    hidden=model.lm(inputs_embeds=e,use_cache=False,output_hidden_states=True).hidden_states
    h=hidden[-1][:,-model.query_embeddings.size(0):]
    p=model._prompts_from_hidden(h).reshape(1,model.num_layers,model.args.num_vpt_tokens,-1)
    return hidden,p

def probes(bank, labels, buckets):
    labels=np.array(labels); buckets=np.array(buckets); result={}
    # Repeated permutations give a null distribution; preprocessing fits training data only.
    for name, vals in bank.items():
        x=np.stack(vals); acc=[]; null=[]
        for b in range(3):
            tr=buckets!=b; te=~tr
            if len(np.unique(labels[tr]))<2: continue
            def fit(y):
                pipe=make_pipeline(StandardScaler(),PCA(n_components=min(16,tr.sum()-1,x.shape[1]),svd_solver='full'),
                                   LogisticRegression(C=1.,max_iter=2000))
                pipe.fit(x[tr],y)
                return balanced_accuracy_score(labels[te],pipe.predict(x[te]))
            acc.append(float(fit(labels[tr])))
            for seed in range(5):
                null.append(float(fit(np.random.RandomState(100*b+seed).permutation(labels[tr]))))
        result[name]=dict(fold_accuracy=acc, mean_accuracy=float(np.mean(acc)),
                         shuffled_train_label_mean=float(np.mean(null)), shuffled_train_label_sd=float(np.std(null)))
    return result

@torch.no_grad()
def main():
    args,t=transformers.HfArgumentParser((ReIDTrainingArguments,TraceArguments)).parse_args_into_dataclasses()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise FileExistsError(out/'summary.json')
    _seed_all(t.trace_seed)
    torch.backends.cuda.matmul.allow_tf32=False
    model=build_model(args,'cuda',t.checkpoint)
    state=torch.load(Path(t.checkpoint)/'pytorch_model.bin',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True); del state
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    names=t.domains.split(','); qn=model.query_embeddings.size(0)
    pair_ids,truth=schedule(t.trace_k,args.num_icl_samples,t.trace_seed)
    records=[]; sets={}
    for name in names:
        ds=dataset(name); pool=pools(ds); ids=np.array(sorted(pool))
        buckets=np.array_split(np.random.RandomState(t.trace_seed).permutation(ids),3)
        assert all(len(b)>=t.trace_k for b in buckets),(name,[len(b) for b in buckets])
        sets[name]=dict(identity_buckets=[b.tolist() for b in buckets], pool_images=len(ds.train))
        for b,ids in enumerate(buckets):
            for draw in range(t.contexts_per_bucket):
                rng=np.random.RandomState(t.trace_seed+b*100+draw)
                selected=rng.choice(ids,t.trace_k,replace=False)
                pairs=pairs_for(pool,selected,rng,name not in NO_CAMERA_DOMAINS)
                records.append(dict(domain=name,bucket=b,draw=draw,pids=selected.tolist(),pairs=pairs))
    # An independent source context selects masked dimensions, without fitting on probe test examples.
    sp=pools(dataset('market1501')); rng=np.random.RandomState(t.trace_seed+999)
    cp=pairs_for(sp,rng.choice(sorted(sp),t.trace_k,False),rng,True)
    hh,_=encode(model,load_images([p for pair in cp for p in pair],'cuda'),pair_ids,truth)
    mask_dims=hh[-1][0,-qn:].float().square().sum(0).topk(4).indices
    del hh
    save(out/'manifest.json',dict(checkpoint=str(Path(t.checkpoint).resolve()), trace_args=vars(t),
         architecture={k:getattr(args,k) for k in ['backbone','num_vpt_tokens','num_id_tokens','num_icl_samples','prompt_mode']},
         selection_unit='identity (diagnostic controlled slots; not annotation-budget comparison)',
         masks_calibrated_on='one Market1501 train context',masked_dimensions=mask_dims.tolist(),
         pair_slots=pair_ids,answers=truth,domains=sets,contexts=records,calibration_pairs=cp,
         torch=torch.__version__,eval_mode=True,tf32=False))
    # Same images for every context; identity/camera distances in this fixed subset are diagnostic.
    grid=dataset('grid'); q=grid.query[:16]
    matching=[next(r for r in grid.gallery if r[1]==x[1] and r[2]!=x[2]) for x in q]
    fixed_rows=q+matching; fixed_imgs=load_images([r[0] for r in fixed_rows],'cuda')
    moments=Moments(); bank=defaultdict(list); prompt_bank={}; labels=[]; folds=[]; rows=[]; noise=[]
    for i,r in enumerate(records):
        imgs=load_images([p for pair in r['pairs'] for p in pair],'cuda')
        hh,p=encode(model,imgs,pair_ids,truth)
        if i==0:
            # Rebuilt generator must match the checkpoint's production forward path.
            _seed_all(t.trace_seed)
            reference=model(imgs.reshape(t.trace_k,2,*imgs.shape[1:]),labels=torch.arange(t.trace_k,device='cuda'))['prompts'][:1]
            # Production returns prompts AFTER casting to the visual encoder's dtype.
            diff=float((reference.float()-p.to(reference.dtype).float()).abs().max())
            print('PRODUCTION_PROMPT_MAX_DIFF',diff,flush=True)
            assert diff<1e-5,('generator mismatch',diff)
            for rep in range(3):
                hr,pr=encode(model,imgs,pair_ids,truth)
                noise.append(dict(prompt_max_abs=float((pr-p).abs().max()),
                    query_max_abs=float((hr[-1][:,-qn:]-hh[-1][:,-qn:]).abs().max())))
                del hr,pr
        for layer,h in enumerate(hh):
            query=h[0,-qn:].float()
            ctx=h[0,:-qn].reshape(args.num_icl_samples,args.num_id_tokens+1,-1)[:,:args.num_id_tokens].float().mean(0)
            for key,val in [(f'query_{layer:02}',query),(f'context_{layer:02}',ctx)]:
                moments.add(key,r['domain'],val)
                bank[key].append(val.mean(0).cpu().numpy())
        last=hh[-1][:,-qn:].clone(); last[:,:,mask_dims]=0
        masked=model._prompts_from_hidden(last).reshape_as(p)
        for key,val in [('prompt',p[0]),('prompt_at_encoder_dtype',p[0].to(model.encoder.patch_embed.proj.weight.dtype)),('query_last_masked',last[0]),('prompt_masked',masked[0])]:
            moments.add(key,r['domain'],val)
            bank[key].append(val.reshape(-1,val.size(-1)).mean(0).float().cpu().numpy())
        f=features(model,fixed_imgs,p,bs=32).float()
        moments.add('fixed_image_features',r['domain'],f)
        dist=(1-f[:16]@f[16:].T).cpu().numpy()
        own=np.diag(dist); other=dist[~np.eye(16,dtype=bool)]
        if i==0: refdist=dist.copy(); reff=f.clone(); reforder=np.argsort(refdist,axis=1)
        rows.append(dict(**{k:r[k] for k in ['domain','bucket','draw']},
           feature_l2_change=float((f-reff).norm(dim=1).mean()),
           positive_distance=float(own.mean()),negative_distance=float(other.mean()),
           margin=float(other.mean()-own.mean()),
           top1_changed=float((np.argmin(dist,1)!=np.argmin(refdist,1)).mean()),
           top5_overlap=float(np.mean([len(set(a[:5])&set(b[:5]))/5 for a,b in zip(np.argsort(dist,axis=1),reforder)])),
           distance_change_rms=float(np.sqrt(((dist-refdist)**2).mean()))))
        if r['bucket']==0 and r['draw']==0:
            prompt_bank[r['domain']]=p.cpu(); prompt_bank[r['domain']+'_masked']=masked.cpu()
        labels.append(names.index(r['domain'])); folds.append(r['bucket'])
        del hh,imgs,f,p,masked
        print(f'TRACE {i+1}/{len(records)} {r["domain"]} bucket={r["bucket"]}',flush=True)
    save(out/'geometry.json',moments.summary()); save(out/'feature_effects.json',rows)
    save(out/'repeatability.json',noise)
    torch.save(prompt_bank,out/'diagnostic_prompts.pt')
    np.savez_compressed(out/'pooled_representations.npz',**{k:np.stack(v) for k,v in bank.items()},labels=labels,buckets=folds)
    del moments
    save(out/'probes.json',probes(bank,labels,folds))
    retrieval=[]
    if not t.skip_retrieval:
        for dst in names:
            ds=_subsample_ids(dataset(dst),t.retrieval_max_ids) if t.retrieval_max_ids else dataset(dst)
            qi=load_images([r[0] for r in ds.query],'cuda'); gi=load_images([r[0] for r in ds.gallery],'cuda')
            ref=None
            # Own reference first; fixed query/gallery and FP32 distances for every intervention.
            conditions=[dst]+[n for n in names if n!=dst]+[dst+'_masked']
            for src in conditions:
                p=prompt_bank[src].to('cuda')
                qf=features(model,qi,p,bs=96); gf=features(model,gi,p,bs=96)
                d=(1-qf@gf.T).cpu().numpy()
                if ref is None: ref=d.copy()
                cmc,mAP=evaluate_rank(d,np.array([r[1] for r in ds.query]),np.array([r[1] for r in ds.gallery]),
                    np.array([r[2] for r in ds.query]),np.array([r[2] for r in ds.gallery]),max_rank=10)
                rr=dict(target=dst,context=src,mAP=float(mAP)*100,rank1=float(cmc[0])*100,
                    n_query=len(ds.query),n_gallery=len(ds.gallery),distance_change_rms=float(np.sqrt(((d-ref)**2).mean())),
                    note='single fixed context/split; capped identities; exploratory diagnostic only')
                retrieval.append(rr); print('RETRIEVAL',json.dumps(rr),flush=True)
                save(out/'retrieval.json',retrieval)
            del qi,gi
    save(out/'summary.json',dict(contexts=len(records),noise_max=max(x['prompt_max_abs'] for x in noise),
         checkpoint=t.checkpoint,complete=True,warning='Variance/domain separability are not useful-information measures.'))

if __name__=='__main__': main()
