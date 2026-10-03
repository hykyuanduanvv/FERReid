"""Protocol-2 VPT: full target splits, audited selectors, fixed prompt updates.

Stages: audit (CPU), preflight, replay-preflight, adapt, verify.
No test-score feedback changes any hyperparameter. Partial trials are retained;
reruns use new numbered attempts, while validated completed trials are reused.
"""
import sys, os, json, hashlib, argparse, time, shutil, subprocess
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torchreid.metrics import evaluate_rank
from adapters.trainer_reid import _get_dataset_cls, cosine_distmat
from adapters.config_reid import DOMAIN_CONFIG, NO_CAMERA_DOMAINS
from adapters.active_vpt_selection import METHODS, annotate_selected, unit_checks
import scripts.active_vpt_b1 as legacy
from scripts.p2b_source import DOMAINS, TAG, sha

ROOT=Path('experiments')/TAG
TOL=2e-4  # percentage points; FP32 block-weighted accumulation vs Cython
EXPECTED={'market1501':(12936,3368,15913),'msmt17':(30248,11659,82161),
          'cuhksysu':(15088,2900,5447),'cuhk03':(7368,1400,5328)}

def save(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False))
    tmp.replace(p)

def read(p): return json.loads(Path(p).read_text())
def digest(v): return legacy.digest_json(v)
def close(a,b):
    for k in ('mAP','rank1','rank5'): assert abs(a[k]-b[k])<=TOL,(k,a,b)

def raw_id(domain,path):
    p=Path(path)
    if domain=='market1501': return p.name.split('_')[0]
    if domain=='cuhk03': return p.parent.name
    if domain=='cuhksysu':
        from adapters.cuhksysu import _pid_of
        base=Path(DOMAIN_CONFIG['data_root'])/'cuhksysu/cuhksysu4reid'
        return str(_pid_of(str(p),str(base/p.relative_to(base).parts[0])))
    # MSMT official train and test use separate local person-id namespaces.
    rel=p.relative_to(Path(DOMAIN_CONFIG['data_root'])/'msmt17/MSMT17_V1')
    return rel.parts[0]+':'+rel.parts[1]

def load_data(domain):
    ds=_get_dataset_cls(domain)(root=DOMAIN_CONFIG['data_root'],combineall=False,verbose=False)
    assert tuple(len(getattr(ds,s)) for s in ('train','query','gallery'))==EXPECTED[domain]
    train={str(Path(r[0]).resolve()) for r in ds.train}
    test={str(Path(r[0]).resolve()) for r in ds.query+ds.gallery}
    assert not train&test
    ti={raw_id(domain,p) for p in train}; vi={raw_id(domain,p) for p in test}
    assert not ti&vi
    manifest=dict(domain=domain,full_test=True,source_all_images=False,
        split_counts={s:dict(images=len(getattr(ds,s)),identities=len({r[1] for r in getattr(ds,s)}))
                      for s in ('train','query','gallery')},path_overlap=0,
        raw_identity_overlap=0 if domain!='msmt17' else None,
        raw_identity_rule=('Official split-scoped identity namespaces; independent global raw-ID proof unavailable. '
             'Numeric train/test PID overlap is expected, not evidence of identity leakage.' if domain=='msmt17'
             else 'Original filename/directory identity before train relabeling'),
        records={s:getattr(ds,s) for s in ('train','query','gallery')})
    manifest['records_sha256']={s:digest(v) for s,v in manifest['records'].items()}
    return ds,manifest

def audit(domain):
    out=ROOT/'data'/domain; out.mkdir(parents=True,exist_ok=True)
    ds,manifest=load_data(domain)
    if (out/'complete.json').exists():
        old=read(out/'manifest.json'); assert old==json.loads(json.dumps(manifest)); return
    start=time.perf_counter(); hashes={}
    for split in ('train','query','gallery'):
        hashes[split]=[dict(path=r[0],sha256=sha(r[0])) for r in getattr(ds,split)]
    sets={s:{r['sha256'] for r in v} for s,v in hashes.items()}
    save(out/'manifest.json',manifest); save(out/'image_sha256.json',hashes)
    overlap=sets['train']&(sets['query']|sets['gallery'])
    if overlap:
        save(out/'content_overlap_gate_failure.json',dict(domain=domain,
            content_groups=len(overlap),hashes=sorted(overlap),passed=False))
        raise AssertionError('train/test image content leakage; see saved evidence')
    lists={}
    if domain=='msmt17':
        base=Path(DOMAIN_CONFIG['data_root'])/'msmt17/MSMT17_V1'
        lists={p.name:sha(p) for p in base.glob('list_*.txt')}
    if domain=='cuhksysu':
        base=Path(DOMAIN_CONFIG['data_root'])/'cuhksysu/cuhksysu4reid'
        for name in ('preparation_summary.json','validation_summary.json','annotation_duplicates.json'):
            shutil.copy2(base/name,out/name)
    save(out/'complete.json',dict(domain=domain,path_overlap=0,content_overlap_train_test=0,
        query_gallery_shared_content=len(sets['query']&sets['gallery']),
        manifest_sha256=sha(out/'manifest.json'),official_list_sha256=lists,
        elapsed_seconds=time.perf_counter()-start))
    print('DATA_AUDIT_COMPLETE',domain,flush=True)

def valid_queries(qp,gp,qc,gc):
    by_pid=defaultdict(set)
    for p,c in zip(gp,gc): by_pid[int(p)].add(int(c))
    return np.array([bool(by_pid[int(p)]-{int(c)}) for p,c in zip(qp,qc)],dtype=bool)

def rank_chunked(q,g,qp,gp,qc,gc,chunk=64,device='cuda'):
    qp,gp,qc,gc=[np.asarray(v) for v in (qp,gp,qc,gc)]
    valid=valid_queries(qp,gp,qc,gc)
    n=int(valid.sum()); assert n>0,'No valid evaluation query'
    cmc_sum=np.zeros(min(10,len(g)),dtype=np.float64); ap_sum=0.
    gallery=g.to(device)
    for start in range(0,len(q),chunk):
        end=min(start+chunk,len(q)); count=int(valid[start:end].sum())
        if not count: continue
        dist=(1-q[start:end].to(device)@gallery.T).float().cpu().numpy()
        cmc,ap=evaluate_rank(dist,qp[start:end],gp,qc[start:end],gc,max_rank=10)
        cmc_sum+=np.asarray(cmc,dtype=np.float64)*count; ap_sum+=float(ap)*count
    return dict(mAP=100*ap_sum/n,rank1=100*cmc_sum[0]/n,rank5=100*cmc_sum[4]/n)

@torch.no_grad()
def evaluate(model,ds,prompt):
    tick=time.perf_counter()
    q,qp,qc=legacy.extract(model,ds.query,prompt)
    g,gp,gc=legacy.extract(model,ds.gallery,prompt)
    extract_seconds=time.perf_counter()-tick; tick=time.perf_counter()
    result=rank_chunked(q,g,qp,gp,qc,gc)
    print('FULL_EVAL',json.dumps(dict(query=len(q),gallery=len(g),
        extraction_seconds=extract_seconds,rank_seconds=time.perf_counter()-tick,**result)),flush=True)
    return result

def cpu_checks():
    result=unit_checks()
    rng=np.random.RandomState(718)
    q=torch.tensor(rng.normal(size=(19,32)),dtype=torch.float32)
    g=torch.tensor(rng.normal(size=(43,32)),dtype=torch.float32)
    q=torch.nn.functional.normalize(q,dim=1);g=torch.nn.functional.normalize(g,dim=1)
    qp=np.array([100]*4+list(range(15)));gp=np.arange(43)%15
    qc=np.arange(19)%2;gc=np.arange(43)%3
    cmc,ap=evaluate_rank((1-q@g.T).numpy(),qp,gp,qc,gc,max_rank=10)
    expected=dict(mAP=float(ap)*100,rank1=float(cmc[0])*100,rank5=float(cmc[4])*100)
    for chunk in (1,4,7,64): close(rank_chunked(q,g,qp,gp,qc,gc,chunk,'cpu'),expected)
    result.update(rank_chunks=[1,4,7,64],unequal_valid_query_counts=True,
        zero_valid_blocks=True,tolerance_percentage_points=TOL)
    save(ROOT/'cpu_checks.json',result); print('CPU_CHECKS_PASS',flush=True)

def build(domain):
    src=ROOT/('source_'+domain); source=read(src/'complete.json')
    checkpoint=Path(source['checkpoint'])
    assert source['trainer_steps']==12000
    assert sha(checkpoint/'pytorch_model.bin')==source['checkpoint_sha256']
    model,args=legacy.build_model(checkpoint)
    assert set(args.source_domains.split(','))==set(DOMAINS)-{domain}
    assert args.source_all_images is False and args.val_domains=='none'
    ds,manifest=load_data(domain)
    assert read(ROOT/'data'/domain/'manifest.json')==json.loads(json.dumps(manifest))
    read(ROOT/'data'/domain/'complete.json')
    return model,args,ds,manifest,source

def fit(model,ds,support,initial,selection,out,baseline,frozen,steps,eval_steps,seed):
    # Reuse audited B1 optimizer verbatim; replace only its evaluation callback.
    # No old source files or previous B1/B2 results are modified.
    legacy.evaluate=evaluate
    return legacy.fit_prompt(model,ds,support,initial,selection,out,baseline,frozen,
        steps=steps,eval_steps=eval_steps,seed=seed,lr=1e-4)

def snapshot(out):
    files=[]
    for directory in ('scripts','adapters','ops'):
        files.extend(str(p) for p in Path(directory).glob('*.py'))
    files.extend(['models.py','custom_trainer.py'])
    hashes={p:sha(p) for p in files}
    for p in files:
        dest=out/'code'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    return hashes

def preflight(domain):
    out=ROOT/'preflight'/domain
    out.mkdir(parents=True,exist_ok=False)
    model,args,ds,manifest,source=build(domain)
    initial=model.prompt.detach().clone();frozen=legacy.state_hash(model.state_dict().items())
    config=dict(legacy.FORMAL,status='P2 frozen',source_steps=12000,source_seed=42,
        selector_facility_existing_limits=dict(n_eval=4000,n_cand=8000),
        rank_chunk=64,rank_tolerance_percentage_points=TOL,full_target_test=True,
        no_camera_domain='cuhksysu',source_domains=args.source_domains,
        checkpoint=source['checkpoint'],checkpoint_sha256=source['checkpoint_sha256'],
        frozen_hash=frozen,initial_prompt_hash=legacy.tensor_hash(initial))
    save(out/'configuration.json',config)
    save(out/'provenance.json',dict(code_sha256=snapshot(out),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        data_manifest_sha256=sha(ROOT/'data'/domain/'manifest.json'),
        torch=torch.__version__,cuda=torch.version.cuda))
    start=time.perf_counter();baseline=evaluate(model,ds,initial)
    baseline_seconds=time.perf_counter()-start
    save(out/'baseline.json',baseline)
    # Actual feature reference using a small query/gallery view, no model tuning.
    q,qp,qc=legacy.extract(model,ds.query[:96],initial)
    gids=set(qp.tolist()); small=[r for r in ds.gallery if r[1] in gids][:2048]
    g,gp,gc=legacy.extract(model,small,initial)
    cmc,ap=evaluate_rank(cosine_distmat(q,g,'cuda'),qp.numpy(),gp.numpy(),qc.numpy(),gc.numpy(),max_rank=10)
    reference=dict(mAP=float(ap)*100,rank1=float(cmc[0])*100,rank5=float(cmc[4])*100)
    actual=rank_chunked(q,g,qp,gp,qc,gc,17);close(actual,reference)
    save(out/'rank_parity.json',dict(reference=reference,chunked=actual,tolerance=TOL))
    metadata=dict(domain=domain,checkpoint_sha256=source['checkpoint_sha256'],
        records_hash=digest(ds.train),frozen_hash=frozen,initial_prompt_hash=legacy.tensor_hash(initial),
        preprocess='legacy flip-summed features then FP32 renormalization')
    features=legacy.feature_cache(model,ds.train,initial,out,metadata)
    selections=legacy.selection_checks(ds,domain,features,out)
    trials=[];seen={}
    for sel in [r for r in selections if r['draw']==0]:
        signature=sel['ordered_support_hash']
        if signature in seen: continue
        seen[signature]=sel['method']
        support=legacy.load_images([p for pair in sel['pairs'] for p in pair],'cuda') if sel['pairs'] else torch.empty(0,device='cuda')
        steps=100 if sel['method']=='random' else 20
        result=fit(model,ds,support,initial,sel,out/('smoke_'+sel['method']),baseline,frozen,steps,[steps],42)
        if result['status']=='complete': assert result['nonzero_gradient_steps']>0
        trials.append(dict(method=sel['method'],steps=steps,directory=str(out/('smoke_'+sel['method'])),result=result))
        del support
    assert legacy.state_hash(model.state_dict().items())==frozen
    random=next(r['result'] for r in trials if r['method']=='random')
    seconds=random.get('milliseconds_per_update',0)/1000
    save(out/'complete.json',dict(domain=domain,baseline=baseline,trials=trials,
        baseline_seconds=baseline_seconds,seconds_per_update=seconds,
        estimated_seconds_per_formal_trial=5000*seconds+5*baseline_seconds))
    print('P2_PREFLIGHT_COMPLETE',domain,flush=True)

def validate_binding(domain,model):
    pre=ROOT/'preflight'/domain
    prov=read(pre/'provenance.json');conf=read(pre/'configuration.json')
    for p,h in prov['code_sha256'].items(): assert sha(p)==h,('code changed after preflight',p)
    assert legacy.state_hash(model.state_dict().items())==conf['frozen_hash']
    assert legacy.tensor_hash(model.prompt)==conf['initial_prompt_hash']
    return pre,conf

def replay(domain,formal=False):
    model,args,ds,manifest,source=build(domain);pre,conf=validate_binding(domain,model)
    baseline=evaluate(model,ds,model.prompt);close(baseline,read(pre/'baseline.json'))
    rows=[]
    for method in ('random','facility'):
        if formal:
            result=read(ROOT/'adapt'/domain/method/'complete.json')
            trial=next(r for r in result['trials'] if r['draw']==0 and r['seed']==42)
            directory=Path(trial['directory']);steps=5000
        else:
            trial=next(r for r in read(pre/'complete.json')['trials'] if r['method']==method)
            directory=Path(trial['directory']);steps=trial['steps']
        expected=read(directory/'complete.json')
        prompt=torch.load(directory/f'prompt-{steps}.pt',map_location='cuda',weights_only=True)
        if expected.get('prompt_hash'): assert legacy.tensor_hash(prompt)==expected['prompt_hash']
        actual=evaluate(model,ds,prompt);close(actual,expected['rows'][-1])
        rows.append(dict(method=method,actual=actual,expected=expected['rows'][-1],
            prompt_sha256=sha(directory/f'prompt-{steps}.pt'),passed=True))
    dest=ROOT/'adapt'/domain if formal else pre
    save(dest/'fresh_replay.json',dict(baseline_pass=True,independent_process=True,
        frozen_hash_exact=True,tolerance_percentage_points=TOL,trials=rows))
    print('P2_REPLAY_COMPLETE',domain,formal,flush=True)

def trial_directory(parent):
    parent.mkdir(parents=True,exist_ok=True)
    attempts=sorted(parent.glob('attempt_*'))
    return parent/f'attempt_{len(attempts)+1:03d}'

def adapt(domain,method):
    model,args,ds,manifest,source=build(domain);pre,conf=validate_binding(domain,model)
    assert read(pre/'fresh_replay.json')['baseline_pass']
    baseline=read(pre/'baseline.json');initial=model.prompt.detach().clone()
    frozen=conf['frozen_hash'];out=ROOT/'adapt'/domain/method
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        assert read(out/'configuration.json')['checkpoint_sha256']==source['checkpoint_sha256'];return
    save(out/'configuration.json',dict(conf,domain=domain,method=method))
    selections=[r for r in read(pre/'selections.json') if r['method']==method]
    assert len(selections)==3
    rows=[];start=time.perf_counter()
    for sel in selections:
        reconstructed=annotate_selected(ds.train,sel['anchors'],has_cameras=domain not in NO_CAMERA_DOMAINS,
            domain=domain,split=0,annotation_seed=sel['annotation_seed'])
        assert digest(reconstructed['pairs'])==sel['ordered_support_hash']
        support=legacy.load_images([p for pair in sel['pairs'] for p in pair],'cuda') if sel['pairs'] else torch.empty(0,device='cuda')
        for seed in (42,43):
            parent=out/f"draw{sel['draw']}_seed{seed}"
            completed=sorted(parent.glob('attempt_*/complete.json'))
            if completed:
                assert len(completed)==1
                directory=completed[0].parent;result=read(completed[0]);c=read(directory/'configuration.json')
                assert c['seed']==seed and c['steps']==5000 and c['frozen_hash']==frozen
                assert c['selection']['ordered_support_hash']==sel['ordered_support_hash']
                prompt=torch.load(directory/'prompt-5000.pt',map_location='cpu',weights_only=True)
                if result.get('prompt_hash'): assert legacy.tensor_hash(prompt)==result['prompt_hash']
            else:
                directory=trial_directory(parent)
                result=fit(model,ds,support,initial,sel,directory,baseline,frozen,5000,[100,300,1000,3000,5000],seed)
            assert result['actual_updates']==5000 or result['status']=='insufficient_support_kept_baseline'
            rows.append(dict(domain=domain,method=method,draw=sel['draw'],seed=seed,
                effective_identities=sel['effective_identities'],annotation=sel['annotation'],
                anchor_set_hash=sel['anchor_set_hash'],support_hash=sel['ordered_support_hash'],
                directory=str(directory),result=result))
            save(out/'trials.partial.json',rows)
            print('P2_TRIAL_COMPLETE',domain,method,sel['draw'],seed,flush=True)
        del support
    assert legacy.state_hash(model.state_dict().items())==frozen
    save(out/'complete.json',dict(domain=domain,method=method,baseline=baseline,trials=rows,
        elapsed_seconds=time.perf_counter()-start))
    print('P2_METHOD_COMPLETE',domain,method,flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',required=True,choices=['cpu-checks','audit','preflight','replay-preflight','adapt','verify'])
    ap.add_argument('--domain',choices=DOMAINS)
    ap.add_argument('--method',choices=METHODS)
    a=ap.parse_args();torch.set_num_threads(4)
    if a.stage=='cpu-checks': cpu_checks();return
    assert a.domain
    if a.stage=='audit': audit(a.domain)
    elif a.stage=='preflight': preflight(a.domain)
    elif a.stage=='replay-preflight': replay(a.domain)
    elif a.stage=='verify': replay(a.domain,True)
    else:
        assert a.method;adapt(a.domain,a.method)

if __name__=='__main__':main()

