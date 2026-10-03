import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.prompt_trace import Moments

root=Path('experiments/trace_old_vicp')
m=json.loads((root/'manifest.json').read_text())
audits={}
for domain,desc in m['domains'].items():
    groups=desc['identity_buckets']
    assert all(not(set(a)&set(b)) for i,a in enumerate(groups) for b in groups[i+1:])
    paths=[]
    for bucket in range(3):
        sub=[r for r in m['contexts'] if r['domain']==domain and r['bucket']==bucket]
        assert len(sub)==10
        assert all(set(r['pids'])<=set(groups[bucket]) for r in sub)
        paths.append({p for r in sub for pair in r['pairs'] for p in pair})
    assert all(not(a&b) for i,a in enumerate(paths) for b in paths[i+1:])
    audits[domain]=dict(identity_counts=list(map(len,groups)),observed_images=list(map(len,paths)),
                        train_test_identity_overlap=0,train_test_image_overlap=0)
rng=np.random.RandomState(10); values=rng.randn(12,5,7)
acc=Moments()
for i,x in enumerate(values): acc.add('x',str(i//4),torch.tensor(x))
got=acc.summary()['x']
expected=((values-values.mean(0))**2).sum((1,2)).mean()/(values**2).sum((1,2)).mean()
assert abs(got['rho']-expected)<1e-12
assert abs(got['within_variation']+got['between_variation']-got['absolute_variation'])<1e-12
ckpt=Path(m['checkpoint'])
saved=torch.load(ckpt/'training_args.bin',map_location='cpu',weights_only=False)
fields=['source_domains','source_all_images','val_domains','model_type','backbone','seed','max_steps',
        'num_icl_samples','num_vpt_tokens','lora_layers','lora_rank','fp16','weight_decay']
metadata={k:getattr(saved,k,'not present in historical args') for k in fields}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
result=dict(split_audit=audits,moment_test='passed against direct NumPy definition and variance decomposition',
            checkpoint_args=metadata,checkpoint_sha256=sha(ckpt/'pytorch_model.bin'),
            source_hashes={str(p):sha(p) for p in [Path('scripts/prompt_trace.py'),Path('scripts/analyze_prompt_trace.py'),Path('scripts/eval_trace_prompts.py')]})
(root/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
g=json.loads((root/'geometry.json').read_text()); p=json.loads((root/'probes_repeated.json').read_text())
for k in ['context_00','context_28','query_00','query_01','query_03','query_18','query_27','query_28','prompt','prompt_at_encoder_dtype','query_last_masked','prompt_masked','fixed_image_features']:
    print(k,json.dumps(g[k]),'probe',p.get(k,{}).get('mean_accuracy'))
f=json.loads((root/'feature_effects.json').read_text())
for name in ['feature_l2_change','positive_distance','negative_distance','margin','top1_changed','top5_overlap','distance_change_rms']:
    a=np.array([r[name] for r in f]); print('FEATURE',name,a.min(),a.mean(),a.max())
