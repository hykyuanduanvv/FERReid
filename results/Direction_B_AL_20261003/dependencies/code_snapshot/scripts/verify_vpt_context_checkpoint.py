"""Reload the saved warm-start subclass and reproduce a final retrieval result."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from adapters.warm_start_reid import WarmStartReIDModel, load_exact
from adapters.trainer_reid import DGReIDTrainer, cosine_distmat
from adapters.config_reid import DOMAIN_CONFIG
from scripts.train_vpt_context import make_prompt, save, final_evaluation
from torchreid.metrics import evaluate_rank


@torch.no_grad()
def main():
    run=Path(sys.argv[1]); ckpt=run/'checkpoint-300'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args=torch.load(ckpt/'training_args.bin',map_location='cpu',weights_only=False)
    # Match construction-time dtype for non-persistent LM RoPE buffers too.
    model=WarmStartReIDModel(args).to(device='cuda',dtype=torch.float16)
    state=torch.load(ckpt/'pytorch_model.bin',map_location='cpu',weights_only=True)
    load_exact(model,state,'cuda')
    model.requires_grad_(False).eval()
    manifest=json.loads((ckpt/'warm_start_manifest.json').read_text())
    assert model.frozen_hash()==manifest['frozen_hash']
    t=DGReIDTrainer.__new__(DGReIDTrainer)
    t.args=args; t.model=model; t._device='cuda'
    t._data_root=DOMAIN_CONFIG['data_root']; t._dataset_cache={}
    t.initial_frozen_hash=manifest['frozen_hash']
    if not (run/'retrieval.json').exists():
        # A fresh checkpoint load has no Accelerate AMP/FP32-output wrapper.
        # Do not resume training; repeat only the interrupted evaluation.
        final_evaluation(t)
    p,ctx=make_prompt(t,'cuhk03',0)
    bank=torch.load(run/'evaluation_prompts.pt',map_location='cpu',weights_only=True)
    difference=float((p.cpu()-bank['cuhk03_0']).abs().max())
    assert difference==0, difference
    ds,_=t._load_split('cuhk03',0,args.val_max_ids)
    q,qp,qc=t._extract(ds.query,p,1600)
    g,gp,gc=t._extract(ds.gallery,p,1600)
    cmc,ap=evaluate_rank(cosine_distmat(q,g,'cuda'),qp.numpy(),gp.numpy(),qc.numpy(),gc.numpy(),max_rank=10)
    target=next(r for r in json.loads((run/'retrieval.json').read_text()) if r['condition']=='own' and r['draw']==0)
    result=dict(prompt_max_difference=difference,mAP=float(ap)*100,rank1=float(cmc[0])*100,
                expected_mAP=target['mAP'],expected_rank1=target['rank1'],frozen_hash=model.frozen_hash())
    assert abs(result['mAP']-target['mAP'])<1e-6 and abs(result['rank1']-target['rank1'])<1e-6,result
    save(run/'checkpoint_reload_verification.json',result)
    with (run/'audit.jsonl').open('a') as f:
        f.write(json.dumps(dict(event='checkpoint_evaluation_complete',step=300,
                   note='Fresh checkpoint evaluation, no Trainer forward wrapper; no additional optimizer updates'))+'\n')
        f.write(json.dumps(dict(event='complete',step=300))+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
