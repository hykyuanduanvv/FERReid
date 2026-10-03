"""Full-gallery retrieval for diagnostic prompts; no training or model selection."""
import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import dataclass
import torch
import numpy as np
import transformers
from adapters.args_reid import ReIDTrainingArguments
from scripts.prompt_trace import dataset,save
from scripts.context_sensitivity import build_model,load_images,features
from torchreid.metrics import evaluate_rank

@dataclass
class TraceEvalArguments:
    trace_directory: str = ''

@torch.no_grad()
def main():
    args,t=transformers.HfArgumentParser((ReIDTrainingArguments,TraceEvalArguments)).parse_args_into_dataclasses()
    trace=Path(t.trace_directory); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'retrieval_full.json').exists(): raise FileExistsError(out/'retrieval_full.json')
    manifest=json.loads((trace/'manifest.json').read_text()); ckpt=manifest['checkpoint']
    torch.backends.cuda.matmul.allow_tf32=False
    model=build_model(args,'cuda',ckpt)
    model.load_state_dict(torch.load(Path(ckpt)/'pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
    model.requires_grad_(False).eval()
    bank=torch.load(trace/'diagnostic_prompts.pt',map_location='cpu',weights_only=True)
    domains=list(manifest['domains']); rows=[]; noise=[]
    for dst in domains:
        ds=dataset(dst)
        qi=load_images([r[0] for r in ds.query],'cuda'); gi=load_images([r[0] for r in ds.gallery],'cuda')
        qpid=np.array([r[1] for r in ds.query]); gpid=np.array([r[1] for r in ds.gallery])
        qcam=np.array([r[2] for r in ds.query]); gcam=np.array([r[2] for r in ds.gallery])
        valid=~((qpid[:,None]==gpid[None,:])&(qcam[:,None]==gcam[None,:]))
        same=(qpid[:,None]==gpid[None,:])&valid; different=(qpid[:,None]!=gpid[None,:])&valid
        ref=None
        for src in [dst]+[d for d in domains if d!=dst]+[dst+'_masked']:
            p=bank[src].to('cuda')
            qf=features(model,qi,p,bs=96); gf=features(model,gi,p,bs=96)
            d=(1-qf@gf.T).cpu().numpy()
            order=np.argsort(np.where(valid,d,np.inf),axis=1)[:,:10]
            if ref is None:
                ref=d.copy(); reforder=order.copy(); refq=qf.clone()
                repeat=features(model,qi[:32],p,bs=32)
                same_bs=features(model,qi[:32],p,bs=32)
                noise.append(dict(domain=dst,same_batch_repeat_max=float((repeat-same_bs).abs().max()),
                                  different_batch_size_max=float((repeat-qf[:32]).abs().max())))
            cmc,mAP=evaluate_rank(d,qpid,gpid,qcam,gcam,max_rank=10)
            top10=float(np.mean([len(set(a)&set(b))/10 for a,b in zip(order,reforder)]))
            row=dict(target=dst,context=src,mAP=float(mAP)*100,rank1=float(cmc[0])*100,
                n_query=len(ds.query),n_gallery=len(ds.gallery),
                mean_query_feature_change=float((qf-refq).norm(dim=1).mean()),
                positive_distance=float(d[same].mean()),negative_distance=float(d[different].mean()),
                margin=float(d[different].mean()-d[same].mean()),
                top1_changed=float((order[:,0]!=reforder[:,0]).mean()),top10_overlap=top10,
                distance_change_rms=float(np.sqrt(((d-ref)**2).mean())),
                note='Full query/gallery, split 0, one fixed context per source domain; diagnostic only')
            rows.append(row); print(json.dumps(row),flush=True)
            save(out/'retrieval_full.partial.json',rows)
        del qi,gi,refq,qf,gf
    save(out/'feature_repeatability.json',noise)
    save(out/'retrieval_full.json',rows)

if __name__=='__main__': main()
