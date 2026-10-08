from pathlib import Path
import os,sys,json,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import dataclass,field
import numpy as np
import torch,transformers
from adapters.args_reid import ReIDTrainingArguments
from adapters.active.image_store import features
from adapters.active.loop import default_prompt
from adapters.active.pseudo import PoolGraph
from adapters.baseline_model import load_checkpoint_model
from scripts.eval_active import load_split
@dataclass
class ExportArgs:
    checkpoint:str=field(default='')
    domains:str=field(default='msmt17')
    cache_max:int=field(default=0)
def main():
    args,a=transformers.HfArgumentParser((ReIDTrainingArguments,ExportArgs)).parse_args_into_dataclasses()
    torch.set_num_threads(4); transformers.set_seed(args.seed)
    model=load_checkpoint_model(args,'cuda',a.checkpoint)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    for domain in a.domains.split(','):
        dest=out/(domain+'_epoch0.npz')
        if dest.exists(): raise RuntimeError('Refusing to overwrite '+str(dest))
        start=time.time(); split=load_split(domain,'cuda',a.cache_max,args.eval_num_workers)
        f=features(model,split.pool,default_prompt(model))
        graph=PoolGraph(f,30,6,0.6,4)
        arrays={}
        for eps,tag in [(0.55,'055'),(0.6,'060'),(0.65,'065')]:
            graph.eps=eps; arrays['labels_eps_'+tag]=graph.cluster()
        arrays.update(features=f.float().cpu().numpy(),labels=arrays['labels_eps_060'],pids=np.asarray(split.pool_pids,dtype=np.int64),cams=np.asarray(split.pool_cams,dtype=np.int64),paths=np.asarray(split.pool_paths,dtype=str))
        if not np.isfinite(arrays['features']).all(): raise ValueError('Non-finite features')
        n=len(arrays['labels'])
        if any(len(v)!=n for v in arrays.values()): raise ValueError('Mismatched lengths')
        np.savez_compressed(dest,**arrays)
        labels=arrays['labels']; report={'domain':domain,'checkpoint':a.checkpoint,'backbone':'clip_b16','shape':list(arrays['features'].shape),'clusters':int(len(np.unique(labels[labels>=0]))),'outliers':int((labels<0).sum()),'seconds':time.time()-start,'sha256':hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),'pid_usage':'stored for offline evaluation/oracle only; not used in feature extraction or clustering'}
        (out/(domain+'_epoch0.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        del graph,f,split; torch.cuda.empty_cache()
if __name__=='__main__': main()
