"""Frozen Protocol-2 source recipe; explicit JSON arguments, no common.sh expansion."""
import sys, json, hashlib, argparse, dataclasses, subprocess, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, transformers
from adapters.args_reid import ReIDTrainingArguments
from adapters.trainer_reid import DGReIDTrainer, _get_dataset_cls
from adapters.config_reid import DOMAIN_CONFIG

DOMAINS = ['market1501', 'msmt17', 'cuhksysu', 'cuhk03']
TAG = 'p2b_20261002_v1'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def save(p,v):
    Path(p).write_text(json.dumps(v,indent=2,default=str,allow_nan=False))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--target',choices=DOMAINS)
    a=ap.parse_args()
    root=Path('experiments')/TAG
    if a.prepare:
        root.mkdir(exist_ok=False)
        old=torch.load('experiments/s2_vpt_long/checkpoint-12000/training_args.bin',map_location='cpu',weights_only=False)
        config={f.name:getattr(old,f.name) for f in dataclasses.fields(ReIDTrainingArguments) if f.init}
        config=json.loads(json.dumps(config,default=str))
        # Runtime device/Accelerate instances are regenerated, never copied from pickle.
        config['accelerator_config']={}
        config.update(do_eval=False,eval_strategy='no',val_domains='none',val_max_ids=0,
            max_steps=12000,save_steps=3000,save_total_limit=2,logging_steps=50,
            disable_tqdm=True,report_to=[],num_train_ids=0,local_rank=-1,
            logging_dir=None,run_name=None,resume_from_checkpoint=None)
        tasks=[]
        for target in DOMAINS:
            c=dict(config,source_domains=','.join(d for d in DOMAINS if d!=target),
                   target_domains=target,output_dir=str(root/('source_'+target)))
            path=root/('source_'+target+'.json')
            save(path,c)
            parsed=transformers.HfArgumentParser(ReIDTrainingArguments).parse_json_file(str(path.resolve()))[0]
            assert parsed.model_type=='vpt' and parsed.source_all_images is False
            save(root/('effective_'+target+'.json'),parsed.to_dict())
            tasks.append(f'{TAG}_source_{target} | "$PYTHON" -u scripts/p2b_source.py --target {target}')
        Path('plans/p2b_source_vpt.tasks').write_text('\n'.join(tasks)+'\n')
        save(root/'source_recipe.json',dict(tag=TAG,source_steps=12000,seed=42,
            reference_checkpoint='experiments/s2_vpt_long/checkpoint-12000',
            rationale='Fixed historical strong recipe, no target metric feedback; source length selected for this run.',
            checkpoint_interval=3000,retain_last=2,loss='hardest Triplet + 0.01 WPA; no ICL/CE',
            caveat='CUHK03 influenced historical method development. One source seed only.'))
        print('SOURCE_CONFIGS_VALIDATED',flush=True)
        return
    assert a.target
    path=root/('source_'+a.target+'.json')
    args=transformers.HfArgumentParser(ReIDTrainingArguments).parse_json_file(str(path.resolve()))[0]
    assert set(args.source_domains.split(','))==set(DOMAINS)-{a.target}
    assert args.model_type=='vpt' and args.source_all_images is False
    assert args.val_domains=='none' and args.max_steps==12000
    out=Path(args.output_dir)
    out.mkdir(exist_ok=False)
    data=[]
    for domain in args.source_domains.split(','):
        ds=_get_dataset_cls(domain)(root=DOMAIN_CONFIG['data_root'],combineall=False,verbose=False)
        assert not {r[0] for r in ds.train}&{r[0] for r in ds.query+ds.gallery}
        data.append(dict(domain=domain,images=len(ds.train),identities=ds.num_train_pids,
            train_manifest=ds.train,train_only=True))
    save(out/'source_data.json',data)
    save(out/'provenance.json',dict(target=a.target,config_sha256=sha(path),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256={p:sha(p) for p in ['scripts/p2b_source.py','scripts/train_reid.py',
            'adapters/trainer_reid.py','adapters/baseline_model.py','adapters/reid_dataset.py','ops/losses.py']},
        torch=torch.__version__,cuda=torch.version.cuda))
    transformers.set_seed(args.seed)
    trainer=DGReIDTrainer(args=args,device='cuda')
    save(out/'effective_arguments.json',trainer.args.to_dict())
    start=time.perf_counter()
    trainer.train()
    trainer.save_state()
    ckpt=out/'checkpoint-12000'
    assert trainer.state.global_step==12000 and (ckpt/'pytorch_model.bin').exists()
    opt=torch.load(ckpt/'optimizer.pt',map_location='cpu',weights_only=True)
    counts=sorted({int(v['step']) for v in opt['state'].values() if 'step' in v})
    save(out/'complete.json',dict(target=a.target,checkpoint=str(ckpt.resolve()),
        checkpoint_sha256=sha(ckpt/'pytorch_model.bin'),training_args_sha256=sha(ckpt/'training_args.bin'),
        trainer_steps=trainer.state.global_step,optimizer_step_counts=counts,
        elapsed_seconds=time.perf_counter()-start))
    print('SOURCE_COMPLETE',a.target,flush=True)

if __name__=='__main__': main()

