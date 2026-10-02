"""5000-update VPT active selection; B1 audit is required, no LLM or source retraining."""
import sys,json,hashlib,argparse,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.active_vpt_b1 import (FORMAL,build_model,fit_prompt,digest_json,
    load_data,evaluate,save,load_images,state_hash,tensor_hash)
from adapters.active_vpt_selection import METHODS,annotate_selected
from adapters.config_reid import NO_CAMERA_DOMAINS


def b1_path(domain): return Path(f'experiments/b1_{domain}_20261002_v2')


def check_b1(domain):
    root=b1_path(domain)
    required=['complete.json','fresh_replay.json','legacy_replay.json','selection_audit.json']
    for name in required: assert (root/name).exists(),(domain,name,'B1 missing')
    assert json.loads((root/'fresh_replay.json').read_text())['baseline_exact']
    assert json.loads((root/'legacy_replay.json').read_text())['all_300_input_and_gradient_logs_exact']
    p=json.loads((root/'provenance.json').read_text())
    for name,digest in p['code_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,('B1 code changed',name)
    return root,p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--domain',required=True,choices=['grid','viper','ilids','cuhk03'])
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--output-root',default='experiments/b2_vpt_active_20261002')
    parser.add_argument('--verify',action='store_true')
    a=parser.parse_args();b1,provenance=check_b1(a.domain)
    model,args=build_model(Path(provenance['checkpoint']))
    ds,manifest,_=load_data(a.domain)
    assert a.domain not in args.source_domains.split(',')
    assert digest_json(manifest)==digest_json(json.loads((b1/'data_manifest.json').read_text()))
    initial=model.prompt.detach().clone();frozen=state_hash(model.state_dict().items())
    assert frozen==provenance['frozen_hash'] and tensor_hash(initial)==provenance['initial_prompt_hash']
    baseline=evaluate(model,ds,initial)
    assert baseline==json.loads((b1/'baseline.json').read_text())
    domain_root=Path(a.output_root)/a.domain
    domain_root.mkdir(parents=True,exist_ok=True)
    if a.verify:
        rows=[]
        for method in ('random','facility'):
            trial=domain_root/method/'draw0_seed42'
            expected=json.loads((trial/'complete.json').read_text())
            prompt=torch.load(trial/'prompt-5000.pt',map_location='cuda',weights_only=True)
            actual=evaluate(model,ds,prompt)
            assert all(actual[k]==expected['rows'][-1][k] for k in actual)
            rows.append(dict(method=method,metrics=actual,exact=True))
        save(domain_root/'fresh_replay.json',dict(baseline_exact=True,trials=rows,frozen_exact=True))
        print('B2_FRESH_REPLAY',a.domain,flush=True);return
    assert a.method
    out=domain_root/a.method
    out.mkdir(exist_ok=False)
    save(out/'baseline.json',baseline)
    configuration=dict(FORMAL)
    configuration.update(stage='B2',status='B2 running',domain=a.domain,method=a.method,
        checkpoint=provenance['checkpoint'],frozen_hash=frozen,b1_provenance=provenance,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(out/'configuration.json',configuration)
    selections=[r for r in json.loads((b1/'selections.json').read_text()) if r['method']==a.method]
    assert len(selections)==3
    rows=[];start=time.perf_counter()
    for selected in selections:
        reconstructed=annotate_selected(ds.train,selected['anchors'],
            has_cameras=a.domain not in NO_CAMERA_DOMAINS,domain=a.domain,
            split=0,annotation_seed=selected['annotation_seed'])
        assert digest_json(reconstructed['pairs'])==selected['ordered_support_hash']
        support=load_images([p for pair in selected['pairs'] for p in pair],'cuda') if selected['pairs'] else torch.empty(0,device='cuda')
        for seed in FORMAL['optimization_seeds']:
            trial=out/f"draw{selected['draw']}_seed{seed}"
            result=fit_prompt(model,ds,support,initial,selected,trial,baseline,frozen,
                steps=FORMAL['steps'],eval_steps=FORMAL['eval_steps'],seed=seed,lr=FORMAL['lr'])
            rows.append(dict(domain=a.domain,method=a.method,draw=selected['draw'],seed=seed,
                effective_identities=selected['effective_identities'],annotation=selected['annotation'],
                anchor_set_hash=selected['anchor_set_hash'],support_hash=selected['ordered_support_hash'],
                directory=str(trial),result=result))
            save(out/'trials.partial.json',rows)
            print('TRIAL_COMPLETE',a.domain,a.method,selected['draw'],seed,flush=True)
        del support
    final=evaluate(model,ds,initial)
    assert final==baseline and state_hash(model.state_dict().items())==frozen
    save(out/'complete.json',dict(domain=a.domain,method=a.method,baseline=baseline,
        final_baseline=final,trials=rows,elapsed_seconds=time.perf_counter()-start))
    print('B2_METHOD_COMPLETE',a.domain,a.method,flush=True)


if __name__=='__main__':main()
