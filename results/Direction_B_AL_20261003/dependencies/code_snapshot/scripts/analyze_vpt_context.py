"""Summarize all predeclared step-4 runs, including negative results."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/vpt_context_warm_20261001'


def read(p): return json.loads(Path(p).read_text())


def save(p,data):
    Path(p).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    allruns={}; evidence={}; summary=[]; batches={}
    for seed in [42,43]:
        for variant in ['context','shared']:
            name=f'step4_{variant}_s{seed}'; run=ROOT/'experiments'/name
            a=[json.loads(x) for x in (run/'audit.jsonl').read_text().splitlines()]
            assert a[-1]['event']=='complete',name
            opt=[r for r in a if r['event']=='optimizer']
            assert len(opt)==300 and [r['step'] for r in opt]==list(range(1,301))
            batches[name]=[r for r in a if r['event']=='batch']
            assert len(batches[name])==300
            config=read(run/'configuration.json'); gate=read(run/'initial_equivalence.json')
            reload=read(run/'checkpoint_reload_verification.json')
            geo=read(run/'prompt_geometry.json'); rows=read(run/'retrieval.json')
            bank=torch.load(run/'evaluation_prompts.pt',map_location='cpu',weights_only=True)
            base_prompt=torch.load(ROOT/config['arguments']['warm_checkpoint']/'pytorch_model.bin',
                                   map_location='cpu',weights_only=True)['prompt']
            flat=torch.stack([v.flatten().float() for v in bank.values()])
            delta=flat-base_prompt.half().float().flatten()
            total=float(delta.square().mean())
            varying=float((delta-delta.mean(0)).square().mean())
            geo.update(delta_rms=total**.5,context_variation_rms=varying**.5,
                       context_variation_fraction_of_delta_energy=varying/total if total else 0.0,
                       geometry_note='Energy fraction measures variation, not useful information.')
            trainer=read(run/'trainer_state.json')
            vals={r['step']:r['val_mean_mAP'] for r in trainer['log_history'] if 'val_mean_mAP' in r}
            assert set(vals)=={100,200,300}
            initialization=next(r for r in a if r['event']=='initialization')
            for logical,archived in [('scripts/train_vpt_context.py','experiments/step4_training_runner_v1.py'),
                                     ('adapters/warm_start_reid.py','experiments/step4_model_training_v1.py')]:
                assert hashlib.sha256((ROOT/archived).read_bytes()).hexdigest()==config['sources'][logical]
            manifest=read(run/'checkpoint-300/warm_start_manifest.json')
            assert initialization['frozen_hash']==gate['frozen_hash']==geo['frozen_hash']==reload['frozen_hash']==manifest['frozen_hash']
            own=[r['mAP'] for r in rows if r['condition']=='own']
            cross=[r['mAP'] for r in rows if r['condition']=='cross']
            base=next(r['mAP'] for r in rows if r['condition']=='frozen_vpt')
            mean_source=next((r['mAP'] for r in rows if r['condition']=='mean_source_prompt'),None)
            result=dict(run=name,seed=seed,variant=variant,validation=vals,baseline=base,
                own_mean=float(np.mean(own)),own_min=min(own),own_max=max(own),
                own_std=float(np.std(own,ddof=1)) if len(own)>1 else 0.0,
                gain_vs_vpt=float(np.mean(own)-base),
                cross_mean=float(np.mean(cross)) if cross else None,
                specificity=float(np.mean(own)-np.mean(cross)) if cross else None,
                mean_source_prompt=mean_source,geometry=geo,
                successful_updates=sum(not r['skipped'] for r in opt),skipped_steps=[r['step'] for r in opt if r['skipped']],
                actual_context_draws=3, note='Shared output is invariant across all nine generated contexts; scored once.' if variant=='shared' else '')
            summary.append(result)
            evidence[name]=dict(initialization=initialization,gate=gate,reload=reload,config=config,
                 gradient_checks=[r for r in a if r['event']=='retrieval_gradient'])
            allruns[name]=rows
            dest=OUT/name; dest.mkdir(exist_ok=True)
            for f in ['audit.jsonl','configuration.json','initial_equivalence.json','checkpoint_reload_verification.json',
                      'retrieval.json','prompt_geometry.json','contexts.json','trainer_state.json']:
                shutil.copy2(run/f,dest/f)
    matching={}
    for seed in [42,43]:
        c,s=f'step4_context_s{seed}',f'step4_shared_s{seed}'
        assert evidence[c]['initialization']['all_parameter_hash']==evidence[s]['initialization']['all_parameter_hash']
        assert batches[c]==batches[s], f'input/RNG mismatch seed {seed}'
        matching[str(seed)]=dict(initial_parameters_equal=True,all_300_augmented_batches_labels_rng_equal=True)
        ctx=next(r for r in summary if r['run']==c); shared=next(r for r in summary if r['run']==s)
        ctx['gain_vs_shared']=ctx['own_mean']-shared['own_mean']
    save(OUT/'summary.json',summary); save(OUT/'evidence.json',evidence); save(OUT/'matching.json',matching)
    save(OUT/'retrieval_all.json',allruns)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    colors={'context':'#c44e52','shared':'#4c72b0'}
    for r in summary:
        label=f"{r['variant']} seed {r['seed']}"
        vals=r['validation']; x=[100,200,300]; y=[vals[k] for k in x]
        axs[0].plot(x,y,marker='o',color=colors[r['variant']],linestyle='-' if r['seed']==42 else '--',label=label)
    axs[0].set(xlabel='Additional training steps',ylabel='CUHK03 validation mAP (%)',title='Training validation (Trainer AMP)')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
    for i,seed in enumerate([42,43]):
        r=next(r for r in summary if r['seed']==seed and r['variant']=='context')
        s=next(r for r in summary if r['seed']==seed and r['variant']=='shared')
        values=[r['baseline'],s['own_mean'],r['own_mean'],r['cross_mean'],r['mean_source_prompt']]
        axs[1].bar(np.arange(5)+(i-.5)*.34,values,width=.34,label=f'seed {seed}')
    axs[1].set_xticks(np.arange(5),['Frozen\nVPT','Shared\ndelta','Own\ncontext','Cross\ncontext','Mean source\nprompt'])
    axs[1].set(ylabel='CUHK03 mAP (%)',title='300-step fixed endpoint; context means')
    axs[1].legend(); axs[1].grid(axis='y',alpha=.25)
    fig.tight_layout(); fig.savefig(OUT/'warm_start_results.png',dpi=180); plt.close(fig)
    print(json.dumps(summary))


if __name__=='__main__':main()
