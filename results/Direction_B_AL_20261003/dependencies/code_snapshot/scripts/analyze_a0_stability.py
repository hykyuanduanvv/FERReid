"""Summarize only the authorized two-seed, 300-step stability audit."""
import argparse,json,hashlib,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch

def read(p): return json.loads(p.read_text())
def events(p): return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.')
    a=ap.parse_args(); root=Path(a.root).resolve(); dest=root/'results/a0_stability_20261001'
    dest.mkdir(parents=True,exist_ok=True)
    runs={}; summaries=[]; verification={}; initial={}
    for seed in [42,43]:
        for variant in ['ln','ln_fp32']:
            name=f'step3_{variant}_s{seed}'; run=root/'experiments'/name
            es=events(run/'numerics.jsonl'); assert es[-1]['event']=='complete',name
            state=read(run/'trainer_state.json'); assert state['global_step']==300
            initial[name]=next(e for e in es if e['event']=='initialization')
            scaler_init=next(e for e in es if e['event']=='scaler_initialization')
            assert scaler_init['state']['scale']==128 and scaler_init['dynamic']
            batches={e['step']:e for e in es if e['event']=='batch'}
            opts={e['step']:e for e in es if e['event']=='optimizer'}
            assert set(batches)==set(range(1,301)) and set(opts)==set(range(1,301))
            by={event:{e['step']:e for e in es if e['event']==event} for event in
                ['generator','features','triplet_gradient','counterfactual_fp32_gradient','parameter_gradient','losses']}
            val={int(e['step']):e for e in state['log_history'] if 'val_cuhk03_mAP' in e}
            assert set(val)=={100,200,300},(name,val.keys())
            evaluations=[]
            for s in [100,200,300]:
                ckpt=run/f'checkpoint-{s}'
                intervention=read(ckpt/'diagnostic_intervention.json')
                assert intervention['variant']==variant and intervention['seed']==seed and intervention['initial_loss_scale']==128
                opt=torch.load(ckpt/'optimizer.pt',map_location='cpu',weights_only=True)
                adam_counts=dict(Counter(int(v['step']) for v in opt['state'].values() if 'step' in v)); del opt
                updates=sum(not opts[i]['skipped'] for i in range(1,s+1))
                assert set(adam_counts)=={updates},(name,s,updates,adam_counts)
                scaler=torch.load(ckpt/'scaler.pt',map_location='cpu',weights_only=True)
                assert scaler['scale']==opts[s]['scale']
                f=by['features'][s]; g=by['generator'][s]; grad=by['triplet_gradient'][s]
                evaluations.append(dict(step=s,mAP=val[s]['val_cuhk03_mAP'],rank1=val[s]['val_cuhk03_rank1'],
                    successful_updates=updates,skipped_updates=s-updates,adam_update_counts=adam_counts,
                    scaler=scaler,feature_std=f['std'],prompt_rms=g['prompt']['rms'],
                    amp_zero_distance=f['amp']['zero_fraction'],fp32_zero_distance=f['fp32']['zero_fraction'],
                    actual_zero_distance=f['fp32' if variant=='ln_fp32' else 'amp']['zero_fraction'],
                    unscaled_triplet_gradient_zero=grad['zero_fraction'],unscaled_triplet_gradient_finite=grad['finite'],
                    unscaled_triplet_gradient_norm=grad['norm'],triplet=by['losses'][s]['triplet'],icl=by['losses'][s]['icl'],
                    fp32_hard_gap=f['fp32']['hard_gap'],positive_distance=f['fp32']['positive_mean'],negative_distance=f['fp32']['negative_mean']))
            runs[name]=dict(events=es,batches=batches,optimizer=opts,by=by)
            summaries.append(dict(run=name,variant=variant,seed=seed,evaluations=evaluations,
                skipped_steps=[s for s,e in opts.items() if e['skipped']],
                first_observed_all_zero_triplet_gradient=next((s for s,e in sorted(by['triplet_gradient'].items()) if e['zero_fraction']==1),None)))
            sub=dest/name; sub.mkdir(exist_ok=True)
            for f in ['numerics.jsonl','trainer_state.json']: shutil.copy2(run/f,sub/f)
            for s in [100,200,300]: shutil.copy2(run/f'checkpoint-{s}'/'diagnostic_intervention.json',sub/f'intervention_step{s}.json')
        n1=f'step3_ln_s{seed}'; n2=f'step3_ln_fp32_s{seed}'
        assert initial[n1]['all_parameters_sha256']==initial[n2]['all_parameters_sha256']
        comparisons={}
        for key in ['images_sha256','labels_sha256','cpu_rng_sha256','cuda_rng_sha256']:
            different=[s for s in range(1,301) if runs[n1]['batches'][s][key]!=runs[n2]['batches'][s][key]]
            comparisons[key]=dict(matching_steps=300-len(different),different_steps=different)
            assert not different,(seed,key,different)
        fa=torch.load(root/'experiments'/n1/'features_step001.pt',map_location='cpu',weights_only=True)['features']
        fb=torch.load(root/'experiments'/n2/'features_step001.pt',map_location='cpu',weights_only=True)['features']
        verification[str(seed)]=dict(initialization_sha256=initial[n1]['all_parameters_sha256'],batches=comparisons,
            first_feature_max_abs_difference=float((fa.float()-fb.float()).abs().max()))
    assert initial['step3_ln_s42']['all_parameters_sha256']!=initial['step3_ln_s43']['all_parameters_sha256']
    # Compare seed-42 setup/input against step 2 without claiming bitwise-identical optimization.
    prior={}
    for variant in ['ln','ln_fp32']:
        old=root/'experiments'/('step2_'+variant)
        if not (old/'numerics.jsonl').exists(): continue
        es=events(old/'numerics.jsonl'); ini=next(e for e in es if e['event']=='initialization')
        new=f'step3_{variant}_s42'; b={e['step']:e for e in es if e['event']=='batch'}
        match={key:sum(b[s][key]==runs[new]['batches'][s][key] for s in range(1,101))
               for key in ['images_sha256','labels_sha256','cpu_rng_sha256','cuda_rng_sha256']}
        val=[e for e in read(old/'trainer_state.json')['log_history'] if 'val_cuhk03_mAP' in e][-1]
        prior[variant]=dict(initialization_match=ini['all_parameters_sha256']==initial[new]['all_parameters_sha256'],
            first100_matching_steps=match,previous_100_mAP=val['val_cuhk03_mAP'],previous_initial_scale=65536,
            note='Changing initial scale also changes which updates are skipped and finite-precision arithmetic; it is not an isolated test of skipped-update count.')
    descriptive=[]; paired=[]
    for s in [100,200,300]:
        for variant in ['ln','ln_fp32']:
            scores=[next(e for e in r['evaluations'] if e['step']==s)['mAP'] for r in summaries if r['variant']==variant]
            descriptive.append(dict(step=s,variant=variant,mean=float(np.mean(scores)),minimum=min(scores),maximum=max(scores),
                                    n_seeds=2,note='Descriptive only; two seeds do not establish robust generalization.'))
        for seed in [42,43]:
            score={r['variant']:next(e for e in r['evaluations'] if e['step']==s)['mAP'] for r in summaries if r['seed']==seed}
            paired.append(dict(step=s,seed=seed,fp32_minus_amp=score['ln_fp32']-score['ln']))
    def save(name,value): (dest/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    save('summary.json',summaries); save('matching_verification.json',verification)
    save('step2_comparison.json',prior); save('descriptive_statistics.json',dict(by_variant=descriptive,paired_differences=paired))
    save('manifest.json',dict(scope='Authorized stability step only; no automatic continuation',
         training='Market1501+MSMT17 train only; CUHK03 fixed 500 validation IDs',
         seeds=[42,43],variants=['ln','ln_fp32'],global_steps=300,initial_loss_scale=128,dynamic_scaling=True,
         initializations=initial,source_hashes={str(p.relative_to(root)):sha(p) for p in
             [root/'scripts/diagnose_a0_stability.py',root/'scripts/analyze_a0_stability.py',root/'plans/a0_stability.tasks']},
         notes=['Detailed features/gradients sampled before attempted updates, validation after 100/200/300 global steps',
                'Every attempted optimizer update logged and checked against Adam checkpoint state',
                'AMP-zero-distance diagnostic is counterfactual for FP32-loss arms',
                'No target-context utility claim follows from stable training alone']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r in summaries:
        name=r['run']; label=r['variant']+' seed '+str(r['seed']); by=runs[name]['by']
        s=sorted(by['features']); f=[by['features'][i] for i in s]
        axes[0,0].plot([e['step'] for e in r['evaluations']],[e['mAP'] for e in r['evaluations']],marker='o',label=label)
        axes[0,1].plot(s,[x['std'] for x in f],label=label)
        axes[1,0].plot(s,[100*by['triplet_gradient'][i]['zero_fraction'] for i in s],label=label)
        axes[1,1].plot(s,[by['generator'][i]['prompt']['rms'] for i in s],label=label)
    axes[0,0].set(title='CUHK03 validation mAP (500 identities)',xticks=[100,200,300]); axes[0,0].legend()
    axes[0,1].set(title='Normalized image-feature std',yscale='log')
    axes[1,0].set(title='Zero Triplet feature-gradient entries (%)',ylim=(-2,102))
    axes[1,1].set(title='Prompt RMS')
    for ax in axes.flat: ax.grid(alpha=.2); ax.set_xlabel('Global training step')
    fig.savefig(dest/'stability_overview.png',dpi=180); plt.close(fig)
    print(json.dumps(dict(runs=[dict(run=r['run'],skips=r['skipped_steps'],
        mAP=[e['mAP'] for e in r['evaluations']],std=[e['feature_std'] for e in r['evaluations']],
        gradient_zero=[e['unscaled_triplet_gradient_zero'] for e in r['evaluations']]) for r in summaries],
        paired_differences=paired,previous_step=prior),indent=2))

if __name__=='__main__': main()
