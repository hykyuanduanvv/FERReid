"""Validate matching and summarize the completed 2x2 a0 audit."""
import argparse, json, hashlib, shutil, zipfile
from collections import Counter
from pathlib import Path
import numpy as np

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.')
    a=ap.parse_args(); root=Path(a.root).resolve()
    variants=['control','fp32','ln','ln_fp32']; streams={}; states={}
    for v in variants:
        run=root/'experiments'/('step2_'+v)
        streams[v]=[json.loads(line) for line in (run/'numerics.jsonl').read_text().splitlines()]
        assert streams[v][-1]['event']=='complete',v+' did not complete'
        states[v]=read(run/'trainer_state.json')
        assert states[v]['global_step']==100
    initial={v:next(r for r in s if r['event']=='initialization') for v,s in streams.items()}
    assert len({r['all_parameters_sha256'] for r in initial.values()})==1,'initial weights differ'
    assert len({r['trainable_parameters_sha256'] for r in initial.values()})==1
    batches={v:{r['step']:r for r in s if r['event']=='batch'} for v,s in streams.items()}
    assert all(len(b)==100 for b in batches.values())
    fields=['images_sha256','labels_sha256','cpu_rng_sha256','cuda_rng_sha256']
    comparison={}
    for f in fields:
        different=[step for step in range(1,101) if len({batches[v][step][f] for v in variants})!=1]
        comparison[f]=dict(matching_steps=100-len(different),different_steps=different)
        assert not different,(f,different)
    import torch
    first_features={v:torch.load(root/'experiments'/('step2_'+v)/'features_step001.pt',map_location='cpu',weights_only=True)['features'] for v in variants}
    first_diff={v:float((x.float()-first_features['control'].float()).abs().max()) for v,x in first_features.items()}
    dest=root/'results/a0_factorial_20261001'; dest.mkdir(parents=True,exist_ok=True)
    summary=[]; histories={}
    for v in variants:
        events=streams[v]
        byevent={name:{r['step']:r for r in events if r['event']==name} for name in
                 ['generator','features','triplet_gradient','counterfactual_fp32_gradient','losses','optimizer']}
        histories[v]=byevent
        g=byevent['generator'][100]; f=byevent['features'][100]; grad=byevent['triplet_gradient'][100]
        val=[r for r in states[v]['log_history'] if 'val_cuhk03_mAP' in r]
        assert val, 'missing validation '+v
        times=[r for r in states[v]['log_history'] if 'train_runtime' in r]
        ckpt=root/'experiments'/('step2_'+v)/'checkpoint-100'
        scaler=torch.load(ckpt/'scaler.pt',map_location='cpu',weights_only=True)
        optimizer=torch.load(ckpt/'optimizer.pt',map_location='cpu',weights_only=True)
        adam_steps=dict(Counter(int(s['step']) for s in optimizer['state'].values() if 'step' in s))
        del optimizer
        summary.append(dict(variant=v,validation_mAP=val[-1]['val_cuhk03_mAP'],validation_rank1=val[-1]['val_cuhk03_rank1'],
            step100_feature_std=f['std'],step100_prompt_rms=g['prompt']['rms'],step100_hidden_rms=g['raw_hidden']['rms'],
            step100_head_input_rms=g['head_input']['rms'],step100_amp_zero_distance=f['amp']['zero_fraction'],
            step100_fp32_zero_distance=f['fp32']['zero_fraction'],step100_triplet_gradient_zero=grad['zero_fraction'],
            step100_triplet_gradient_finite=grad['finite'],step100_triplet_gradient_norm=grad['norm'],
            step100_counterfactual_fp32_gradient_zero=byevent['counterfactual_fp32_gradient'][100]['zero_fraction'],
            step100_triplet=byevent['losses'][100]['triplet'],step100_icl=byevent['losses'][100]['icl'],
            step100_fp32_positive_distance=f['fp32']['positive_mean'],step100_fp32_negative_distance=f['fp32']['negative_mean'],
            step100_fp32_hard_gap=f['fp32']['hard_gap'],
            first_observed_amp_zero_above_50pct=next((s for s,x in sorted(byevent['features'].items()) if x['amp']['zero_fraction']>.5),None),
            first_observed_all_zero_triplet_gradient=next((s for s,x in sorted(byevent['triplet_gradient'].items()) if x['zero_fraction']==1),None),
            sampled_optimizer_skips=[s for s,x in byevent['optimizer'].items() if x['skipped']],
            adam_update_counts=adam_steps,final_scaler_state=scaler,
            train_runtime_seconds=times[-1]['train_runtime'] if times else None))
        sub=dest/('step2_'+v); sub.mkdir(exist_ok=True)
        for name in ['numerics.jsonl','trainer_state.json']:
            shutil.copy2(root/'experiments'/('step2_'+v)/name,sub/name)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (dest/'matching_verification.json').write_text(json.dumps(dict(initialization=initial,batches=comparison,initial_feature_max_diff=first_diff,
        note='Matching hashes verify exact initialized parameters, augmented input tensors, labels and RNG states at all 100 steps.'),indent=2))
    (dest/'manifest.json').write_text(json.dumps(dict(scope='Step 2 only; seed 42, 100 steps; no long-run or P3 claim',
        runtime_interventions='LayerNorm before Head and/or FP32 Triplet, installed by diagnose_a0_factorial.py; production defaults unchanged',
        training='Market1501+MSMT17 train only; CUHK03 fixed 500 validation identities',
        source_hashes={str(p.relative_to(root)):sha(p) for p in [root/'scripts/diagnose_a0_factorial.py',root/'plans/a0_factorial.tasks',root/'scripts/analyze_a0_factorial.py']},
        trace_points='Steps 1-10 and every 5 through 100; features/generator measured before attempted optimizer update; validation after 100 global training steps, including AMP-skipped updates',
        limitations=['One seed and short horizon','Unscaled Triplet feature gradient excludes other loss terms',
                     'LayerNorm changes input scale/centering and optimization; does not prove improved context utility',
                     'Feature tensors already use visual-forward precision; FP32 Triplet does not make the whole model FP32']),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for v in variants:
        s=sorted(histories[v]['features']); f=[histories[v]['features'][i] for i in s]
        g=[histories[v]['generator'][i] for i in s]; grad=[histories[v]['triplet_gradient'][i] for i in s]
        axes[0,0].plot(s,[x['prompt']['rms'] for x in g],label=v)
        axes[0,1].plot(s,[x['std'] for x in f],label=v)
        axes[0,2].plot(s,[100*x['amp']['zero_fraction'] for x in f],label=v)
        axes[1,0].plot(s,[100*x['zero_fraction'] for x in grad],label=v)
        axes[1,1].plot(s,[x['fp32']['hard_gap'] for x in f],label=v)
    axes[0,0].set(title='Prompt RMS'); axes[0,0].set_yscale('symlog',linthresh=.01)
    axes[0,1].set(title='Normalized image-feature std',yscale='log')
    axes[0,2].set(title='Zero off-diagonal distances under AMP (%)',ylim=(-2,102))
    axes[1,0].set(title='Zero Triplet feature-gradient entries (%)',ylim=(-2,102))
    axes[1,1].set(title='FP32 hardest-negative minus hardest-positive')
    axes[1,2].bar(variants,[r['validation_mAP'] for r in summary]); axes[1,2].set(title='CUHK03 mAP after 100 training steps')
    for ax in axes.flat: ax.grid(alpha=.2); ax.set_xlabel('Training step')
    axes[1,2].set_xlabel('Variant'); axes[0,0].legend()
    fig.savefig(dest/'early_collapse.png',dpi=180); plt.close(fig)
    print(json.dumps(dict(summary=summary,matching=comparison),indent=2))

if __name__=='__main__': main()
