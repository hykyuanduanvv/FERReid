"""Complete, hierarchical summary of predeclared few-shot VPT prompt trials."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/vpt_fewshot_20261001'


def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summaries=[];allrows=[];audits={};paired=[]
    for domain in ['cuhk03','grid','viper','ilids']:
        d=ROOT/'experiments'/f'step5_{domain}';complete=read(d/'complete.json')
        baseline=read(d/'baseline.json');selections=read(d/'selections.json');provenance=read(d/'provenance.json')
        for p,h in provenance['source_hashes'].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,('source changed',p)
        replay=read(d/'fresh_replay.json')
        assert replay['exact_reproduction'] and replay['baseline']==baseline
        assert baseline==complete['final_base']
        assert len(complete['trials'])==12
        rows=[];training={};details=[];skip_count=0;updates=0
        dest=OUT/domain;dest.mkdir(exist_ok=True)
        for f in ['baseline.json','selections.json','data_manifest.json','provenance.json','complete.json','fresh_replay.json']:
            shutil.copy2(d/f,dest/f)
        for p in sorted(d.glob('draw*_seed*_lr*')):
            r=read(p/'complete.json');cfg=read(p/'configuration.json')
            logs=[json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()]
            assert len(logs)==300 and [x['step'] for x in logs]==list(range(1,301))
            assert sum(not x['skipped'] for x in logs)==r['successful_updates']
            assert r['frozen_hash']==provenance['frozen_hash']==cfg['frozen_model_hash']
            assert cfg['initial_prompt_hash']==provenance['initial_prompt_hash'] and r['saved_prompt_exact']
            updates+=r['successful_updates'];skip_count+=len(r['skipped_steps'])
            training[p.name]=logs
            for row in r['rows']:rows.append(dict(domain=domain,**row))
            details.append(dict(trial=p.name,draw=cfg['selection']['draw'],seed=cfg['seed'],lr=cfg['lr'],
                support_loss_before=r['support_loss_before'],support_loss_after=r['support_loss_after'],
                augmented_loss_first20=r['augmented_loss_first20'],augmented_loss_last20=r['augmented_loss_last20'],
                delta_rms=r['delta_rms'],nonzero_gradient_steps=r['nonzero_gradient_steps'],
                successful_updates=r['successful_updates'],skipped_steps=r['skipped_steps']))
            pd=dest/p.name;pd.mkdir(exist_ok=True)
            for f in ['training.jsonl','configuration.json','complete.json']:
                shutil.copy2(p/f,pd/f)
        assert len(rows)==24
        for draw in range(3):
            for seed in [42,43]:
                a,b=[training[f'draw{draw}_seed{seed}_lr{lr:g}'] for lr in [1e-4,1e-3]]
                assert all(x['images_sha256']==y['images_sha256'] and x['labels_sha256']==y['labels_sha256']
                           for x,y in zip(a,b)),(domain,draw,seed,'learning-rate arms unmatched')
                paired.append(dict(domain=domain,draw=draw,seed=seed,all_300_augmented_batches_equal=True))
        for lr in [1e-4,1e-3]:
            for step in [100,300]:
                sub=[r for r in rows if r['lr']==lr and r['step']==step]
                drawmeans=[];seeddiffs=[]
                for draw in range(3):
                    sr=sorted([r for r in sub if r['draw']==draw],key=lambda r:r['seed'])
                    assert [r['seed'] for r in sr]==[42,43]
                    drawmeans.append(float(np.mean([r['mAP'] for r in sr])))
                    seeddiffs.append(abs(sr[0]['mAP']-sr[1]['mAP']))
                means=float(np.mean(drawmeans))
                summaries.append(dict(domain=domain,lr=lr,step=step,baseline=baseline['mAP'],
                    mean_mAP=means,gain=means-baseline['mAP'],
                    rank1_mean=float(np.mean([r['rank1'] for r in sub])),
                    selection_mean_sd=float(np.std(drawmeans,ddof=1)),selection_means=drawmeans,
                    selection_mean_min=min(drawmeans),selection_mean_max=max(drawmeans),
                    selection_range=max(drawmeans)-min(drawmeans),
                    optimization_seed_abs_differences=seeddiffs,
                    mean_optimization_seed_abs_difference=float(np.mean(seeddiffs)),
                    individual_min=min(r['mAP'] for r in sub),individual_max=max(r['mAP'] for r in sub),
                    positive_runs=sum(r['gain']>0 for r in sub),
                    effective_identities=[r['effective_identities'] for r in selections],
                    n_selections=3,optimization_seeds_per_selection=2,
                    uncertainty_note='SD across 3 selection means, each averaging 2 optimization seeds; not a confidence interval.'))
        allrows.extend(rows)
        audits[domain]=dict(successful_updates=updates,skipped_steps=skip_count,
                           baseline_reproduced=True,frozen_hash=provenance['frozen_hash'],trial_details=details)
    assert len({v['frozen_hash'] for v in audits.values()})==1
    save(OUT/'summary.json',summaries);save(OUT/'all_results.json',allrows)
    save(OUT/'audit.json',audits);save(OUT/'learning_rate_matching.json',paired)
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    domains=['cuhk03','grid','viper','ilids'];colors=['#4c72b0','#c44e52']
    for j,lr in enumerate([1e-4,1e-3]):
        rows=[next(r for r in summaries if r['domain']==d and r['lr']==lr and r['step']==300) for d in domains]
        axs[0].bar(np.arange(4)+(j-.5)*.36,[r['gain'] for r in rows],width=.36,color=colors[j],label=f'lr={lr:g}')
    axs[0].axhline(0,color='black',linewidth=1)
    axs[0].set_xticks(range(4),['CUHK03','GRID','VIPeR','iLIDS'])
    axs[0].set(ylabel='mAP change vs frozen VPT (points)',title='300-step endpoint, all 6 runs averaged')
    axs[0].legend();axs[0].grid(axis='y',alpha=.25)
    for i,d in enumerate(domains):
        sub=[r for r in allrows if r['domain']==d and r['lr']==1e-4 and r['step']==300]
        for draw in range(3):
            a=[r for r in sub if r['draw']==draw]
            x=i+(draw-1)*.18
            ys=[r['gain'] for r in a]
            axs[1].plot([x,x],ys,color='#888888',linewidth=1)
            axs[1].scatter([x,x],ys,c=['#4c72b0','#dd8452'],s=28)
    axs[1].axhline(0,color='black',linewidth=1)
    axs[1].set_xticks(range(4),['CUHK03','GRID','VIPeR','iLIDS'])
    axs[1].set(ylabel='mAP change vs frozen VPT (points)',title='Primary lr=1e-4: 3 selections x 2 seeds')
    axs[1].grid(axis='y',alpha=.25)
    from matplotlib.lines import Line2D
    axs[1].legend(handles=[Line2D([],[],marker='o',linestyle='',color='#4c72b0',label='seed 42'),
                           Line2D([],[],marker='o',linestyle='',color='#dd8452',label='seed 43')])
    fig.tight_layout();fig.savefig(OUT/'fewshot_results.png',dpi=180);plt.close(fig)
    print(json.dumps([r for r in summaries if r['step']==300]))


if __name__=='__main__':main()
