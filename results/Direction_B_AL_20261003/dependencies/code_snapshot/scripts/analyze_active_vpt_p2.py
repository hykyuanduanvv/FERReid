"""Summarize only the frozen P2 run; complete audit requires every planned condition."""
import sys,json,csv,statistics,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p2b_source import DOMAINS,TAG,sha
from adapters.active_vpt_selection import METHODS

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def write_csv(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--domains',default=','.join(DOMAINS))
    args=ap.parse_args();domains=args.domains.split(',')
    assert domains and len(set(domains))==len(domains) and set(domains)<=set(DOMAINS)
    full=set(domains)==set(DOMAINS)
    root=Path('experiments')/TAG;out=Path('results')/TAG
    if not full:out=out/('partial_'+str(len(domains))+'_targets')
    out.mkdir(parents=True,exist_ok=True)
    details=[];curves=[];summary=[];baselines={};sources=[];replays=[]
    for domain in domains:
        sources.append(read(root/('source_'+domain)/'complete.json'))
        replay=read(root/'adapt'/domain/'fresh_replay.json')
        assert replay['baseline_pass'] and all(r['passed'] for r in replay['trials'])
        replays.append(dict(domain=domain,**replay))
        baselines[domain]=read(root/'preflight'/domain/'baseline.json')
        for method in METHODS:
            result=read(root/'adapt'/domain/method/'complete.json')
            assert len(result['trials'])==6
            assert {(r['draw'],r['seed']) for r in result['trials']}=={(d,s) for d in range(3) for s in (42,43)}
            for trial in result['trials']:
                t=trial['result'];directory=Path(trial['directory']);last=t['rows'][-1]
                assert t['actual_updates']==5000 or t['status']=='insufficient_support_kept_baseline'
                if t['actual_updates']==5000: assert [r['step'] for r in t['rows']]==[100,300,1000,3000,5000]
                details.append(dict(domain=domain,method=method,draw=trial['draw'],seed=trial['seed'],
                    mAP=last['mAP'],rank1=last['rank1'],rank5=last['rank5'],status=t['status'],
                    actual_updates=t['actual_updates'],attempts=t['attempts'],
                    amp_skips=len(t.get('skipped_attempts',[])),effective_identities=trial['effective_identities'],
                    duplicate_anchors=trial['annotation']['n_dup'],failed_anchors=trial['annotation']['n_fail'],
                    anchor_set_hash=trial['anchor_set_hash'],support_hash=trial['support_hash'],
                    prompt_sha256=sha(directory/'prompt-5000.pt'),
                    training_log_sha256=sha(directory/'training.jsonl') if (directory/'training.jsonl').exists() else None,
                    train_seconds=t.get('train_seconds',0),evaluation_seconds=t.get('evaluation_seconds',0),
                    directory=str(directory)))
                for point in [dict(step=0,**baselines[domain])]+t['rows']:
                    curves.append(dict(domain=domain,method=method,draw=trial['draw'],seed=trial['seed'],
                        step=point['step'],mAP=point['mAP'],rank1=point['rank1'],actual_updates=t['actual_updates']))
        random=[r for r in details if r['domain']==domain and r['method']=='random']
        random_map=statistics.mean(r['mAP'] for r in random);random_rank=statistics.mean(r['rank1'] for r in random)
        for method in METHODS:
            rr=[r for r in details if r['domain']==domain and r['method']==method]
            m=statistics.mean(r['mAP'] for r in rr);rank=statistics.mean(r['rank1'] for r in rr)
            paired=[r['mAP']-next(x['mAP'] for x in random if (x['draw'],x['seed'])==(r['draw'],r['seed'])) for r in rr]
            summary.append(dict(domain=domain,method=method,mAP_mean=m,mAP_sd=statistics.stdev(r['mAP'] for r in rr),
                rank1_mean=rank,rank1_sd=statistics.stdev(r['rank1'] for r in rr),
                mAP_vs_random=m-random_map,mAP_vs_frozen=m-baselines[domain]['mAP'],
                rank1_vs_random=rank-random_rank,rank1_vs_frozen=rank-baselines[domain]['rank1'],
                paired_mAP_gain_sd=statistics.stdev(paired),
                unique_anchor_sets=len({r['anchor_set_hash'] for r in rr}),
                unique_support_sets=len({r['support_hash'] for r in rr}),
                effective_identities_mean=statistics.mean(r['effective_identities'] for r in rr)))
    assert len(details)==42*len(domains)
    write_csv(out/'all_trials.csv',details);write_csv(out/'summary.csv',summary);write_csv(out/'learning_curves.csv',curves)
    save(out/'summary.json',dict(baselines=baselines,methods=summary,source_checkpoints=sources))
    failures=[str(p) for p in (root/'adapt').glob('*/*/draw*/attempt_*') if not (p/'complete.json').exists()]
    audit=dict(planned_conditions=168,completed_conditions=len(details),included_domains=domains,
        actual_updates=sum(r['actual_updates'] for r in details),
        insufficient_support=sum(r['status']!='complete' for r in details),
        amp_skips=sum(r['amp_skips'] for r in details),
        failed_attempt_directories=failures,independent_reloads=replays,
        train_seconds=sum(r['train_seconds'] for r in details),
        evaluation_seconds=sum(r['evaluation_seconds'] for r in details),
        complete=full,available_fold_audit_passed=True)
    save(out/'audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for domain,ax in zip(domains,axes.flat):
        for method in METHODS:
            points=[r for r in curves if r['domain']==domain and r['method']==method]
            steps=sorted({r['step'] for r in points})
            ax.plot(steps,[statistics.mean(r['mAP'] for r in points if r['step']==s) for s in steps],label=method)
        ax.axhline(baselines[domain]['mAP'],color='black',linestyle='--',label='frozen VPT')
        ax.set(title=domain,xlabel='Actual optimizer updates',ylabel='mAP (%)');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7,ncol=2);fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    lines=['Protocol-2 Direction B: '+('completed' if full else 'PARTIAL - requested four-fold protocol NOT complete')+' fixed-budget VPT adaptation','',
        'Four leave-one-domain-out source models; train-only sources; full target query/gallery.',
        'Source: 12000 Trainer steps, seed 42; individual optimizer counts recorded in source provenance.',
        'Target: k=16 anchors, 3 selection/pair draws x 2 optimizer seeds; fixed 5000 actual updates.',
        'Main endpoint was not selected using target performance. LLM/VICP not used.',
        'Source loss: Triplet + 0.01 WPA. Adaptation: external prompt only, FP32 Triplet, Adam lr=1e-4.',
        '', 'domain | method | mAP mean +/- SD | R1 mean +/- SD | mAP gain random | mAP gain frozen']
    for domain in domains:
        b=baselines[domain];lines.append(f"{domain} | frozen | {b['mAP']:.4f} | {b['rank1']:.4f} | - | 0")
        for r in [r for r in summary if r['domain']==domain]:
            lines.append(f"{domain} | {r['method']} | {r['mAP_mean']:.4f} +/- {r['mAP_sd']:.4f} | {r['rank1_mean']:.4f} +/- {r['rank1_sd']:.4f} | {r['mAP_vs_random']:+.4f} | {r['mAP_vs_frozen']:+.4f}")
    lines+=['','Limitations:',
        '- CUHK03 influenced earlier method development; no claim that all four domains were untouched.',
        '- Only one source training seed. Six trials do not imply six independent labeled samples.',
        '- MSMT17 train/test PIDs use separate official namespaces; no global raw-ID mapping was available.',
        '- SYSU lacks real cameras; official duplicate annotations, including a query/gallery crop, were retained.',
        '- Existing facility limits n_eval=4000, n_cand=8000 were retained; no new pool truncation.',
        '- SD is descriptive across the six runs. Anchor/support diversity is reported separately.',
        '- Independent prompt reloads cover predeclared random/facility draw0 seed42 in each domain.',
        '- Failure attempts are retained; insufficient support is reported as baseline fallback, not 5000 updates.',
        '',json.dumps(audit,ensure_ascii=False,indent=2)]
    (out/'REPORT.txt').write_text('\n'.join(lines))
    print('P2_ANALYSIS_COMPLETE',out,flush=True)

if __name__=='__main__':main()

