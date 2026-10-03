"""Non-selective B1/B2 reporting. Never chooses checkpoints or modifies training."""
import sys,json,csv,hashlib,zipfile,shutil,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from adapters.active_vpt_selection import METHODS

DOMAINS=('grid','viper','ilids','cuhk03')


def read(path):return json.loads(Path(path).read_text())
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))


def report_b1():
    out=Path('results/active_vpt_b1_20261002');out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for domain in DOMAINS:
        root=Path(f'experiments/b1_{domain}_20261002_v2')
        complete=read(root/'complete.json');fresh=read(root/'fresh_replay.json')
        legacy=read(root/'legacy_replay.json');selections=read(root/'selections.json')
        features=read(root/'pool_features.json');audit=read(root/'selection_audit.json')
        timing=next(x['result'] for x in complete['trials'] if x['method']=='historical_random')
        assert fresh['baseline_exact'] and all(x['exact'] for x in fresh['trials'])
        assert legacy['exact_prompt'] and complete['random_repeat_exact']
        row=dict(domain=domain,baseline=complete['baseline'],pool_images=features['shape'][0],
            selector_counts={m:[r['effective_identities'] for r in selections if r['method']==m] for m in METHODS},
            unique_sets=audit['unique_anchor_sets'],all_checks_pass=True,
            smoke_trials=len(complete['trials'])-1,equivalences=complete['equivalences'],
            train_ms_per_update=timing['milliseconds_per_update'],
            evaluation_seconds_2=timing['evaluation_seconds'],
            projected_5000_seconds=timing['train_seconds']*5000/300+timing['evaluation_seconds']*2.5,
            raw_retrieval_norm_error=features['raw_retrieval_norm_max_error'])
        summary.append(row)
        dest=out/domain;dest.mkdir(exist_ok=True)
        for name in ['complete.json','fresh_replay.json','legacy_replay.json','selection_audit.json',
                     'selection_timings.json','selections.json','pool_features.json','provenance.json',
                     'formal_configuration.json','data_manifest.json','baseline.json']:
            shutil.copy2(root/name,dest/name)
    gpu_hours=sum(r['projected_5000_seconds']*(36 if r['domain']=='ilids' else 42)/3600 for r in summary)
    save(out/'summary.json',dict(domains=summary,nominal_trials=168,unique_trials=162,
        projected_gpu_hours=gpu_hours,ideal_four_gpu_hours=gpu_hours/4,
        all_b1_checks_pass=True,b2_not_in_b1=True))
    text=['B1接入核验已通过。此报告中的20步结果仅检查接口，不用于判断选样性能。',
          '所有域冻结VPT与历史基准一致；300步旧随机实验逐步输入、梯度、prompt及成绩精确复现。',
          '新选择器使用带基础prompt的VPT特征，并在FP32重新归一化。选择/标注/优化随机数隔离。',
          '所有选择器完成三次k=16预算审核；每域短程训练及独立进程重载检查通过。',
          'iLIDS的facility_camera与facility等价，B2复用结果；168个表格条件对应162个独立训练。',
          f'按300步拆分计时估算：总GPU小时{gpu_hours:.2f}，四卡理想均衡{gpu_hours/4:.2f}小时；另计调度/重载/运行波动。',
          'B2主终点5000次实际更新，评测100/300/1000/3000/5000；不按测试成绩选终点。']
    for r in summary:
        text.append(f"{r['domain']}: mAP={r['baseline']['mAP']:.5f}; {r['train_ms_per_update']:.2f} ms/update; pairs={r['selector_counts']}")
    (out/'REPORT_ZH.txt').write_text('\n'.join(text)+'\n')
    return out


def report_b2():
    root=Path('experiments/b2_vpt_active_20261002')
    out=Path('results/active_vpt_b2_20261002');out.mkdir(parents=True,exist_ok=True)
    rows=[]; missing=[];base={};verification=[]
    for domain in DOMAINS:
        b1=Path(f'experiments/b1_{domain}_20261002_v2')
        base[domain]=read(b1/'baseline.json')
        for method in METHODS:
            actual='facility' if domain=='ilids' and method=='facility_camera' else method
            path=root/domain/actual/'complete.json'
            if not path.exists():missing.append(f'{domain}/{method}');continue
            complete=read(path)
            for trial in complete['trials']:
                for metric in trial['result']['rows']:
                    rows.append(dict(domain=domain,method=method,draw=trial['draw'],seed=trial['seed'],
                        step=metric['step'],mAP=metric['mAP'],rank1=metric['rank1'],rank5=metric['rank5'],
                        gain_base=metric['mAP']-base[domain]['mAP'],effective_identities=trial['effective_identities'],
                        n_fail=trial['annotation']['n_fail'],n_dup=trial['annotation']['n_dup'],
                        anchor_set_hash=trial['anchor_set_hash'],support_hash=trial['support_hash'],
                        reused_equivalent=actual!=method,status=trial['result']['status'],
                        actual_updates=trial['result']['actual_updates']))
        verify=root/domain/'fresh_replay.json'
        verification.append(dict(domain=domain,passed=verify.exists() and read(verify)['baseline_exact']))
    random={(r['domain'],r['step'],r['draw'],r['seed']):r['mAP'] for r in rows if r['method']=='random'}
    summary=[]
    for domain in DOMAINS:
        for method in METHODS:
            items=[r for r in rows if r['domain']==domain and r['method']==method and r['step']==5000]
            if not items:continue
            values=np.array([r['mAP'] for r in items]);deltas=[r['mAP']-random[(domain,5000,r['draw'],r['seed'])] for r in items if (domain,5000,r['draw'],r['seed']) in random]
            summary.append(dict(domain=domain,method=method,n=len(items),baseline=base[domain]['mAP'],
                mean_mAP=float(values.mean()),std_mAP=float(values.std(ddof=1)) if len(values)>1 else 0.,
                min_mAP=float(values.min()),gain_base=float(values.mean()-base[domain]['mAP']),
                gain_random=float(np.mean(deltas)) if deltas else None,wins_random=sum(d>0 for d in deltas),
                mean_pairs=float(np.mean([r['effective_identities'] for r in items])),
                unique_anchor_sets=len({r['anchor_set_hash'] for r in items}),
                reused_equivalent=any(r['reused_equivalent'] for r in items)))
    if rows:
        with (out/'all_results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'summary.json',summary);save(out/'all_results.json',rows)
    complete=not missing and all(v['passed'] for v in verification) and len([r for r in rows if r['step']==5000])==168
    save(out/'audit.json',dict(complete=complete,missing=missing,verification=verification,
        nominal_conditions=168,independent_training_trials=162,primary_step=5000,
        primary_comparison='facility vs random',source_protocol='Market+MSMT train; development experiment, not formal P3',
        scope='No B3/B4; all methods reported; iLIDS facility_camera reuses identical facility selections.'))
    lines=['VPT主动选样B2结果（主终点5000步）',f'完整完成：{complete}',
        '主比较：facility相对random；所有方法同一VPT起点、预算、优化设置。',
        '指标单位为mAP百分点；标准差描述这些运行的变化，不是独立数据集置信区间。',
        '旧split0反复用于开发，结果不能直接作为正式P3泛化主表。',
        '域 / 方法 / mAP均值 / 标准差 / 相对random / 相对原VPT / 有效配对均值 / 独特选样集合数']
    for r in summary:
        gain='NA' if r['gain_random'] is None else f"{r['gain_random']:+.2f}"
        lines.append(f"{r['domain']} / {r['method']} / {r['mean_mAP']:.2f} / {r['std_mAP']:.2f} / {gain} / {r['gain_base']:+.2f} / {r['mean_pairs']:.2f} / {r['unique_anchor_sets']}"+(' [复用等价结果]' if r['reused_equivalent'] else ''))
    lines+=['','iLIDS没有真实摄像头，facility_camera与facility等价，未重复训练，也不算独立证据。',
        '按主终点报告，不用100/300/1000/3000步最好分替代5000步。',
        '若优于random但低于原VPT，只能称减少适应损害；选择收益应与使用标注本身的收益区分。',
        '不同选择集合、配对图片及有效身份数共同影响结果；不能把差异唯一归因于样本信息量。',
        '后续B3/B4未自动执行。']
    (out/'REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(14,9))
        for ax,domain in zip(axes.flat,DOMAINS):
            rr=[r for r in summary if r['domain']==domain]
            ax.bar(np.arange(len(rr)),[r['gain_random'] for r in rr],color=['#777777' if r['method']=='random' else '#337ab7' for r in rr])
            ax.axhline(0,color='black',linewidth=.8)
            ax.set_xticks(np.arange(len(rr)),[r['method'] for r in rr],rotation=35,ha='right')
            ax.set_title(domain);ax.set_ylabel('mAP points vs random (5000 updates)')
        fig.tight_layout();fig.savefig(out/'selector_gain.png',dpi=160);plt.close(fig)
    return out


def package(out,stage):
    dest=out/'code';dest.mkdir(exist_ok=True)
    files=['scripts/active_vpt_b1.py','scripts/active_vpt_b2.py','scripts/analyze_active_vpt.py',
           'adapters/active_vpt_selection.py','adapters/selectors.py','scripts/tune_vpt_fewshot.py',
           'scripts/oracle_prompt.py','adapters/warm_start_reid.py','adapters/baseline_model.py',
           'plans/active_vpt_b1.tasks','plans/active_vpt_b2.tasks','scripts/run_active_vpt_b2.py']
    hashes={}
    for name in files:
        p=Path(name)
        if not p.exists():continue
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(out/'package_manifest.json',dict(stage=stage,source_sha256=hashes,weights='server only',
        commit_status='uncommitted; no GitHub push'))
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(out))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['b1','b2'],required=True)
    a=parser.parse_args();out=report_b1() if a.stage=='b1' else report_b2();package(out,a.stage)
    print('REPORT',out,flush=True)
