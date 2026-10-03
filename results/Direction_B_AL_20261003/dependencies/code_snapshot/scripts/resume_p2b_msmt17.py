"""Accept the original MSMT17 split with a disclosed, user-approved overlap.

Does not alter any frozen experiment code or data. Existing three-fold GPU jobs
finish first; this controller then runs the missing target and the four-fold
report. Duplicate records remain in the pool/query/gallery unchanged.
"""
import sys, os, json, time, argparse, subprocess, hashlib, fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p2b_source import DOMAINS, TAG, sha
from scripts.active_vpt_p2 import load_data, save, read, digest
from adapters.active_vpt_selection import METHODS

ROOT=Path('experiments')/TAG
DATA=ROOT/'data/msmt17'
SOURCES=[
    dict(project='OSNet / deep-person-reid',url='https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/image/msmt17.py',
         observation='Reads listed train/query/gallery records; val is added only with combineall. No cross-split content deduplication in this loader.'),
    dict(project='TransReID',url='https://github.com/damo-cv/TransReID/blob/main/datasets/msmt17.py',
         observation='Reads listed records; appends val to train. No cross-split content deduplication in this loader.'),
    dict(project='CLIP-ReID',url='https://github.com/Syliz517/CLIP-ReID/blob/master/datasets/msmt17.py',
         observation='Reads listed records; appends val to train. No cross-split content deduplication in this loader.')]

def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def accept():
    ds,manifest=load_data('msmt17')
    assert read(DATA/'manifest.json')==json.loads(json.dumps(manifest))
    hashes=read(DATA/'image_sha256.json')
    report=read(DATA/'content_overlap_failure.json')
    assert report['cross_split_content_groups']==214
    sets={}
    for split in ('train','query','gallery'):
        rows=hashes[split]
        assert [r['path'] for r in rows]==[r[0] for r in getattr(ds,split)]
        for row in rows: assert sha(row['path'])==row['sha256'],('data changed',row['path'])
        sets[split]={r['sha256'] for r in rows}
    overlap=sets['train']&(sets['query']|sets['gallery'])
    assert overlap=={g['sha256'] for g in report['groups']}
    policy=dict(domain='msmt17',decision='retain_original_split_without_cross_split_deduplication',
        authorized_at=now(),authorized_by='User instruction in this chat on 2026-10-02',
        user_instruction='Search how papers using the dataset handle it; if they do not handle it, leave it unchanged. User supplied matching public-loader evidence.',
        evidence_scope='The inspected public loaders do not deduplicate. This does not establish private preprocessing or identical dataset copies in those papers.',
        sources=SOURCES,strict_content_isolation_passed=False,execution_authorized_with_disclosure=True,
        overlap_groups=214,overlap_records=dict(train=214,query=27,gallery=188),
        no_data_modified=True,no_candidate_or_test_filtering=True,
        train_images=30248,query_images=11659,gallery_images=82161,
        train_val_merge=False,source_all_images=False,
        note='Keep this project train-only recipe; public repos merging val does not change our frozen protocol.',
        disclosure='沿用 MSMT17 原有划分，未进行跨 train/test 图片去重；当前数据副本检测到 214 组完全相同的图片内容。',
        manifest_sha256=sha(DATA/'manifest.json'),image_hash_manifest_sha256=sha(DATA/'image_sha256.json'),
        overlap_evidence_sha256=sha(DATA/'content_overlap_failure.json'))
    policy_path=DATA/'accepted_original_split_policy.json'
    if policy_path.exists():
        old=read(policy_path)
        for k in ('decision','manifest_sha256','image_hash_manifest_sha256','overlap_evidence_sha256'):
            assert old[k]==policy[k]
        policy=old
    else: save(policy_path,policy)
    # This completion means the data audit finished with an accepted exception;
    # it explicitly does NOT assert clean content isolation.
    completion=dict(domain='msmt17',status='accepted_original_split_with_known_content_overlap',
        path_overlap=0,content_overlap_train_test=214,strict_content_isolation_passed=False,
        execution_authorized_with_disclosure=True,manifest_sha256=sha(DATA/'manifest.json'),
        query_gallery_shared_content=len(sets['query']&sets['gallery']),
        accepted_policy=str(policy_path),accepted_policy_sha256=sha(policy_path),
        original_failure_evidence_retained=True)
    if (DATA/'complete.json').exists(): assert read(DATA/'complete.json')==completion
    else:save(DATA/'complete.json',completion)
    save(ROOT/'target_scope.json',dict(requested_targets=DOMAINS,active_targets=DOMAINS,
        held_target=None,queued_target='msmt17',reason='User accepted original split; waiting for current GPU queue to finish',
        accepted_policy=str(policy_path)))
    print('MSMT17_ORIGINAL_SPLIT_ACCEPTED_WITH_DISCLOSURE',flush=True)

def make_plans():
    codehash=sha('scripts/active_vpt_p2.py')[:10]
    files=[]
    for stage in ('preflight','replay-preflight','adapt','verify'):
        lines=[]
        for method in METHODS if stage=='adapt' else [None]:
            suffix='_'+method if method else ''
            cmd=f'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" -u scripts/active_vpt_p2.py --stage {stage} --domain msmt17'
            if method:cmd+=' --method '+method
            lines.append(f'{TAG}_{codehash}_accepted_{stage}_msmt17{suffix} | {cmd}')
        p=Path('plans')/f'p2b_msmt17_accepted_{stage}.tasks'
        text='\n'.join(lines)+'\n'
        if p.exists():assert p.read_text()==text
        else:p.write_text(text)
        files.append((stage,p))
    return files

def disclosure_report():
    out=Path('results')/TAG
    policy=read(DATA/'accepted_original_split_policy.json')
    audit=read(out/'audit.json')
    assert audit['completed_conditions']==168 and audit['complete']
    audit.update(dataset_overlap_policy=policy,strict_train_test_content_isolation_all_domains=False,
        experiment_completed_under_disclosed_original_splits=True)
    save(out/'audit.json',audit)
    summary=read(out/'summary.json');summary['dataset_overlap_policy']=policy;save(out/'summary.json',summary)
    save(out/'MSMT17_original_split_policy.json',policy)
    report=out/'REPORT.txt';text=report.read_text()
    notice=('DATASET POLICY: MSMT17 ORIGINAL SPLIT RETAINED\n'+policy['disclosure']+'\n'
        '214 train records, 27 query records and 188 gallery records participate in these 214 groups.\n'
        'This known overlap was explicitly accepted by the user; strict content isolation did not pass.\n'
        'All images and original train/query/gallery lists were retained. No cross-split deduplication or PID filtering.\n'
        'Public-loader inspection does not prove identical data copies or absence of private preprocessing in cited papers.\n\n')
    if not text.startswith('DATASET POLICY:'):report.write_text(notice+text)

def followup():
    lock=(ROOT/'msmt17_followup.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(DATA/'complete.json')['execution_authorized_with_disclosure']
    plans=make_plans()
    save(ROOT/'full_protocol_status.json',dict(stage='queued_msmt17_after_current_three_fold_jobs',
        timestamp=now(),requested_targets=DOMAINS,held_target=None,complete=False,
        remaining_msmt17_conditions=42,accepted_policy=str(DATA/'accepted_original_split_policy.json')))
    print('MSMT17_QUEUED_AFTER_EXISTING_GPU_JOBS',flush=True)
    # Do not share/oversubscribe a card or terminate the four active experiments.
    # Old controller owns all four authorized GPUs until its verification/report ends.
    previous_lock=(ROOT/'controller.lock').open('r')
    while True:
        try:
            fcntl.flock(previous_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:time.sleep(30)
    if (ROOT/'controller_failure.json').exists():raise RuntimeError('Existing controller failed; inspect its evidence before continuing')
    partial=read(Path('results')/TAG/'partial_3_targets/audit.json')
    assert partial['available_fold_audit_passed'] and partial['completed_conditions']==126
    assert set(partial['included_domains'])==set(DOMAINS)-{'msmt17'}
    for stage,path in plans:
        save(ROOT/'full_protocol_status.json',dict(stage='msmt17_'+stage,timestamp=now(),complete=False))
        subprocess.run([sys.executable,'-u','scripts/launch_tasks.py',str(path),'--gpus','0,1,2,7'],check=True)
        if stage=='replay-preflight':
            timing=read(ROOT/'preflight/msmt17/complete.json')
            save(ROOT/'msmt17_measured_cost.json',dict(timestamp=now(),
                estimated_seconds_per_trial=timing['estimated_seconds_per_formal_trial'],
                ideal_four_gpu_hours=42*timing['estimated_seconds_per_formal_trial']/4/3600))
    subprocess.run([sys.executable,'scripts/analyze_active_vpt_p2.py'],check=True)
    disclosure_report()
    save(ROOT/'full_protocol_status.json',dict(stage='complete',timestamp=now(),complete=True,
        completed_conditions=168,disclosed_msmt17_overlap_groups=214,report=f'results/{TAG}/REPORT.txt'))
    print('P2_FOUR_FOLDS_COMPLETE_WITH_DISCLOSED_MSMT17_ORIGINAL_SPLIT',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-data',action='store_true')
    parser.add_argument('--queue',action='store_true')
    parser.add_argument('--check-plans',action='store_true')
    a=parser.parse_args()
    if a.accept_data:accept()
    if a.check_plans:
        for _,p in make_plans():
            subprocess.run([sys.executable,'scripts/launch_tasks.py',str(p),'--gpus','0,1,2,7','--dry-run'],check=True)
    if a.queue:followup()

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'msmt17_followup_failure.json',dict(timestamp=now(),error=repr(e)))
        raise

