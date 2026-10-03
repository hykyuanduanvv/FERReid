"""Dependency-gated P2 controller: audit -> source completion -> preflight -> reload -> adapt -> verify -> report."""
import sys,os,json,time,subprocess,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p2b_source import DOMAINS,TAG

ROOT=Path('experiments')/TAG
METHODS=['random','dedup','kcenter','typical','hard_negative','facility','facility_camera']

def save(name,obj):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)

def run(command):
    print('RUN',command,flush=True)
    subprocess.run(command,check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hold-target',choices=DOMAINS,help='Retain a failed target data gate while advancing independent folds')
    args=ap.parse_args()
    targets=[d for d in DOMAINS if d!=args.hold_target]
    import fcntl
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save('target_scope.json',dict(requested_targets=DOMAINS,active_targets=targets,held_target=args.hold_target,
        reason='Pending user decision on confirmed MSMT17 train/test image-content overlap' if args.hold_target else None))
    failure=ROOT/'controller_failure.json'
    if failure.exists():
        n=len(list(ROOT.glob('controller_failure_*.json')))+1
        failure.rename(ROOT/f'controller_failure_{n:03d}.json')
    save('controller_status.json',dict(stage='data_audit',time=time.time(),active_targets=targets,held_target=args.hold_target))
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    jobs=[]
    for d in targets:
        log=(ROOT/f'data_{d}.log').open('a')
        p=subprocess.Popen([sys.executable,'-u','scripts/active_vpt_p2.py','--stage','audit','--domain',d],
             stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs.append((d,p,log))
    for d,p,log in jobs:
        assert p.wait()==0,('data audit failed',d);log.close()
    save('controller_status.json',dict(stage='waiting_for_source',time=time.time()))
    print('DATA_AUDITS_PASSED',flush=True)
    while True:
        statuses=[Path('experiments/_launch')/f'{TAG}_source_{d}.status' for d in DOMAINS]
        for p in statuses:
            if p.exists(): assert p.read_text().strip()=='0',('source failed',str(p))
        if all(p.exists() for p in statuses):break
        time.sleep(30)
    for d in DOMAINS:
        assert json.loads((ROOT/('source_'+d)/'complete.json').read_text())['trainer_steps']==12000
    codehash=hashlib.sha256(Path('scripts/active_vpt_p2.py').read_bytes()).hexdigest()[:10]
    for stage,filename in [('preflight','p2b_preflight'),('replay-preflight','p2b_preflight_verify'),
                           ('adapt','p2b_adapt'),('verify','p2b_verify')]:
        lines=[]
        for d in targets:
            for method in (METHODS if stage=='adapt' else [None]):
                suffix=f'_{method}' if method else ''
                command=f'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" -u scripts/active_vpt_p2.py --stage {stage} --domain {d}'
                if method:command+=' --method '+method
                lines.append(f'{TAG}_{codehash}_{stage}_{d}{suffix} | {command}')
        path=Path('plans')/(filename+'.tasks');path.write_text('\n'.join(lines)+'\n')
        save('controller_status.json',dict(stage=stage,time=time.time(),codehash=codehash))
        run([sys.executable,'-u','scripts/launch_tasks.py',str(path),'--gpus','0,1,2,7'])
        if stage=='replay-preflight':
            timing={d:json.loads((ROOT/'preflight'/d/'complete.json').read_text())['estimated_seconds_per_formal_trial'] for d in targets}
            save('measured_cost_estimate.json',dict(seconds_per_trial=timing,
                ideal_four_gpu_adaptation_hours=sum(timing.values())*42/4/3600,
                note='Measured full evaluation and prompt updates; shared I/O contention adds uncertainty.'))
            print('MEASURED_COST',timing,flush=True)
    run([sys.executable,'scripts/analyze_active_vpt_p2.py','--domains',','.join(targets)])
    save('controller_status.json',dict(stage='awaiting_data_decision' if args.hold_target else 'complete',
        time=time.time(),completed_targets=targets,held_target=args.hold_target,
        requested_protocol_complete=not bool(args.hold_target)))
    print('P2_AVAILABLE_FOLDS_COMPLETE' if args.hold_target else 'P2_ALL_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save('controller_failure.json',dict(error=repr(e),time=time.time()))
        raise

