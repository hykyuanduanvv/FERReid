"""Durable B2 queue -> independent reload checks -> report, then stop. No B3/B4."""
import os,sys,json,subprocess,hashlib,shutil,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT=ROOT/'experiments/b2_vpt_active_20261002'
PY='/SSD_Data01/miniforge/envs/ferreid/bin/python'


def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def status(state,**kwargs):
    target=OUT/'driver_status.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=state,time=stamp(),pid=os.getpid(),**kwargs),indent=2))
    tmp.replace(target)


def run(args,logname):
    with (OUT/logname).open('w') as log:
        return subprocess.run([PY,*args],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')).returncode


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT))
    from scripts.active_vpt_b2 import check_b1
    for domain in ('grid','viper','ilids','cuhk03'):check_b1(domain)
    assert json.loads((ROOT/'experiments/b1_cpu_20261002_v2/unit_checks.json').read_text())['anonymous_interface']
    audit=json.loads((ROOT/'results/active_vpt_b1_20261002/summary.json').read_text())
    assert audit['all_b1_checks_pass']
    assert shutil.disk_usage(ROOT).free>10*1024**3,'need 10 GiB free for safe experiment outputs'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(row.split(',')[0]):int(row.split(',')[1]) for row in gpu.strip().splitlines()}
    assert all(memory[i]<1000 for i in [0,1,2,7]),('authorized GPU busy',memory)
    files=['adapters/active_vpt_selection.py','scripts/active_vpt_b1.py','scripts/active_vpt_b2.py',
           'scripts/analyze_active_vpt.py','scripts/run_active_vpt_b2.py','plans/active_vpt_b2.tasks',
           'plans/active_vpt_b2_verify.tasks']
    code=OUT/'code';code.mkdir()
    for name in files:
        dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    (OUT/'launch_manifest.json').write_text(json.dumps(dict(
        started_at=stamp(),gpus=[0,1,2,7],steps=5000,nominal_conditions=168,independent_trials=162,
        aliases={'ilids/facility_camera':'ilids/facility'},
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        authorization='User explicitly requested B2 after B1; no commit/push, no B3/B4.'),indent=2))
    status('training',expected_independent_trials=162)
    code=run(['scripts/launch_tasks.py','plans/active_vpt_b2.tasks','--gpus','0,1,2,7'],'queue.log')
    verify=None
    if code==0:
        status('verifying')
        verify=run(['scripts/launch_tasks.py','plans/active_vpt_b2_verify.tasks','--gpus','0,1,2,7'],'verify.log')
    status('reporting',training_exit=code,verify_exit=verify)
    analysis=run(['scripts/analyze_active_vpt.py','--stage','b2'],'analysis.log')
    report=ROOT/'results/active_vpt_b2_20261002/audit.json'
    passed=code==0 and verify==0 and analysis==0 and report.exists() and json.loads(report.read_text())['complete']
    status('complete' if passed else 'failed',training_exit=code,verify_exit=verify,
           analysis_exit=analysis,report=str(report.parent),further_stages_launched=False)


if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():status('failed',error=traceback.format_exc(),further_stages_launched=False)
        raise
