"""Compact read-only progress for this P2 run (timestamps in Asia/Shanghai)."""
import sys,json,ast,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.p2b_source import DOMAINS,TAG

root=Path('experiments')/TAG
status={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'run':str(root),'sources':{},'adapt':{}}
for d in DOMAINS:
    out=root/('source_'+d)
    done=out/'complete.json'
    if done.exists():status['sources'][d]=json.loads(done.read_text());continue
    log=Path('experiments/_launch')/f'{TAG}_source_{d}.log'
    lines=log.read_text(errors='replace').splitlines() if log.exists() else []
    rows=[]
    for line in lines:
        if line.startswith("{'loss':"):
            try:rows.append(ast.literal_eval(line))
            except (ValueError,SyntaxError):pass
    row=rows[-1] if rows else {}
    step=round(row.get('epoch',0)*100000)
    age=time.time()-((out/'provenance.json').stat().st_mtime if (out/'provenance.json').exists() else time.time())
    status['sources'][d]=dict(approx_step=step,total=12000,loss=row.get('loss'),
        elapsed_hours=age/3600,eta_hours=(age/step*(12000-step)/3600 if step else None))
    checkpoints=sorted(out.glob('checkpoint-*/trainer_state.json'))
    if checkpoints:status['sources'][d]['saved_checkpoints']=[p.parent.name for p in checkpoints]
for name in ('controller_status.json','controller_failure.json','measured_cost_estimate.json'):
    if (root/name).exists():status[name]=json.loads((root/name).read_text())
for d in DOMAINS:
    files=list((root/'adapt'/d).glob('*/draw*/attempt_*/complete.json'))
    status['adapt'][d]=dict(completed_trials=len(files),planned=42)
    recent=[]
    for p in (root/'adapt'/d).glob('*/draw*/attempt_*/training.jsonl'):
        if (p.parent/'complete.json').exists():continue
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-2048));lines=f.read().splitlines()
        try:last=json.loads(lines[-1]);recent.append(dict(trial=str(p.parent),actual_updates=last['step'],attempt=last['attempt']))
        except (ValueError,IndexError):pass
    status['adapt'][d]['incomplete_attempts']=recent
print(json.dumps(status,indent=2,ensure_ascii=False))

