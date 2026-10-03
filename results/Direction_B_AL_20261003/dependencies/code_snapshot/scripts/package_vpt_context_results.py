import json,hashlib,zipfile
from pathlib import Path
root=Path('/root/hyk/FERReid'); result=root/'results/vpt_context_warm_20261001'
files=['adapters/warm_start_reid.py','scripts/train_vpt_context.py',
       'scripts/verify_vpt_context_checkpoint.py','scripts/analyze_vpt_context.py',
       'plans/vpt_context_warm.tasks','plans/vpt_context_reload.tasks',
       'experiments/step4_training_runner_v1.py','experiments/step4_model_training_v1.py']
provenance={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files}
provenance['notes']=['Training completed: 4 x 300 actual updates; initial combined exits 1 due to evaluation gate.',
    'Final checkpoint evaluation tasks all exit 0; training was not restarted.',
    'Initial combined logs and first reload failure logs are retained; no result filtering.',
    'Weights and prompt tensors stay on the server.']
(result/'provenance.json').write_text(json.dumps(provenance,indent=2))
archive=root/'experiments/step4_vpt_context_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(result.rglob('*')):
        if p.is_file(): z.write(p,'results/'+str(p.relative_to(result)))
    for p in files: z.write(root/p,p)
    for p in sorted((root/'experiments/_launch').glob('step4*')):
        if p.is_file() and p.suffix in ('.log','.status'):
            z.write(p,'logs/'+p.name)
    for p in sorted((root/'experiments/_launch/step4_reload_first_attempt').glob('*.log')):
        z.write(p,'logs/reload_first_attempt/'+p.name)
print(json.dumps(dict(size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                     summary=json.loads((result/'summary.json').read_text()))))
