import json,hashlib,zipfile,subprocess
from pathlib import Path
root=Path('/root/hyk/FERReid');result=root/'results/vpt_fewshot_20261001'
files=['scripts/tune_vpt_fewshot.py','scripts/verify_vpt_fewshot.py','scripts/analyze_vpt_fewshot.py',
       'scripts/oracle_prompt.py','adapters/warm_start_reid.py','scripts/package_vpt_fewshot_results.py',
       'plans/vpt_fewshot.tasks','plans/vpt_fewshot_verify.tasks','plans/vpt_fewshot_verify_ready.tasks']
manifest=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              files={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
              notes=['All 48 predeclared trials are included; primary LR=1e-4, sensitivity LR=1e-3.',
                     '3 annotation selections x 2 optimization seeds x 2 LRs x 4 domains, 300 steps each.',
                     'No test-based choice of selection, seed, checkpoint, or LR.',
                     'Model weights and saved prompt tensors remain on the server.'])
(result/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=root/'experiments/step5_vpt_fewshot_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(result.rglob('*')):
        if p.is_file():z.write(p,'results/'+str(p.relative_to(result)))
    for p in files:z.write(root/p,p)
    for p in sorted((root/'experiments/_launch').glob('step5*')):
        if p.is_file() and p.suffix in ('.log','.status'):z.write(p,'logs/'+p.name)
print(json.dumps(dict(size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
