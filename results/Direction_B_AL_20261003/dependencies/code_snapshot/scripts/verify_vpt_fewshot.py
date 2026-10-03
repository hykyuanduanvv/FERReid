"""Fresh-process replay of a saved primary prompt, one per target domain."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.tune_vpt_fewshot import load_data,evaluate,save
from adapters.baseline_model import VPTReIDModel
from adapters.warm_start_reid import load_exact,state_hash

def main():
    out=Path(sys.argv[1]);complete=json.loads((out/'complete.json').read_text())
    domain=complete['domain'];provenance=json.loads((out/'provenance.json').read_text())
    ckpt=Path(provenance['checkpoint'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args=torch.load(ckpt/'training_args.bin',map_location='cpu',weights_only=False)
    model=VPTReIDModel(args).to(device='cuda',dtype=torch.float16)
    state=torch.load(ckpt/'pytorch_model.bin',map_location='cpu',weights_only=True)
    load_exact(model,state,'cuda');del state
    model.requires_grad_(False).eval()
    assert state_hash(model.state_dict().items())==provenance['frozen_hash']
    ds,_,_=load_data(domain)
    trial=out/'draw0_seed42_lr0.0001'
    prompt=torch.load(trial/'prompt-300.pt',map_location='cuda',weights_only=True)
    actual=evaluate(model,ds,prompt)
    expected=json.loads((trial/'complete.json').read_text())['rows'][-1]
    assert all(actual[k]==expected[k] for k in ['mAP','rank1','rank5']), (actual,expected)
    base=evaluate(model,ds,model.prompt)
    assert base==complete['baseline'],(base,complete['baseline'])
    result=dict(domain=domain,primary_trial=trial.name,metrics=actual,baseline=base,
                exact_reproduction=True,frozen_hash=provenance['frozen_hash'])
    save(out/'fresh_replay.json',result);print(json.dumps(result))

if __name__=='__main__':main()
