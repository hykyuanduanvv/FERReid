"""Isolated a0 numerical probes; does not change production model defaults.

Use plans/a0_diagnostics.tasks. Parameter-gradient norms may still be AMP-scaled
when max_grad_norm=0; use zero/nonzero diagnostics, not cross-run norm magnitudes.
Nonfinite diagnostic scalars are written as null in the JSON trace.
"""
import json, math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ops import losses
from models import Model
from adapters.trainer_reid import DGReIDTrainer
from transformers import TrainerCallback

if '--help' in sys.argv or '-h' in sys.argv:
    from scripts.train_reid import main
    main()
    raise SystemExit(0)
variant = os.environ['A0_PROBE_VARIANT']
if variant not in {'control','fp32_triplet','noicl','lr1e5','hidden_norm','random_init'}:
    raise ValueError('Unknown A0_PROBE_VARIANT: ' + variant)
out = Path(sys.argv[sys.argv.index('--output_dir')+1])
out.mkdir(parents=True, exist_ok=True)
trace = (out / 'numerics.jsonl').open('w', buffering=1)
counter = int(os.environ.get('A0_PROBE_STEP_OFFSET', '0'))
def emit(**kw):
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value
    trace.write(json.dumps(clean(dict(variant=variant, **kw)), allow_nan=False)+'\n')
emit(event='environment', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
original_loss = losses.HardTripletLoss.forward
def loss_forward(self, x, y):
    global counter
    if self.training:
        counter += 1
    if variant == 'fp32_triplet':
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = original_loss(self, x.float(), y)
    else:
        result = original_loss(self, x, y)
    if self.training and (counter <= 5 or counter % 10 == 0):
        with torch.no_grad():
            dlow = losses._pairwise_distance(x)
            with torch.autocast(device_type=x.device.type, enabled=False):
                dfull = losses._pairwise_distance(x.float())
            off = ~torch.eye(x.size(0), device=x.device, dtype=torch.bool)
            emit(event='features', step=counter, dtype=str(x.dtype),
                 std=x.float().std(dim=0).mean().item(), triplet=result.item(),
                 zero_distance_amp=(dlow[off]==0).float().mean().item(),
                 zero_distance_fp32=(dfull[off]==0).float().mean().item(),
                 distance_amp_mean=dlow[off].mean().item(), distance_fp32_mean=dfull[off].mean().item())
        step=counter
        if x.requires_grad:
            x.register_hook(lambda g: emit(event='feature_gradient', step=step,
                zero_fraction=(g==0).float().mean().item(), finite=bool(torch.isfinite(g).all())))
    return result
losses.HardTripletLoss.forward = loss_forward
original_prompts = Model._prompts_from_hidden
def prompts(self, h):
    if variant == 'hidden_norm':
        h = torch.nn.functional.layer_norm(h.float(), (h.size(-1),))
    p = original_prompts(self, h)
    next_step = counter + 1
    if self.training and (next_step <= 5 or next_step % 10 == 0):
        emit(event='prompts', step=next_step, hidden_rms=h.float().square().mean().sqrt().item(),
             prompt_rms=p.float().square().mean().sqrt().item(), prompt_max=p.float().abs().max().item())
    return p
Model._prompts_from_hidden = prompts

class GradProbe(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        s=state.global_step+1
        if s<=5 or s%10==0:
            norms={}
            for prefix in ['prompt_mlp','query_embeddings','mm_projector','encoder.']:
                terms=[p.grad.float().square().sum() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                norms[prefix]=torch.stack(terms).sum().sqrt().item() if terms else None
            emit(event='parameter_gradients', step=s, norms=norms)
original_init=DGReIDTrainer.__init__
def trainer_init(self,*a,**kw):
    original_init(self,*a,**kw)
    if variant == 'random_init':
        generator = torch.Generator(device=self.model.prompt_mlp.weight.device).manual_seed(4242)
        with torch.no_grad():
            self.model.prompt_mlp.weight.normal_(0, 0.0005, generator=generator)
    self.add_callback(GradProbe())
DGReIDTrainer.__init__=trainer_init
from scripts.train_reid import main
main()
trace.close()
