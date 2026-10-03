"""Two-seed 300-step LN versus LN+FP32 stability audit, AMP initial scale 128.

Experimental runtime interventions only. For evaluation of an emitted checkpoint,
the same intervention must be installed; training_args.bin alone does not encode it.
Batch/initialization/RNG hashes verify matching. Gradient probes differentiate the
unscaled Triplet explicitly rather than comparing AMP-scaled backward hooks.
"""
import sys, os, json, math, hashlib, random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrainerCallback
from models import Model
from ops import losses
from adapters.trainer_reid import DGReIDTrainer

VARIANTS={'ln':(True,False),'ln_fp32':(True,True)}
variant=os.environ['A0_STABILITY_VARIANT']
use_ln,use_fp32=VARIANTS[variant]
out=Path(sys.argv[sys.argv.index('--output_dir')+1]); out.mkdir(parents=True,exist_ok=True)
if (out/'numerics.jsonl').exists(): raise FileExistsError(out/'numerics.jsonl')
log=(out/'numerics.jsonl').open('w',buffering=1)
step=0
def clean(x):
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    return x
def emit(event,**kw):
    log.write(json.dumps(clean(dict(event=event,variant=variant,step=step,**kw)),allow_nan=False)+'\n')
def digest(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def observed(): return step<=10 or step%5==0
def tensor_stats(x):
    x=x.detach().float()
    return dict(rms=float(x.square().mean().sqrt()),max_abs=float(x.abs().max()),finite=bool(torch.isfinite(x).all()))

# A fixed precision policy shared by all four arms. No RNG calls in instrumentation.
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
emit('environment',torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
     ln=use_ln,fp32_triplet=use_fp32,tf32=False,initial_scale_requested=128,
     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

original_forward=Model.forward
def model_forward(self,image_crops,labels=None,prompts=None):
    global step
    if self.training:
        step+=1
        emit('batch',images_sha256=digest(image_crops),labels_sha256=digest(labels),
             cpu_rng_sha256=digest(torch.get_rng_state()),cuda_rng_sha256=digest(torch.cuda.get_rng_state()),
             unique_ids=int(labels.unique().numel()),ids=int(labels.numel()))
    output=original_forward(self,image_crops,labels=labels,prompts=prompts)
    if self.training and observed():
        emit('losses',total=float(output['loss'].detach()),icl=float(output['icl_loss'].detach()),
             triplet=float(output['id_loss'].detach()),wpa=float(output['ot_loss'].detach()))
    return output
Model.forward=model_forward

original_prompt=Model._prompts_from_hidden
def make_prompt(self,h):
    raw=h
    if use_ln: h=F.layer_norm(h.float(),(h.size(-1),))
    p=original_prompt(self,h)
    if self.training and observed():
        with torch.no_grad():
            energy=raw.float().square().sum(dim=(0,1)); top=energy.topk(4)
            emit('generator',raw_hidden=tensor_stats(raw),head_input=tensor_stats(h),prompt=tensor_stats(p),
                 head_weight=tensor_stats(self.prompt_mlp.weight),
                 hidden_top4_energy_fraction=float(top.values.sum()/energy.sum()),hidden_top4_dims=top.indices.tolist())
            if step in [1,2,10,20,100,200,300]:
                torch.save(dict(hidden=raw.detach().cpu(),prompt=p.detach().cpu()),out/f'generator_step{step:03d}.pt')
    return p
Model._prompts_from_hidden=make_prompt

original_loss=losses.HardTripletLoss.forward
def loss_forward(self,x,y):
    if use_fp32:
        with torch.autocast(device_type=x.device.type,enabled=False): result=original_loss(self,x.float(),y)
    else: result=original_loss(self,x,y)
    if self.training and observed():
        with torch.no_grad():
            amp=losses._pairwise_distance(x)
            with torch.autocast(device_type=x.device.type,enabled=False):
                xf=x.float(); full=losses._pairwise_distance(xf)
                direct=(xf[:,None,:]-xf[None,:,:]).square().sum(-1).sqrt()
                off=~torch.eye(len(x),dtype=torch.bool,device=x.device)
                pos=(y[:,None]==y[None,:])&off; neg=(y[:,None]!=y[None,:])
                def distances(d):
                    hp=d.masked_fill(~pos,0).max(1).values
                    hn=d.masked_fill(~neg,float('inf')).min(1).values
                    return dict(zero_fraction=float((d[off]==0).float().mean()),mean=float(d[off].mean()),
                        positive_mean=float(d[pos].mean()),negative_mean=float(d[neg].mean()),
                        hardest_positive=float(hp.mean()),hardest_negative=float(hn.mean()),hard_gap=float((hn-hp).mean()))
                emit('features',dtype=str(x.dtype),std=float(xf.std(0).mean()),feature_rms=float(xf.square().mean().sqrt()),
                     amp=distances(amp),fp32=distances(full),direct=distances(direct),
                     amp_vs_direct_rmse=float((amp-direct).square().mean().sqrt()),
                     fp32_vs_direct_rmse=float((full-direct).square().mean().sqrt()))
                if step in [1,2,10,20,100,200,300]: torch.save(dict(features=x.detach().cpu(),labels=y.cpu()),out/f'features_step{step:03d}.pt')
        # This derivative is unscaled; it is not an AMP-scaled parameter gradient.
        g=torch.autograd.grad(result,x,retain_graph=True)[0]
        emit('triplet_gradient',zero_fraction=float((g==0).float().mean()),finite=bool(torch.isfinite(g).all()),
             norm=float(g.float().norm()),gradient_dtype=str(g.dtype),unscaled=True)
        # Counterfactual FP32 derivative on EXACTLY the same feature tensor.
        with torch.enable_grad(),torch.autocast(device_type=x.device.type,enabled=False):
            xx=x.detach().float().requires_grad_(True)
            other=original_loss(self,xx,y)
            gg=torch.autograd.grad(other,xx)[0]
        emit('counterfactual_fp32_gradient',zero_fraction=float((gg==0).float().mean()),
             finite=bool(torch.isfinite(gg).all()),norm=float(gg.norm()),loss=float(other.detach()))
    return result
losses.HardTripletLoss.forward=loss_forward

class AuditCallback(TrainerCallback):
    def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
        if observed():
            for prefix in ['prompt_mlp','query_embeddings','mm_projector','encoder.']:
                grads=[p.grad.detach().float() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                if grads:
                    emit('parameter_gradient',group=prefix,zero_fraction=sum(int((g==0).sum()) for g in grads)/sum(g.numel() for g in grads),
                         finite=all(bool(torch.isfinite(g).all()) for g in grads),
                         note='total-loss gradient; may be AMP-scaled; magnitudes deliberately omitted')
    def on_step_end(self,args,state,control,**kw):
        # Every step, so successful updates and skipped updates can be counted exactly.
        scaler=trainer_ref[0].accelerator.scaler
        emit('optimizer',global_step=state.global_step,scale=float(scaler.get_scale()),
             skipped=bool(trainer_ref[0].accelerator.optimizer_step_was_skipped))
    def on_save(self,args,state,control,**kw):
        path=out/f'checkpoint-{state.global_step}'/'diagnostic_intervention.json'
        path.write_text(json.dumps(dict(variant=variant,head_input_layernorm=use_ln,triplet_fp32=use_fp32,
            initial_loss_scale=128,dynamic_scaling=True,seed=args.seed,global_step=state.global_step,
            runner='scripts/diagnose_a0_stability.py',
            warning='Evaluation must install the same runtime intervention; production defaults do not encode it.'),indent=2))

trainer_ref=[]
original_init=DGReIDTrainer.__init__
def trainer_init(self,*a,**kw):
    original_init(self,*a,**kw); trainer_ref.append(self)
    assert self.args.gradient_accumulation_steps==1
    assert self.args.model_type=='vicp' and self.model.prompt_mode=='vicp'
    assert self.args.max_steps==300 and self.args.seed in (42,43)
    assert not self.args.resume_from_checkpoint, 'Stability audit starts from matched fresh initialization'
    scaler=self.accelerator.scaler
    assert scaler is not None and scaler.is_enabled()
    amp_state=scaler.state_dict()
    assert amp_state['_growth_tracker']==0
    amp_state['scale']=128.0
    scaler.load_state_dict(amp_state)
    assert scaler.get_scale()==128.0
    emit('scaler_initialization',state=scaler.state_dict(),dynamic=True)
    hs=hashlib.sha256(); trainable=hashlib.sha256()
    for name,p in self.model.named_parameters():
        item=name.encode()+digest(p).encode()
        hs.update(item)
        if p.requires_grad: trainable.update(item)
    emit('initialization',all_parameters_sha256=hs.hexdigest(),trainable_parameters_sha256=trainable.hexdigest(),
         head_weight_max=float(self.model.prompt_mlp.weight.abs().max()),
         source_domains=self.source_domains,source_all_images=self.args.source_all_images,
         seed=self.args.seed,learning_rate=self.args.learning_rate,max_steps=self.args.max_steps,
         train_batch_size=self.args.per_device_train_batch_size,num_icl_samples=self.args.num_icl_samples,
         trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad))
    self.add_callback(AuditCallback())
DGReIDTrainer.__init__=trainer_init

from scripts.train_reid import main
main()
emit('complete'); log.close()
