"""Approved short Direction-B run. All artifacts isolated from the cancelled run."""
import os, sys, json, time, signal, argparse, hashlib, subprocess, fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch

REPO = Path("/root/hyk/FERReid")
sys.path.insert(0, str(REPO))
import scripts.active_vpt_p2 as old
import scripts.active_vpt_b1 as legacy
from adapters.active_vpt_selection import METHODS, annotate_selected
from adapters.config_reid import NO_CAMERA_DOMAINS

RUN = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve()
DOMAINS = ["market1501", "cuhksysu", "cuhk03", "msmt17"]
GPUS = [0, 1, 2, 7]
CONFIG = dict(version=1, steps=100, eval_steps=[30, 100], save_steps=[0, 10, 30, 50, 100],
              lr=1e-5, optimizer="Adam", betas=[.9, .999], eps=1e-8, weight_decay=0.,
              radius_ratio=.10, margin=.1, k=16, draws=[0, 1, 2], seeds=[42, 43],
              methods=list(METHODS), domains=DOMAINS, gpus=GPUS, planned_trials=168,
              loss="FP32 hardest Triplet only", trainable="external prompt only",
              augmentation="unchanged legacy flip/pad-crop/erasing; normalized zero padding",
              selection="reuse frozen label-free selections and deterministic annotations",
              main_step=100, secondary_step=30, model_mode="eval; autograd enabled for external prompt",
              symmetry="retain existing token symmetry and all 32 tokens",
              interpretation="revised exploratory protocol; no per-trial test-selected checkpoint",
              old_run="experiments/p2b_20261002_v1", no_old_jobs_resumed=True)

def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

def read(p):
    return json.loads(Path(p).read_text())

def save(p, value):
    old.save(p, value)

def sha(p):
    return old.sha(p)

def stopped():
    if (RUN / "STOP").exists():
        raise RuntimeError("Run cancelled: STOP marker present")

@torch.no_grad()
def project(p, initial, ratio):
    delta = p - initial
    radius = initial.norm() * ratio
    assert float(radius) > 0
    norm = delta.norm()
    hit = bool(norm > radius)
    if hit:
        p.copy_(initial + delta * (radius / norm))
    relative = float((p - initial).norm() / initial.norm())
    assert relative <= ratio + 2e-6, (relative, ratio)
    return hit, relative

def cpu_checks():
    initial = torch.tensor([1., 2., -3.])
    p = torch.nn.Parameter(initial + torch.tensor([10., -20., 15.]))
    before = p.detach().clone() - initial
    hit, rel = project(p, initial, .1)
    assert hit and abs(rel - .1) < 1e-6
    assert torch.allclose(torch.nn.functional.normalize(p-initial, dim=0),
                          torch.nn.functional.normalize(before, dim=0), atol=1e-6)
    same = p.detach().clone()
    _, rel2 = project(p, initial, .1)
    assert torch.allclose(p, same, atol=1e-7)
    p2 = torch.nn.Parameter(initial.clone())
    opt = torch.optim.Adam([p2], lr=1e-2)
    hits = 0
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        (-(p2*torch.tensor([1., 2., 3.])).sum()).backward()
        opt.step()
        h, r = project(p2, initial, .1)
        hits += int(h)
        assert r <= .100002
    assert hits > 0 and int(opt.state[p2]["step"]) == 100
    return dict(projection_direction=True, bounded_adam_steps=100, boundary_hits=hits,
                idempotence=True, final_relative_delta=r)

def prepare():
    os.chdir(REPO)
    assert not (RUN/"manifest.json").exists(), "Manifest already exists; do not overwrite frozen setup"
    result = cpu_checks()
    folds = {}
    for domain in DOMAINS:
        pre = old.ROOT/"preflight"/domain
        source = read(old.ROOT/("source_"+domain)/"complete.json")
        conf = read(pre/"configuration.json")
        assert read(pre/"fresh_replay.json")["baseline_pass"]
        sels = read(pre/"selections.json")
        assert len(sels)==21
        folds[domain] = dict(checkpoint=source["checkpoint"], checkpoint_sha256=source["checkpoint_sha256"],
            frozen_hash=conf["frozen_hash"], initial_prompt_hash=conf["initial_prompt_hash"],
            selection_sha256=sha(pre/"selections.json"), baseline_sha256=sha(pre/"baseline.json"),
            data_manifest_sha256=sha(old.ROOT/"data"/domain/"manifest.json"),
            baseline=read(pre/"baseline.json"))
    policy=read(old.ROOT/"data/msmt17/accepted_original_split_policy.json")
    manifest=dict(created_at=now(), configuration=CONFIG, runner_sha256=sha(SCRIPT), folds=folds,
        cpu_checks=result, git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        msmt17_original_split_policy=policy, strict_all_fold_content_isolation=False,
        approval="User authorized GPUs 0,1,2,7 and the reviewed 168-trial short protocol on 2026-10-03")
    save(RUN/"manifest.json",manifest)
    save(RUN/"status.json",dict(stage="prepared",time=now(),complete=False,planned_trials=168))
    print("PREPARED",json.dumps(result),flush=True)

def binding(domain):
    manifest=read(RUN/"manifest.json")
    assert manifest["configuration"]==CONFIG
    assert sha(SCRIPT)==manifest["runner_sha256"], "New runner changed after prepare"
    model,args,ds,_,source=old.build(domain)
    pre,conf=old.validate_binding(domain,model)
    fold=manifest["folds"][domain]
    assert source["checkpoint_sha256"]==fold["checkpoint_sha256"]
    assert sha(pre/"selections.json")==fold["selection_sha256"]
    assert sha(pre/"baseline.json")==fold["baseline_sha256"]
    assert sha(old.ROOT/"data"/domain/"manifest.json")==fold["data_manifest_sha256"]
    assert conf["frozen_hash"]==fold["frozen_hash"]
    return model,ds,pre,conf,fold

def get_support(ds, domain, selection):
    reconstructed=annotate_selected(ds.train,selection["anchors"],has_cameras=domain not in NO_CAMERA_DOMAINS,
        domain=domain,split=0,annotation_seed=selection["annotation_seed"])
    assert legacy.digest_json(reconstructed["pairs"])==selection["ordered_support_hash"]
    assert reconstructed["revealed_ids"]==selection["revealed_ids"]
    if not selection["pairs"]:
        return torch.empty(0, device="cuda")
    return legacy.load_images([p for pair in selection["pairs"] for p in pair],"cuda")

def run_trial(model, ds, support, initial, selection, seed, out, baseline, frozen, steps=100, eval_steps=(30,100)):
    stopped()
    out.mkdir(parents=True,exist_ok=False)
    config=dict(configuration=CONFIG, steps=steps, eval_steps=list(eval_steps), seed=seed, selection=selection,
                initial_prompt_hash=legacy.tensor_hash(initial), frozen_hash=frozen, runner_sha256=sha(SCRIPT))
    save(out/"configuration.json",config)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    gen=torch.Generator(device="cuda").manual_seed(seed)
    rng=np.random.RandomState(seed)
    p0=initial.detach().float().clone()
    p=torch.nn.Parameter(p0.clone())
    torch.save(p.detach().cpu(),out/"prompt-0.pt")
    opt=torch.optim.Adam([p],lr=CONFIG["lr"],betas=tuple(CONFIG["betas"]),eps=CONFIG["eps"],weight_decay=0.)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    criterion=legacy.HardTripletLoss(margin=CONFIG["margin"],hardest=True)
    labels=torch.arange(len(support)//2,device="cuda").repeat_interleave(2)
    rows=[]; nonzero=0; hits=0; train_seconds=0.;eval_seconds=0.;checkpoint_hashes={}
    if len(support)//2<2:
        for step in sorted(set(CONFIG["save_steps"])|{steps}):
            if step<=steps:
                torch.save(p.detach().cpu(),out/f"prompt-{step}.pt")
        rows=[dict(step=s,**baseline,gain=0.,used_source_prompt=True) for s in eval_steps]
        result=dict(status="insufficient_support_kept_baseline",actual_updates=0,rows=rows,frozen_hash=frozen)
        save(out/"complete.json",result)
        return result
    with torch.no_grad():
        before=float(legacy.fp32_loss(model,support,labels,p,criterion))
    start=time.perf_counter()
    with (out/"training.jsonl").open("w",buffering=1) as log:
        for step in range(1,steps+1):
            stopped()
            torch.cuda.synchronize(); tick=time.perf_counter()
            idx=[]
            for pid in rng.permutation(len(support)//2):
                idx.extend((2*pid+rng.permutation(2)).tolist())
            ii=torch.tensor(idx,device="cuda")
            images=legacy.augment(support[ii],gen); y=labels[ii]
            loss=legacy.fp32_loss(model,images,y,p,criterion)
            assert bool(torch.isfinite(loss)), "Nonfinite loss"
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), "Missing/nonfinite prompt gradient"
            gn=float(p.grad.float().norm());nonzero+=int(gn>0)
            scale=scaler.get_scale()
            opt_before=p.detach().clone()
            scaler.step(opt);scaler.update()
            assert scaler.get_scale()>=scale, "AMP skipped step; fail without extending training"
            hit,relative=project(p,p0,CONFIG["radius_ratio"]);hits+=int(hit)
            record=dict(step=step,loss=float(loss),gradient_norm=gn,relative_prompt_delta=relative,
                delta_rms=float((p.detach()-p0).square().mean().sqrt()),projection_hit=hit,
                parameter_update_norm=float((p.detach()-opt_before).norm()),scale=float(scaler.get_scale()),
                images_sha256=legacy.tensor_hash(images),labels_sha256=legacy.tensor_hash(y))
            log.write(json.dumps(record,allow_nan=False)+"\n")
            torch.cuda.synchronize(); train_seconds+=time.perf_counter()-tick
            if step in set(CONFIG["save_steps"])|{steps}:
                torch.save(p.detach().cpu(),out/f"prompt-{step}.pt")
                checkpoint_hashes[str(step)]=legacy.tensor_hash(p.detach())
            if step in eval_steps:
                tick=time.perf_counter()
                score=old.evaluate(model,ds,p.detach())
                rows.append(dict(step=step,**score,gain=score["mAP"]-baseline["mAP"]))
                save(out/"retrieval.partial.json",rows)
                torch.cuda.synchronize();eval_seconds+=time.perf_counter()-tick
                print("SHORT_EVAL",str(out),json.dumps(rows[-1]),flush=True)
    assert int(opt.state[p]["step"])==steps
    assert legacy.state_hash(model.state_dict().items())==frozen
    assert all(v.grad is None for v in model.parameters())
    restored=torch.load(out/f"prompt-{steps}.pt",map_location="cuda",weights_only=True)
    assert torch.equal(restored,p.detach())
    with torch.no_grad():
        after=float(legacy.fp32_loss(model,support,labels,p,criterion))
        f1=model(support,prompts=p.detach())["features"]
        f2=model(support,prompts=restored)["features"]
        assert torch.equal(f1,f2), "Saved prompt feature replay differs"
    result=dict(status="complete",rows=rows,actual_updates=steps,nonzero_gradient_steps=nonzero,
        support_loss_before=before,support_loss_after=after,projection_hits=hits,
        relative_prompt_delta=float((p.detach()-p0).norm()/p0.norm()),frozen_hash=frozen,
        prompt_hash=legacy.tensor_hash(p.detach()),checkpoint_hashes=checkpoint_hashes,
        saved_prompt_exact=True,feature_replay_exact=True,train_seconds=train_seconds,
        evaluation_seconds=eval_seconds,elapsed_seconds=time.perf_counter()-start)
    save(out/"complete.json",result)
    return result

def preflight(domain):
    if (RUN/"preflight"/domain/"complete.json").exists():return
    model,ds,pre,conf,fold=binding(domain)
    initial=model.prompt.detach().clone()
    assert legacy.tensor_hash(initial)==fold["initial_prompt_hash"]
    baseline=old.evaluate(model,ds,initial)
    old.close(baseline,fold["baseline"])
    selection=next(s for s in read(pre/"selections.json") if s["method"]=="random" and s["draw"]==0)
    support=get_support(ds,domain,selection)
    out=RUN/"preflight"/domain
    result=run_trial(model,ds,support,initial,selection,42,out/"smoke",baseline,conf["frozen_hash"],5,())
    assert result["status"]=="complete"
    save(out/"complete.json",dict(domain=domain,time=now(),baseline=baseline,baseline_pass=True,
        bounded_optimizer_pass=True,saved_feature_replay_pass=True,smoke=result))
    print("SHORT_PREFLIGHT_PASS",domain,flush=True)

def adapt(domain,method):
    assert read(RUN/"preflight"/domain/"complete.json")["baseline_pass"]
    model,ds,pre,conf,fold=binding(domain)
    initial=model.prompt.detach().clone()
    selections=[r for r in read(pre/"selections.json") if r["method"]==method]
    assert len(selections)==3
    records=[]
    for sel in selections:
        support=get_support(ds,domain,sel)
        for seed in CONFIG["seeds"]:
            stopped()
            out=RUN/"adapt"/domain/method/f"draw{sel['draw']}_seed{seed}"
            if (out/"complete.json").exists():
                c=read(out/"configuration.json")
                assert c["configuration"]==CONFIG and c["runner_sha256"]==sha(SCRIPT)
                assert c["selection"]["ordered_support_hash"]==sel["ordered_support_hash"]
                result=read(out/"complete.json")
                assert result["frozen_hash"]==conf["frozen_hash"]
            else:
                result=run_trial(model,ds,support,initial,sel,seed,out,fold["baseline"],conf["frozen_hash"])
            records.append(dict(domain=domain,method=method,draw=sel["draw"],seed=seed,
                                effective_identities=sel["effective_identities"],result=result))
            print("SHORT_TRIAL_COMPLETE",domain,method,sel["draw"],seed,flush=True)
        del support
    save(RUN/"adapt"/domain/method/"complete.json",dict(trials=records,time=now()))

def verify(domain):
    model,ds,pre,conf,fold=binding(domain)
    baseline=old.evaluate(model,ds,model.prompt)
    old.close(baseline,fold["baseline"])
    checked=[]
    for method in ["random","facility"]:
        directory=RUN/"adapt"/domain/method/"draw0_seed42"
        result=read(directory/"complete.json")
        p=torch.load(directory/"prompt-100.pt",map_location="cuda",weights_only=True)
        assert legacy.tensor_hash(p)==result["prompt_hash"]
        actual=old.evaluate(model,ds,p)
        expected=next(r for r in result["rows"] if r["step"]==100)
        old.close(actual,expected)
        checked.append(dict(method=method,step=100,actual=actual,expected=expected,passed=True))
    save(RUN/"verify"/(domain+".json"),dict(domain=domain,baseline_pass=True,independent_process=True,trials=checked,time=now()))
    print("SHORT_VERIFY_PASS",domain,flush=True)

def counts():
    return {d:len(list((RUN/"adapt"/d).glob("*/draw*_seed*/complete.json"))) for d in DOMAINS}

def summarize():
    manifest=read(RUN/"manifest.json")
    report=dict(configuration=CONFIG,completed_at=now(),domains={},msmt17_original_split_policy=manifest["msmt17_original_split_policy"])
    text=["# Direction B：四折七方法短程微调结果", "", "固定 100 步主结果；30 步次要结果。所有数值为探索性修订实验。", ""]
    all_rows=[]
    for domain in DOMAINS:
        assert read(RUN/"verify"/(domain+".json"))["baseline_pass"]
        baseline=manifest["folds"][domain]["baseline"]
        methods={}
        for method in METHODS:
            trials=read(RUN/"adapt"/domain/method/"complete.json")["trials"]
            assert len(trials)==6
            metrics={}
            for step in [30,100]:
                bydraw=[];r1=[]
                for draw in [0,1,2]:
                    rows=[next(r for r in t["result"]["rows"] if r["step"]==step) for t in trials if t["draw"]==draw]
                    assert len(rows)==2
                    bydraw.append(float(np.mean([r["mAP"] for r in rows])))
                    r1.append(float(np.mean([r["rank1"] for r in rows])))
                metrics[str(step)]=dict(mAP_mean=float(np.mean(bydraw)),mAP_std_across_draws=float(np.std(bydraw,ddof=1)),
                    rank1_mean=float(np.mean(r1)),gain_vs_source=float(np.mean(bydraw))-baseline["mAP"],draw_means=bydraw)
            for t in trials:
                r=t["result"]
                assert r["actual_updates"]==100 or r["status"]=="insufficient_support_kept_baseline"
                if r["status"]=="complete":assert r["relative_prompt_delta"]<=.100002
            methods[method]=dict(metrics=metrics,trials=trials)
            all_rows.extend(trials)
        for method in METHODS:
            for step in ["30","100"]:
                methods[method]["metrics"][step]["gain_vs_random"]=methods[method]["metrics"][step]["mAP_mean"]-methods["random"]["metrics"][step]["mAP_mean"]
        report["domains"][domain]=dict(baseline=baseline,methods=methods)
        text += [f"## {domain}","",f"源模型 mAP：{baseline['mAP']:.3f}","", "| 方法 | 30 步 mAP | 100 步 mAP | 100 步对源模型增益 |", "|---|---:|---:|---:|"]
        for method in METHODS:
            m=methods[method]["metrics"]
            text.append(f"| {method} | {m['30']['mAP_mean']:.3f} | {m['100']['mAP_mean']:.3f} | {m['100']['gain_vs_source']:+.3f} |")
        text.append("")
    assert len(all_rows)==168
    report["four_fold_macro"]={m:{s:float(np.mean([report["domains"][d]["methods"][m]["metrics"][s]["mAP_mean"] for d in DOMAINS])) for s in ["30","100"]} for m in METHODS}
    text += ["MSMT17 沿用用户已接受的原划分，保留 214 组跨 train/test 完全重复图片。", "", "未根据目标测试成绩选择最佳检查点；旧实验未恢复。"]
    save(RUN/"summary.json",report)
    (RUN/"REPORT.md").write_text("\n".join(text)+"\n")

def controller():
    lock=(RUN/"controller.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert sha(SCRIPT)==read(RUN/"manifest.json")["runner_sha256"]
    running={}
    def terminate(sig=None,frame=None):
        for info in running.values():
            proc=info["proc"]
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        save(RUN/"status.json",dict(stage="stopped",time=now(),complete=False,completed=counts()))
        raise SystemExit(128+(sig or 15))
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def stage(name,tasks):
        pending=list(tasks); free=list(GPUS)
        while pending or running:
            if (RUN/"STOP").exists():terminate()
            while pending and free:
                task=pending.pop(0);gpu=free.pop(0)
                key="_".join(task)
                logpath=RUN/"logs"/(key+".log");logpath.parent.mkdir(exist_ok=True)
                handle=logpath.open("w")
                args=[sys.executable,"-u",str(SCRIPT),"--stage",task[0],"--domain",task[1]]
                if len(task)==3:args += ["--method",task[2]]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
                proc=subprocess.Popen(args,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                running[key]=dict(proc=proc,log=handle,gpu=gpu,task=task)
                print("START",now(),key,"GPU",gpu,"PID",proc.pid,flush=True)
            finished=[]
            for key,info in running.items():
                code=info["proc"].poll()
                if code is not None:
                    info["log"].close()
                    save(RUN/"task_status"/(key+".json"),dict(task=info["task"],gpu=info["gpu"],returncode=code,time=now()))
                    print("END",now(),key,code,flush=True)
                    if code!=0:
                        save(RUN/"failure.json",dict(task=key,returncode=code,time=now()))
                        terminate()
                    free.append(info["gpu"]);finished.append(key)
            for key in finished:del running[key]
            save(RUN/"status.json",dict(stage=name,time=now(),complete=False,completed=counts(),pending_tasks=len(pending),
                running=[dict(task=k,pid=v["proc"].pid,gpu=v["gpu"]) for k,v in running.items()]))
            time.sleep(2)
    stage("preflight",[("preflight",d) for d in DOMAINS])
    stage("adapt",[("adapt",d,m) for d in DOMAINS for m in METHODS])
    stage("verify",[("verify",d) for d in DOMAINS])
    summarize()
    save(RUN/"status.json",dict(stage="complete",time=now(),complete=True,completed=counts(),report=str(RUN/"REPORT.md")))
    print("SHORT_PROTOCOL_COMPLETE",now(),flush=True)

def main():
    os.chdir(REPO)
    torch.set_num_threads(2)
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",required=True,choices=["cpu-checks","prepare","controller","preflight","adapt","verify","status"])
    ap.add_argument("--domain",choices=DOMAINS)
    ap.add_argument("--method",choices=METHODS)
    a=ap.parse_args()
    if a.stage=="cpu-checks":print(json.dumps(cpu_checks()));return
    if a.stage=="prepare":prepare();return
    if a.stage=="controller":controller();return
    if a.stage=="status":print(json.dumps(dict(status=read(RUN/"status.json"),completed=counts()),indent=2));return
    assert a.domain
    if a.stage=="preflight":preflight(a.domain)
    elif a.stage=="verify":verify(a.domain)
    else:
        assert a.method;adapt(a.domain,a.method)

if __name__=="__main__":
    main()
