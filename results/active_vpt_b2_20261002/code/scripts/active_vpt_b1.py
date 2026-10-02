"""B1 integration and audit ONLY. Does not launch the B2 sweep.

Formal configuration is 5000 updates. B1 runs 20-update interface checks,
a matched 300-update legacy replay, and independent reload verification.
"""
import sys, json, time, hashlib, argparse, subprocess, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from adapters.active_vpt_selection import (METHODS, AnonymousPool, choose,
    annotate_selected, selection_metrics, unit_checks)
from scripts.tune_vpt_fewshot import (load_data, extract, evaluate, save, load_images,
    fp32_loss, augment, HardTripletLoss, VPTReIDModel, load_exact, state_hash, tensor_hash)
from adapters.config_reid import NO_CAMERA_DOMAINS

FORMAL = dict(steps=5000, eval_steps=[100, 300, 1000, 3000, 5000], lr=1e-4,
              k=16, selection_seeds=[1600,1601,1602], annotation_seeds=[2600,2601,2602],
              optimization_seeds=[42,43], methods=list(METHODS), split=0,
              feature_source='frozen VPT prompt; flip-summed retrieval features re-normalized in FP32 for selector cosine',
              adaptation='external prompt only; FP32 hardest triplet; Adam',
              status='B2 not launched; B1 only')


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build_model(checkpoint):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args = torch.load(checkpoint/'training_args.bin', map_location='cpu', weights_only=False)
    assert args.model_type == 'vpt' and not args.source_all_images
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = VPTReIDModel(args).to(device='cuda', dtype=torch.float16)
    state = torch.load(checkpoint/'pytorch_model.bin', map_location='cpu', weights_only=True)
    load_exact(model, state, 'cuda'); del state
    model.requires_grad_(False).eval()
    return model, args


def fit_prompt(model, ds, support, init, selection, out, base, frozen_hash,
               steps=5000, eval_steps=(100,300,1000,3000,5000), seed=42, lr=1e-4):
    """Exactly `steps` actual updates; evaluation/save schedule is parameterized.

    Each optimizer step retains input/label fingerprints. Timings exclude evaluation
    and record CUDA-synchronized accumulated training time. Inputs differ by method.
    """
    out.mkdir(parents=True, exist_ok=False)
    config = dict(steps=steps, eval_steps=sorted(set(eval_steps)|{steps}), seed=seed, lr=lr,
                  selection=selection, initial_prompt_hash=tensor_hash(init),
                  frozen_hash=frozen_hash)
    save(out/'configuration.json', config)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device='cuda').manual_seed(seed)
    rng = np.random.RandomState(seed)
    p = torch.nn.Parameter(init.clone().float())
    opt = torch.optim.Adam([p], lr=lr)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    criterion = HardTripletLoss(margin=.1, hardest=True)
    rows, losses, skipped, nonzero, attempts, completed, train_seconds, eval_seconds = [], [], [], 0, 0, 0, 0., 0.
    if len(support)//2 < 2:
        torch.save(p.detach().cpu(), out/f'prompt-{steps}.pt')
        result = dict(status='insufficient_support_kept_baseline', rows=[dict(step=0,**base)],
                      actual_updates=0, attempts=0, frozen_hash=frozen_hash, saved_prompt_exact=True)
        save(out/'complete.json', result)
        return result
    labels = torch.arange(len(support)//2, device='cuda').repeat_interleave(2)
    with torch.no_grad(): before = float(fp32_loss(model, support, labels, p, criterion))
    wall_start = time.perf_counter()
    with (out/'training.jsonl').open('w', buffering=1) as log:
        while completed < steps:
            attempts += 1
            if attempts > steps + max(20, steps//10):
                raise RuntimeError('Too many AMP skipped updates; fail rather than silently truncate')
            torch.cuda.synchronize(); tick = time.perf_counter()
            ids = rng.permutation(len(support)//2)
            indices = []
            for pid in ids: indices.extend((2*pid+rng.permutation(2)).tolist())
            ii = torch.tensor(indices, device='cuda')
            images = augment(support[ii], gen); y = labels[ii]
            image_hash, label_hash = tensor_hash(images), tensor_hash(y)
            loss = fp32_loss(model, images, y, p, criterion)
            assert torch.isfinite(loss), ('nonfinite loss', attempts)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            grad = p.grad.detach().float()
            finite = bool(torch.isfinite(grad).all())
            gn = float(grad.norm()) if finite else None
            if finite and gn > 0: nonzero += 1
            previous_scale = scaler.get_scale()
            scaler.step(opt); scaler.update()
            skip = scaler.get_scale() < previous_scale
            if skip: skipped.append(attempts)
            else: completed += 1
            record = dict(step=completed, attempt=attempts, loss=float(loss), gradient_norm=gn,
                          gradient_finite=finite, skipped=skip, scale=float(scaler.get_scale()),
                          images_sha256=image_hash, labels_sha256=label_hash,
                          delta_rms=float((p.detach()-init).square().mean().sqrt()))
            log.write(json.dumps(record, allow_nan=False)+'\n'); losses.append(float(loss))
            torch.cuda.synchronize(); train_seconds += time.perf_counter()-tick
            if not skip and completed in config['eval_steps']:
                tick = time.perf_counter()
                score = evaluate(model, ds, p.detach())
                rows.append(dict(step=completed, **score, gain=score['mAP']-base['mAP']))
                torch.save(p.detach().cpu(), out/f'prompt-{completed}.pt')
                save(out/'retrieval.partial.json', rows)
                torch.cuda.synchronize(); eval_seconds += time.perf_counter()-tick
                print('EVAL',out.name,json.dumps(rows[-1]),flush=True)
    updates = int(opt.state[p]['step'])
    assert updates == steps == completed
    assert state_hash(model.state_dict().items()) == frozen_hash
    assert all(v.grad is None for v in model.parameters())
    restored = torch.load(out/f'prompt-{steps}.pt', map_location='cuda', weights_only=True)
    assert torch.equal(restored, p.detach())
    with torch.no_grad(): after = float(fp32_loss(model, support, labels, p, criterion))
    result = dict(status='complete',rows=rows,actual_updates=updates,attempts=attempts,
                  skipped_attempts=skipped,nonzero_gradient_steps=nonzero,
                  support_loss_before=before,support_loss_after=after,
                  frozen_hash=frozen_hash,saved_prompt_exact=True,
                  prompt_hash=tensor_hash(p.detach()),train_seconds=train_seconds,
                  evaluation_seconds=eval_seconds,elapsed_seconds=time.perf_counter()-wall_start,
                  milliseconds_per_update=1000*train_seconds/updates)
    save(out/'complete.json',result)
    return result


def feature_cache(model, records, init, out, metadata):
    start = time.perf_counter()
    features, _, _ = extract(model, records, init)
    raw_norm_error = float((features.norm(dim=1)-1).abs().max())
    features = torch.nn.functional.normalize(features.float(), dim=1)
    x = features.numpy().astype(np.float32)
    assert len(x) == len(records) and np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-5)
    np.save(out/'pool_features.npy', x, allow_pickle=False)
    loaded = np.load(out/'pool_features.npy', allow_pickle=False)
    assert np.array_equal(loaded, x)
    # Same first loader batch, same shape: exact feature-path reproducibility.
    repeat, _, _ = extract(model, records[:min(256,len(records))], init)
    repeat = torch.nn.functional.normalize(repeat.float(), dim=1)
    assert torch.equal(features[:len(repeat)], repeat)
    meta = dict(**metadata, shape=list(x.shape), dtype=str(x.dtype),
                sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                extraction_seconds=time.perf_counter()-start,
                raw_retrieval_norm_max_error=raw_norm_error,
                selector_fp32_normalization=True,
                cache_roundtrip_exact=True, repeated_first_batch_exact=True)
    save(out/'pool_features.json', meta)
    return x


def selection_checks(ds, domain, features, out):
    cameras = [r[2] for r in ds.train]
    pool = AnonymousPool(features, cameras, domain not in NO_CAMERA_DOMAINS)
    feature_hash_before = hashlib.sha256(pool.feats.tobytes()).hexdigest()
    selection_rows, partner_maps, timings = [], {}, []
    for draw in range(3):
        partner_maps[draw] = {}
        for method in METHODS:
            start = time.perf_counter()
            anchors = choose(pool, method, 16, 1600+draw)
            timings.append(dict(method=method,draw=draw,seconds=time.perf_counter()-start))
            assert anchors == choose(pool, method, 16, 1600+draw)
            selection = annotate_selected(ds.train, anchors, has_cameras=pool.has_cameras,
                         domain=domain,split=0,annotation_seed=2600+draw)
            selection.update(method=method,draw=draw,selection_seed=1600+draw,
                             annotation_seed=2600+draw,metrics=selection_metrics(pool,anchors))
            for a,b in selection['annotation']['pair_indices']:
                assert a not in partner_maps[draw] or partner_maps[draw][a] == b
                partner_maps[draw][a] = b
                assert ds.train[a][1] == ds.train[b][1] and a != b
                if pool.has_cameras: assert ds.train[a][2] != ds.train[b][2]
            selection['ordered_support_hash'] = digest_json(selection['pairs'])
            selection['anchor_set_hash'] = digest_json(sorted(anchors))
            selection_rows.append(selection)
    if not pool.has_cameras:
        for draw in range(3):
            same = [r for r in selection_rows if r['draw']==draw and r['method'] in ('facility','facility_camera')]
            assert same[0]['pairs'] == same[1]['pairs']
    save(out/'selections.json', selection_rows)
    assert hashlib.sha256(pool.feats.tobytes()).hexdigest() == feature_hash_before
    save(out/'selection_timings.json', timings)
    save(out/'selection_audit.json', dict(budget_identity='k=n_pairs+n_dup+n_fail',
        reproducible=True,annotation_shared_across_methods=True,
        selector_fields=list(AnonymousPool.__slots__),no_labels_or_paths_in_selector=True,
        unique_anchor_sets={m:len({r['anchor_set_hash'] for r in selection_rows if r['method']==m}) for m in METHODS},
        no_camera_equivalence_checked=not pool.has_cameras))
    return selection_rows


def legacy_check(model,ds,legacy_selection,out,base,init,frozen_hash,domain):
    reference = Path('experiments')/f'step5_{domain}'/'draw0_seed42_lr0.0001'
    assert (reference/'complete.json').exists()
    support = load_images([p for pair in legacy_selection['pairs'] for p in pair], 'cuda')
    actual = fit_prompt(model,ds,support,init,legacy_selection,out/'legacy_random_replay',base,
                        frozen_hash,steps=300,eval_steps=(100,300))
    expected = json.loads((reference/'complete.json').read_text())
    old_lines = [json.loads(x) for x in (reference/'training.jsonl').read_text().splitlines()]
    new_lines = [json.loads(x) for x in (out/'legacy_random_replay'/'training.jsonl').read_text().splitlines()]
    for old,new in zip(old_lines,new_lines):
        for key in ['images_sha256','labels_sha256','loss','gradient_norm','delta_rms']:
            assert old[key] == new[key], (key,old['step'],old[key],new[key])
    assert len(old_lines) == len(new_lines) == 300
    for old,new in zip(expected['rows'],actual['rows']):
        for key in ['step','mAP','rank1','rank5']: assert old[key]==new[key],(key,old,new)
    previous = torch.load(reference/'prompt-300.pt',map_location='cpu',weights_only=True)
    current = torch.load(out/'legacy_random_replay'/'prompt-300.pt',map_location='cpu',weights_only=True)
    assert torch.equal(previous,current)
    save(out/'legacy_replay.json',dict(exact_prompt=True,exact_metrics=True,
        all_300_input_and_gradient_logs_exact=True,reference=str(reference),
        note='Historical annotation stream used ONLY here; formal methods use isolated annotation RNG.'))
    return actual


def replay(out):
    done = json.loads((out/'complete.json').read_text())
    prov = json.loads((out/'provenance.json').read_text())
    model,_ = build_model(Path(prov['checkpoint']))
    ds,_,_ = load_data(done['domain'])
    assert state_hash(model.state_dict().items()) == prov['frozen_hash']
    base = evaluate(model,ds,model.prompt)
    assert base == done['baseline']
    results = []
    for trial in done['trials']:
        prompt = torch.load(out/trial['directory']/f"prompt-{trial['steps']}.pt",map_location='cuda',weights_only=True)
        result = evaluate(model,ds,prompt)
        expected = trial['result']['rows'][-1]
        assert all(result[k] == expected[k] for k in result),(result,expected)
        results.append(dict(directory=trial['directory'],exact=True,metrics=result))
    save(out/'fresh_replay.json',dict(baseline_exact=True,trials=results,
         frozen_hash_exact=True,independent_process=True))
    print('FRESH_REPLAY_COMPLETE',done['domain'],flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',choices=['grid','viper','ilids','cuhk03'])
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--checkpoint',default='experiments/s2_vpt_long/checkpoint-12000')
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--cpu-checks',action='store_true')
    a = parser.parse_args(); out = Path(a.output_dir)
    if a.replay: replay(out); return
    out.mkdir(parents=True,exist_ok=False)
    if a.cpu_checks:
        save(out/'unit_checks.json',unit_checks());print('CPU_CHECKS_PASS',flush=True);return
    assert a.domain
    save(out/'formal_configuration.json',FORMAL)
    ds,manifest,legacy = load_data(a.domain)
    model,args = build_model(Path(a.checkpoint))
    assert a.domain not in args.source_domains.split(',')
    init = model.prompt.detach().clone()
    frozen_hash = state_hash(model.state_dict().items())
    code_files = ['scripts/active_vpt_b1.py','adapters/active_vpt_selection.py',
        'adapters/selectors.py','scripts/tune_vpt_fewshot.py','scripts/oracle_prompt.py',
        'adapters/warm_start_reid.py','adapters/baseline_model.py','ops/losses.py']
    snapshot = out/'code'; snapshot.mkdir()
    for filename in code_files:
        dest = snapshot/filename; dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(filename,dest)
    prov = dict(checkpoint=str(Path(a.checkpoint).resolve()),frozen_hash=frozen_hash,
        initial_prompt_hash=tensor_hash(init),
        code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code_files},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        source_domains=args.source_domains,source_all_images=args.source_all_images,
        torch=torch.__version__,stage='B1; not formal B2; only 20/300 updates')
    save(out/'provenance.json',prov);save(out/'data_manifest.json',manifest)
    base = evaluate(model,ds,init)
    historical = json.loads((Path('experiments')/f'step5_{a.domain}'/'baseline.json').read_text())
    assert base == historical,(base,historical)
    save(out/'baseline.json',base);print('BASELINE',a.domain,base,flush=True)
    metadata = dict(domain=a.domain,split=0,frozen_hash=frozen_hash,initial_prompt_hash=tensor_hash(init),
                    records_hash=digest_json(ds.train),preprocess=FORMAL['feature_source'])
    features = feature_cache(model,ds.train,init,out,metadata)
    selections = selection_checks(ds,a.domain,features,out)
    trials, equivalences, results_by_support = [],[],{}
    for selection in [r for r in selections if r['draw']==0]:
        directory = 'smoke_'+selection['method']
        # Identical ordered support, seed and optimizer produce exactly the same computation.
        signature = selection['ordered_support_hash']
        if signature in results_by_support:
            equivalences.append(dict(method=selection['method'],same_as=results_by_support[signature]))
            continue
        support = load_images([p for pair in selection['pairs'] for p in pair],'cuda')
        result = fit_prompt(model,ds,support,init,selection,out/directory,base,frozen_hash,
                            steps=20,eval_steps=(20,))
        trials.append(dict(directory=directory,steps=20,method=selection['method'],result=result))
        results_by_support[signature] = selection['method']
    random_selection = next(r for r in selections if r['draw']==0 and r['method']=='random')
    support = load_images([p for pair in random_selection['pairs'] for p in pair],'cuda')
    repeat = fit_prompt(model,ds,support,init,random_selection,out/'smoke_random_repeat',base,
                        frozen_hash,steps=20,eval_steps=(20,))
    first = next(r['result'] for r in trials if r['method']=='random')
    assert repeat['prompt_hash'] == first['prompt_hash'] and repeat['rows'] == first['rows']
    assert (out/'smoke_random'/'training.jsonl').read_bytes() == (out/'smoke_random_repeat'/'training.jsonl').read_bytes()
    legacy_result = legacy_check(model,ds,legacy[0],out,base,init,frozen_hash,a.domain)
    trials.append(dict(directory='legacy_random_replay',steps=300,method='historical_random',result=legacy_result))
    final = evaluate(model,ds,init)
    assert final == base and state_hash(model.state_dict().items()) == frozen_hash
    save(out/'complete.json',dict(domain=a.domain,baseline=base,final_baseline=final,
        trials=trials,equivalences=equivalences,random_repeat_exact=True,
        formal_steps=5000,formal_training_launched=False,
        note='20-step scores test integration only; not selector performance evidence.'))
    print('B1_COMPLETE',a.domain,flush=True)


if __name__=='__main__':main()
