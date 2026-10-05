#!/bin/bash
# Pilot watchdog (user requests 2026-10-05), for plans/pilot8r.tasks:
#   pilot verdict GOOD -> the planned runs continue; when the launcher has finished them all: shut down.
#   pilot verdict BAD  -> stop the cluster-repair follow-ups (launcher + eval_active / sim_selection processes),
#                         run plans/fallback.queue on every free GPU until it is empty; then shut down.
#   a pilot task crashes -> not a result: logged, no shutdown, the watchdog exits (fixed by hand).
#   DRY=1 bash watchdog/watchdog.sh  -> decide and log only (never kills, never shuts down)
cd /root/FERReid_CR || exit 1
PY=/root/miniconda3/bin/python
LOG=/root/FERReid_CR/watchdog.log
L=experiments/_launch
J="$PY /root/FERReid_CR/watchdog/judge.py"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
ok() { [ "$(cat $L/$1.status 2>/dev/null)" = "0" ]; }
failed() { [ -e "$L/$1.status" ] && [ "$(cat $L/$1.status)" != "0" ]; }

collect_results() {
  # small result files (csv / json / launcher + watchdog logs; no checkpoints, no prompts) -> one tarball on the
  # data disk; the local monitor downloads it, stores a local copy and commits it to the repository
  local R=results/pilot_cuhk03_$(date +%Y%m%d)
  rm -rf "$R"; mkdir -p "$R"
  for d in experiments/*/; do
    n=$(basename "$d"); case $n in _launch|base_*) continue;; esac
    mkdir -p "$R/$n"; cp "$d"/*.csv "$d"/*.json "$d"/CANCELLED "$R/$n/" 2>/dev/null
    [ -e "$d/active.log" ] && grep -vE "Warning|warn\(|it/s\]" "$d/active.log" > "$R/$n/active.log.txt"
    rmdir "$R/$n" 2>/dev/null
  done
  cp watchdog.log "$R/watchdog.log.txt"; cp pilot8r.out "$R/launcher.out.txt" 2>/dev/null
  cp experiments/_launch/fallback_worker_gpu*.log "$R/" 2>/dev/null
  tar czf /root/autodl-tmp/FERReid_CR_results.tar.gz "$R"
  echo "$R" > "$L/RESULTS_READY"
  log "results packed: /root/autodl-tmp/FERReid_CR_results.tar.gz ($R, $(du -sh $R | cut -f1))"
}

shutdown_now() {
  log "FINISHED: $*"
  collect_results
  local waited=0
  while [ ! -e "$L/RESULTS_FETCHED" ] && [ $waited -lt 120 ]; do sleep 60; waited=$((waited + 1)); done
  if [ -e "$L/RESULTS_FETCHED" ]; then log "results fetched by the local monitor"; else log "results not fetched after 120 min (kept on the data disk)"; fi
  log "SHUTDOWN"
  [ -n "$DRY" ] && { log "DRY run: no shutdown"; exit 0; }
  sync
  /usr/bin/shutdown
  exit 0
}

fallback_workers() {
  cp plans/fallback.queue "$L/fallback.queue"
  log "fallback queue: $(wc -l < $L/fallback.queue) items"
  declare -A wpid
  while true; do
    alive=0
    for g in 0 1 2 3 4 5 6 7; do
      if [ -n "${wpid[$g]}" ] && kill -0 "${wpid[$g]}" 2>/dev/null; then alive=$((alive + 1)); continue; fi
      [ -n "${wpid[$g]}" ] && { log "worker on GPU $g finished"; unset "wpid[$g]"; }
      [ -s "$L/fallback.queue" ] || continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ "$used" -lt 2000 ] || continue
      CUDA_VISIBLE_DEVICES=$g setsid bash -c "cd /root/FERReid_CR; source configs/local.sh; export PATH=/root/miniconda3/bin:\$PATH; \
        source plans/common.sh; run_fallback_queue" >> "$L/fallback_worker_gpu$g.log" 2>&1 < /dev/null &
      wpid[$g]=$!
      alive=$((alive + 1))
      log "worker started on GPU $g (pid ${wpid[$g]})"
      sleep 30
    done
    if [ $alive -eq 0 ] && [ ! -s "$L/fallback.queue" ]; then log "fallback queue done"; return; fi
    sleep 60
  done
}

log "watchdog v3 (pilot8r; shut down when the planned runs are done) started (pid $$)"
while true; do
  for t in pilot_s0a pilot_s0b pilot_s1a pilot_s1b pilot_s2a pilot_s2b pilot_pair_cover pilot_pair_random; do
    if failed $t; then log "PILOT TASK CRASHED: $t (status $(cat $L/$t.status)) -- not a result; no shutdown; watchdog exits"; exit 0; fi
  done
  if ok merge_pilot; then
    v=$($J pilot experiments/pilot/active.csv)
    log "pilot verdict: $v"
    case "$v" in
      BAD*)
        [ -n "$DRY" ] && { log "DRY run: nothing stopped"; exit 0; }
        echo "$v" > "$L/STOP_REPAIR"
        pkill -f "scripts/launch_tasks.py"
        pkill -f "scripts/(eval_active|sim_selection)\.py"
        sleep 30
        pkill -9 -f "scripts/(eval_active|sim_selection)\.py"
        log "launcher and cluster-repair follow-ups stopped"
        fallback_workers
        shutdown_now "pilot bad; fallback validation runs finished";;
      GOOD*)
        log "pilot good: waiting for the launcher to finish the planned runs"
        while pgrep -f "scripts/launch_tasks.py" > /dev/null; do sleep 60; done
        log "launcher finished: $(grep -c ' done ' pilot8r.out) done, $(grep -c 'FAILED' pilot8r.out) failed"
        shutdown_now "all planned runs finished";;
    esac
  fi
  sleep 60
done
