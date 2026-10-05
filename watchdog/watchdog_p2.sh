#!/bin/bash
# Pilot-2 watchdog: when the launcher of plans/pilot2.tasks has finished, pack the results, wait (max 120 min) for the
# local monitor to fetch them, then shut the instance down. No verdict, nothing stopped early.
cd /root/FERReid_CR || exit 1
LOG=/root/FERReid_CR/watchdog.log
L=experiments/_launch
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

collect_results() {
  local R=results/pilot2_cuhk03_$(date +%Y%m%d)
  rm -rf "$R"; mkdir -p "$R"
  for d in experiments/p2_*/ experiments/pilot2/; do
    [ -d "$d" ] || continue
    n=$(basename "$d"); mkdir -p "$R/$n"
    cp "$d"/*.csv "$d"/*.json "$R/$n/" 2>/dev/null
    [ -e "$d/active.log" ] && grep -vE "Warning|warn\(|it/s\]" "$d/active.log" > "$R/$n/active.log.txt"
  done
  cp watchdog.log "$R/watchdog.log.txt"; cp pilot2.out "$R/launcher.out.txt" 2>/dev/null
  tar czf /root/autodl-tmp/FERReid_CR_results.tar.gz "$R"
  echo "$R" > "$L/RESULTS_READY"
  log "results packed: /root/autodl-tmp/FERReid_CR_results.tar.gz ($R, $(du -sh $R | cut -f1))"
}

log "pilot-2 watchdog started (pid $$)"
sleep 30
while pgrep -f "scripts/launch_tasks.py plans/pilot2" > /dev/null; do sleep 60; done
log "FINISHED: launcher done ($(grep -c ' done ' pilot2.out) done, $(grep -c 'FAILED' pilot2.out) failed)"
collect_results
waited=0
while [ ! -e "$L/RESULTS_FETCHED" ] && [ $waited -lt 120 ]; do sleep 60; waited=$((waited + 1)); done
if [ -e "$L/RESULTS_FETCHED" ]; then log "results fetched by the local monitor"; else log "results not fetched after 120 min (kept on the data disk)"; fi
log "SHUTDOWN"
sync
/usr/bin/shutdown
