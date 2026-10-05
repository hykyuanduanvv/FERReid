"""Run a list of independent experiment commands, one per GPU, several GPUs in parallel.

Task file format (plans/*.tasks):
    # comment
    <name> | <shell command>          one task; name = [A-Za-z0-9_.-]+
    <name> after=<a>,<b> | <command>  starts only once tasks a and b (of any group / file) have finished OK;
                                      with --merge, every task starts as soon as a GPU and its inputs are free
    @wait                             barrier: wait until every task above has finished
Commands run with bash from the repository root, after `source configs/local.sh` (if present) and
`source plans/common.sh`, with CUDA_VISIBLE_DEVICES set to one GPU.

Logs:  experiments/_launch/<name>.log ; status: experiments/_launch/<name>.status (exit code).
A task whose status file says 0 is skipped on the next run (resumable); failed tasks are re-run.

    python scripts/launch_tasks.py plans/stage1_loss.tasks --gpus 0,1,2,3
    python scripts/launch_tasks.py plans/stage1_loss.tasks plans/stage2_converge.tasks --gpus 0,1 --dry-run
Several task files run one after another (an implicit @wait between files).
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "experiments" / "_launch"
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
AFTER = {}  # task name -> names of the tasks it waits for


def parse(path):
    groups, current = [], []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "@wait":
            groups.append(current)
            current = []
            continue
        if "|" not in line:
            raise SystemExit("{}:{}: expected '<name> | <command>' or '@wait'".format(path, n))
        head, cmd = (s.strip() for s in line.split("|", 1))
        name, *opts = head.split()
        after = []
        for o in opts:
            if not o.startswith("after="):
                raise SystemExit("{}:{}: unknown option {!r}".format(path, n, o))
            after += [x for x in o[len("after="):].split(",") if x]
        if not NAME.match(name):
            raise SystemExit("{}:{}: bad task name {!r}".format(path, n, name))
        AFTER[name] = after
        current.append((name, cmd))
    groups.append(current)
    return [g for g in groups if g]


def done(name):
    status = LAUNCH / (name + ".status")
    return status.exists() and status.read_text().strip() == "0"


def start(name, cmd, gpu):
    prelude = "set -o pipefail; cd {root}; [ -f configs/local.sh ] && source configs/local.sh; source plans/common.sh; ".format(
        root=str(ROOT).replace(" ", "\\ "))
    log = open(LAUNCH / (name + ".log"), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), FERREID_TASK=name)
    print("[{}] start {} on GPU {}".format(time.strftime("%H:%M:%S"), name, gpu), flush=True)
    return subprocess.Popen(["bash", "-c", prelude + cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT), log


def run_group(tasks, gpus, dry_run):
    todo = [(n, c) for n, c in tasks if not done(n)]
    for n, _ in tasks:
        if done(n):
            print("skip {} (already finished)".format(n))
    if dry_run:
        for n, c in todo:
            print("would run {}: {}".format(n, c))
        return True
    free, running, ok = list(gpus), {}, True

    def failed(n):
        st = LAUNCH / (n + ".status")
        return n not in running and all(n != t for t, _ in todo) and not (st.exists() and st.read_text().strip() == "0")

    while todo or running:
        for name, cmd in list(todo):  # a prerequisite failed (or never ran): this task cannot run
            if any(failed(d) for d in AFTER.get(name, [])):
                todo.remove((name, cmd))
                (LAUNCH / (name + ".status")).write_text("skipped: prerequisite failed")
                print("[{}] {} SKIPPED (prerequisite failed: {})".format(time.strftime("%H:%M:%S"), name,
                                                                      AFTER[name]), flush=True)
                ok = False
        while free:
            ready = [t for t in todo if all(done(d) for d in AFTER.get(t[0], []))]
            if not ready:
                break
            todo.remove(ready[0])
            name, cmd = ready[0]
            gpu = free.pop(0)
            proc, log = start(name, cmd, gpu)
            running[name] = (proc, log, gpu)
        time.sleep(5)
        for name in list(running):
            proc, log, gpu = running[name]
            code = proc.poll()
            if code is None:
                continue
            log.close()
            (LAUNCH / (name + ".status")).write_text(str(code))
            print("[{}] {} {} (GPU {})".format(time.strftime("%H:%M:%S"), name,
                                              "done" if code == 0 else "FAILED exit={}".format(code), gpu), flush=True)
            ok &= code == 0
            free.append(gpu)
            del running[name]
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_files", nargs="+")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids, one task per GPU at a time")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="continue with later groups after a failure")
    parser.add_argument("--merge", action="store_true",
                        help="run all tasks of all files as one pool (ignores @wait); for independent stages")
    a = parser.parse_args()
    gpus = [g for g in a.gpus.split(",") if g != ""]
    LAUNCH.mkdir(parents=True, exist_ok=True)
    groups = [g for path in a.task_files for g in parse(path)]
    if a.merge:
        groups = [[t for g in groups for t in g]]
    names = [n for g in groups for n, _ in g]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit("duplicate task names: {}".format(sorted(dup)))
    for group in groups:
        if not run_group(group, gpus, a.dry_run) and not a.keep_going:
            sys.exit("stopping: a task failed (see experiments/_launch/*.log); rerun to retry failed tasks")
    print("all task groups finished")


if __name__ == "__main__":
    main()
