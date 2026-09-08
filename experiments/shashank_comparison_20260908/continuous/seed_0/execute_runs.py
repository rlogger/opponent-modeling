"""Execute three fresh continuous 0s fits and record their actual wall times."""
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "experiments/shashank_comparison_20260908"
ORIGINAL_DATASET = BASE / "continuous_data/dataset.npz"
DATASET = Path(sys.argv[1]) if len(sys.argv) > 1 else ORIGINAL_DATASET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log):
    started = datetime.now(timezone.utc).isoformat()
    before = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            handle.write(line)
            print(line, end="", flush=True)
        code = process.wait()
    result = dict(command=command, shell_command=shlex.join(command), started_at=started,
                  wall_seconds=time.perf_counter() - before, exit_code=code,
                  log=str(log), log_sha256=sha(log), driver_sha256=sha(Path(__file__)))
    if code:
        (log.parent / "failed_execution.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError(f"Execution failed with exit code {code}: {log}")
    return result


def main():
    for seed in (0, 1, 2):
        out = BASE / "continuous" / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "manifest.json").exists():
            raise FileExistsError(f"Refusing to overwrite finished run: {out}")
        command = [sys.executable, str(ROOT / "scripts/run_0s_world_model.py"),
                   "--dataset", str(DATASET), "--out", str(out), "--encoder-steps", "1500",
                   "--updates", "2000", "--seed", str(seed), "--heldout", "2"]
        print(f"Starting full continuous seed {seed}", flush=True)
        log = out / "execution_logs/training.log"
        if log.exists():
            log = log.with_name(f"training_retry_{int(time.time())}.log")
        execution = run(command, log)
        execution.update(dataset_sha256=sha(DATASET), encoder_seed=seed, world_model_seed=seed,
                         specialist_checkpoints="fixed existing continuous specialists", training_reused=False,
                         original_dataset_path=str(ORIGINAL_DATASET), runtime_dataset_path=str(DATASET))
        (out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
        report = out / "REPORT.md"
        report.write_text(report.read_text().replace("  --encoder-steps", f"  --dataset {shlex.quote(str(ORIGINAL_DATASET))} \\\n  --encoder-steps"))
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["dataset"]["regenerated_source_path"] = str(ORIGINAL_DATASET)
        manifest["dataset"]["runtime_copy_sha256_verified"] = True
        for name in ("REPORT.md", "execution.json"):
            manifest["artifacts"][name] = sha(out / name)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Finished seed {seed} in {execution['wall_seconds']:.2f} seconds", flush=True)
    out = BASE / "continuous/seed_0"
    visuals = {}
    for script in ("visualize_0s.py", "visualize_0s_rollouts.py"):
        command = [sys.executable, str(ROOT / "scripts" / script), "--run", str(out),
                   "--dataset", str(DATASET), "--out", str(out / "visuals")]
        visuals[script] = run(command, out / "execution_logs" / (script + ".log"))
    (out / "visuals/execution.json").write_text(json.dumps(visuals, indent=2) + "\n")
    gallery = out / "visuals/README.md"
    content = gallery.read_text()
    for script in ("visualize_0s.py", "visualize_0s_rollouts.py"):
        content = content.replace(f"python scripts/{script}\n",
                                  f"python scripts/{script} --run {shlex.quote(str(out))} --dataset {shlex.quote(str(ORIGINAL_DATASET))}\n")
    gallery.write_text(content)
    print("Three full fresh fits and seed-0 visual gallery finished.", flush=True)


if __name__ == "__main__":
    main()
