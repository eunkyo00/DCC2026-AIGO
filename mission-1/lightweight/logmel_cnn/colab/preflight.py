"""Read-only Colab checks before the first full extraction."""
from collections import Counter
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_logmel_cnn.py"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--check-all-paths", action="store_true")
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("logmel_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mission = ROOT.parents[1]
    jobs, split_hash = module.load_jobs(mission / "validation/split_assignments.csv",
                                        mission / "validation/manifests/calls.csv",
                                        mission / "eda/eda_outputs/segments.csv", args.data_root)
    probe_ids = {0, len(jobs) // 2, len(jobs) - 1}
    probe = [jobs[i] for i in sorted(probe_ids)]
    missing = [j["wav_path"] for j in (jobs if args.check_all_paths else probe) if not Path(j["wav_path"]).is_file()]
    output = {"data_root": str(args.data_root), "split_sha256": split_hash,
              "calls": len(jobs), "segments": sum(j["n_segments"] for j in jobs),
              "partition_counts": dict(Counter(j["partition"] for j in jobs)),
              "probe_paths": [j["wav_path"] for j in probe],
              "checked_paths": len(jobs) if args.check_all_paths else len(probe),
              "missing_count": len(missing), "missing_examples": missing[:5],
              "free_runtime_gib": round(shutil.disk_usage("/content" if Path("/content").is_dir() else ".").free / 1024**3, 2)}
    try:
        import torch
        output["torch"] = torch.__version__
        output["cuda_available"] = torch.cuda.is_available()
        output["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        output["torch"] = None
        output["cuda_available"] = False
        output["gpu_name"] = None
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit("WAV path check failed; confirm DATA_ROOT contains Training/...")


if __name__ == "__main__":
    main()
