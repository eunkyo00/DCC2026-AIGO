"""Package only this experiment's code and frozen CSV inputs for Colab upload."""
from pathlib import Path
import hashlib
import json
import zipfile

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FILES = [
    "mission-1/lightweight/logmel_cnn/run_logmel_cnn.py",
    "mission-1/lightweight/logmel_cnn/test_logmel_cnn.py",
    "mission-1/lightweight/logmel_cnn/requirements.txt",
    "mission-1/lightweight/logmel_cnn/DESIGN.md",
    "mission-1/lightweight/logmel_cnn/REVIEW.md",
    "mission-1/lightweight/logmel_cnn/README.md",
    "mission-1/lightweight/logmel_cnn/colab/preflight.py",
    "mission-1/lightweight/logmel_cnn/colab/SETUP.md",
    "mission-1/validation/split_assignments.csv",
    "mission-1/validation/manifests/calls.csv",
    "mission-1/eda/eda_outputs/segments.csv",
    "mission-1/baseline/mfcc_svm/results/val_predictions.csv",
]


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    dest = HERE.parent / "artifacts/logmel_colab_bundle.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for relative in FILES:
            path = REPO / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            z.write(path, relative)
            manifest[relative] = {"sha256": hash_file(path), "bytes": path.stat().st_size}
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"path": str(dest), "bytes": dest.stat().st_size,
                      "sha256": hash_file(dest), "files": len(FILES)}, indent=2))


if __name__ == "__main__":
    main()
