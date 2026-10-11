"""Download pinned Hugging Face files into the gitignored data dir, verify SHA256, extract.

    PYTHONPATH=src python tools/download_data.py --config download_hls_burn_scars

Never redistributes data: it fetches from the source at a pinned revision into
data_dir() (outside git). Resumes partial downloads with HTTP Range. Refuses
to extract unless size and SHA256 match the config.
"""
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_file, sha256_json  # noqa: E402
from burnscars.paths import data_dir  # noqa: E402

TIER = "cpu"


def url_for(repo_type, repo, revision, filename):
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{filename}"


def fetch(url, dest: Path, expected_size: int, chunk=1 << 22, retries=20):
    """Resumable download. The file is renamed to `dest` ONLY when complete."""
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        if dest.stat().st_size == expected_size:
            print(f"already downloaded: {dest}")
            return
        dest.rename(part)  # incomplete file from an older run: resume it
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == expected_size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    raise RuntimeError("server ignored Range; delete the .part file and retry")
                done = have
                while block := r.read(chunk):
                    f.write(block)
                    done += len(block)
                    if done // (1 << 28) != (done - len(block)) // (1 << 28):
                        print(f"  {done / 1e9:.2f} / {expected_size / 1e9:.2f} GB", flush=True)
        except (OSError, urllib.error.URLError) as e:
            print(f"  connection error ({e}); resuming (attempt {attempt + 1}/{retries})", flush=True)
    if not part.exists() or part.stat().st_size != expected_size:
        raise SystemExit(f"incomplete download after {retries} attempts: {part}")
    part.rename(dest)


def safe_extract(archive: Path, out: Path):
    marker = out / f".extracted_{archive.name}"
    if marker.exists():
        print(f"already extracted: {archive.name}")
        return
    with tarfile.open(archive) as tf:
        tf.extractall(out, filter="data")  # rejects absolute paths / traversal
    marker.touch()


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    for k in ("repo", "repo_type", "revision", "subdir"):
        C.require(ap.add_argument(f"--{k}"))
    C.require(ap.add_argument("--files", type=dict,
                              help="mapping filename -> {size, sha256} (config only)"))
    ap.add_argument("--extract", type=list, default=[], help="archives to extract (config only)")
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)

    root = data_dir() / args.subdir
    root.mkdir(parents=True, exist_ok=True)
    rec = RunRecorder("download_data", name, cfg, seed=None, smoke=args.smoke)
    digests = {}
    for fname, spec in args.files.items():
        dest = root / fname
        url = url_for(args.repo_type, args.repo, args.revision, fname)
        if args.smoke:
            print(f"smoke: would fetch {url} -> {dest} ({spec['size']:,} bytes)")
            continue
        print(f"fetching {url}", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fetch(url, dest, spec["size"])
        size = dest.stat().st_size
        if size != spec["size"]:
            raise SystemExit(f"{fname}: size mismatch {size} != {spec['size']}")
        digest = sha256_file(dest)
        if digest != spec["sha256"]:
            raise SystemExit(f"{fname}: SHA256 mismatch {digest} != {spec['sha256']}")
        print(f"verified {fname} SHA256 {digest}")
        digests[fname] = digest
    if args.smoke:
        return
    for fname in args.extract:
        safe_extract(root / fname, root)
        print(f"extracted {fname} into {root}")
    rec.data["dataset_hash"] = sha256_json(digests)
    rec.finish(root / "_manifests" / name, {"files": digests})


if __name__ == "__main__":
    main()
