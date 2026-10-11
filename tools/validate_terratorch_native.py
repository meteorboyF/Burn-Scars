"""Cross-check: score the published checkpoint on IBM val with TerraTorch's OWN pipeline.

    PYTHONPATH=src python tools/validate_terratorch_native.py --config validate_terratorch_native

Uses the released config with only these changes (recorded in the manifest):
data roots -> a symlink dir of all chips; train/val/test split files ALL -> IBM
val.txt (the IBM test list is never given to TerraTorch); precision -> config
value; backbone_pretrained -> false (weights come from the checkpoint);
batch_size/num_workers for CPU. Runs `trainer.validate` and records TerraTorch's
logged metrics next to ours computed from the same per-batch predictions.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.chips import build_index  # noqa: E402
from burnscars.data.splits import ibm_ids  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_json  # noqa: E402
from burnscars.paths import REPO_ROOT, data_dir  # noqa: E402

TIER = "cpu"


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    for k in ("subdir", "ckpt_subdir", "out", "precision"):
        C.require(ap.add_argument(f"--{k}"))
    C.require(ap.add_argument("--batch_size", type=int))
    args, name, cfg_run = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    rec = RunRecorder("validate_terratorch_native", name, cfg_run, seed=2, smoke=args.smoke)
    root = data_dir() / args.subdir
    idx = build_index(root)
    link_dir = data_dir() / "hls_burn_scars_flat"
    link_dir.mkdir(exist_ok=True)
    for r in idx.itertuples():
        for rel in (r.scene_path, r.mask_path):
            dst = link_dir / Path(rel).name
            if not dst.exists():
                dst.symlink_to(root / rel)
    val_file = data_dir() / "ibm_splits" / "splits" / "val.txt"
    ids = ibm_ids("val")
    if args.smoke:
        val_file = Path(tempfile.mkdtemp()) / "val_smoke.txt"
        val_file.write_text("\n".join(ids[:2]) + "\n")
    ckdir = data_dir() / args.ckpt_subdir
    cfg = yaml.safe_load((ckdir / "burn_scars_config.yaml").read_text())
    d = cfg["data"]["init_args"]
    for k in ("train_data_root", "val_data_root", "test_data_root"):
        d[k] = str(link_dir)
    for k in ("train_split", "val_split", "test_split"):
        d[k] = str(val_file)  # never the IBM test list
    d["batch_size"], d["num_workers"] = args.batch_size, 2
    cfg["model"]["init_args"]["model_args"]["backbone_pretrained"] = False
    cfg["trainer"]["precision"] = args.precision
    cfg["trainer"]["accelerator"] = "cpu"
    cfg["trainer"]["logger"] = False
    cfg["trainer"]["callbacks"] = []
    out = REPO_ROOT / args.out if not args.smoke else REPO_ROOT / "outputs" / "smoke" / name
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "terratorch_config_used.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    exe = Path(sys.executable).parent / "terratorch"
    r = subprocess.run([str(exe), "validate", "-c", str(cfg_path), "--ckpt_path",
                        str(ckdir / "Prithvi_EO_V2_300M_BurnScars.pt")],
                       capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "4"})
    (out / "terratorch_stdout.txt").write_text(r.stdout[-20000:])
    (out / "terratorch_stderr_tail.txt").write_text(r.stderr[-5000:])
    if r.returncode != 0:
        raise SystemExit(f"terratorch validate failed; see {out}/terratorch_stderr_tail.txt")
    metrics = {}
    for line in r.stdout.splitlines():  # Lightning prints a table: "│ val/mIoU │ 0.89... │"
        parts = [p.strip() for p in line.replace("│", "|").split("|") if p.strip()]
        if len(parts) == 2 and parts[0].startswith("val/"):
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                pass
    rec.data["split_hash"] = sha256_json({"ibm_val": sorted(ids)})
    rec.finish(out, metrics, extra={"n_val_ids": len(ids) if not args.smoke else 2,
                                    "config_changes": "roots->flat symlinks; all splits->IBM val.txt; "
                                                      f"precision={args.precision}; backbone_pretrained=false; "
                                                      f"batch_size={args.batch_size}; accelerator=cpu"})
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
