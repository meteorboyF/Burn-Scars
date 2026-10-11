"""Metric + preprocessing validation with the PUBLISHED Prithvi-EO-2.0-300M-BurnScars checkpoint.

    PYTHONPATH=src python tools/validate_metric_prithvi.py --config validate_metric_prithvi [--smoke]

Runs the released checkpoint (CPU, fp32, no training) on IBM **val** (never test)
with TerraTorch's own datamodule transforms (ToTensorV2 + Normalize from the
released config), scores it with burnscars.eval.metrics AND independently with
torchmetrics MulticlassJaccardIndex(average="macro", ignore_index=-1), and
compares with the model card's val numbers (IoU-burned 84.28, mIoU 90.95).

A match validates OUR METRIC AND PREPROCESSING ONLY. It does not reproduce
training, and the card's checkpoint was itself selected on this val split.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.chips import build_index, dataset_hash  # noqa: E402
from burnscars.data.splits import assert_eval_allowed, ibm_ids  # noqa: E402
from burnscars.eval.metrics import aggregate, chip_scores  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_file, sha256_json  # noqa: E402
from burnscars.paths import REPO_ROOT, data_dir, outputs_dir  # noqa: E402
from burnscars.seed import set_seed  # noqa: E402

TIER = "cpu"


def load_model(ckpt_dir: Path):
    """Build via TerraTorch from the released config. backbone_pretrained is turned off
    in an in-memory copy (avoids fetching base weights that the checkpoint overwrites);
    then every checkpoint tensor is verified to be in the model, bit-exact."""
    from terratorch.cli_tools import LightningInferenceModel
    cfg = yaml.safe_load((ckpt_dir / "burn_scars_config.yaml").read_text())
    cfg["model"]["init_args"]["model_args"]["backbone_pretrained"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
    ckpt = ckpt_dir / "Prithvi_EO_V2_300M_BurnScars.pt"
    lim = LightningInferenceModel.from_config(tmp.name, str(ckpt))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    msd = lim.model.state_dict()
    missing = [k for k in sd if k not in msd]
    differ = [k for k in sd if k in msd and not torch.equal(sd[k].to(msd[k].dtype), msd[k])]
    if missing or differ or len(sd) != len(msd):
        raise SystemExit(f"checkpoint not loaded exactly: missing={missing[:3]} differ={differ[:3]} "
                         f"n_ckpt={len(sd)} n_model={len(msd)}")
    lim.model.eval()
    return lim.model, lim.datamodule, cfg


def predict(model, dm, x: np.ndarray, precision: str = "fp32") -> np.ndarray:
    """Exactly the released inference path: test_transform -> datamodule.aug -> model -> argmax.
    precision="bf16": CPU autocast to bfloat16, mimicking the config's bf16-mixed."""
    t = dm.test_transform(image=x.transpose(1, 2, 0))
    t["image"] = t["image"].unsqueeze(0)
    t = dm.aug(t)["image"]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        out = model(t).output.float()
    return out.argmax(dim=1)[0].numpy().astype(np.int64)


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    for k in ("subdir", "ckpt_subdir", "out"):
        C.require(ap.add_argument(f"--{k}"))
    C.require(ap.add_argument("--card_val_iou_burned", type=float))
    C.require(ap.add_argument("--card_val_miou", type=float))
    C.require(ap.add_argument("--tolerance_pts", type=float))
    C.require(ap.add_argument("--threads", type=int))
    ap.add_argument("--max_chips", type=int, default=None, help="CLI only: timing runs")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    set_seed(0)
    torch.set_num_threads(args.threads)
    root = data_dir() / args.subdir
    rec = RunRecorder("validate_metric_prithvi", name, cfg, seed=0, smoke=args.smoke)

    idx = build_index(root).set_index("chip_id")
    val, train = ibm_ids("val"), ibm_ids("train")
    # the card model was trained on IBM train (and selected on IBM val): never test
    assert_eval_allowed("ibm", train, "ibm_val", val)
    n = 2 if args.smoke else args.max_chips
    chips = val[:n] if n else val

    model, dm, _ = load_model(data_dir() / args.ckpt_subdir)
    from torchmetrics.classification import MulticlassJaccardIndex
    tm_macro = MulticlassJaccardIndex(num_classes=2, ignore_index=-1, average="macro")
    tm_none = MulticlassJaccardIndex(num_classes=2, ignore_index=-1, average="none")

    rows, scores, secs = [], [], []
    pred_dir = outputs_dir() / "validate_metric_prithvi" / name / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i, cid in enumerate(chips):
        r = idx.loc[cid]
        with rasterio.open(root / r.scene_path) as s:
            x = s.read().astype(np.float32)
        with rasterio.open(root / r.mask_path) as m:
            y = m.read(1).astype(np.int64)
        t0 = time.perf_counter()
        p = predict(model, dm, x, args.precision)
        secs.append(time.perf_counter() - t0)
        np.savez_compressed(pred_dir / f"{cid}.npz", pred=p.astype(np.uint8))
        s = chip_scores(y, p)
        scores.append(s)
        tm_macro.update(torch.from_numpy(p), torch.from_numpy(y))
        tm_none.update(torch.from_numpy(p), torch.from_numpy(y))
        rows.append({"chip_id": cid, "iou_burned": s["iou_burned"], "iou_unburned": s["iou_unburned"],
                     "miou": s["miou"], "n_valid": s["n_valid"], "cm": json.dumps(s["cm"].tolist()),
                     "seconds": secs[-1]})
        print(f"{i + 1}/{len(chips)} {cid} {secs[-1]:.1f}s iou_b={s['iou_burned']:.3f}", flush=True)

    agg = aggregate(scores)
    tm_cls = tm_none.compute().tolist()
    metrics = {
        "n_chips": len(chips), "full_val": len(chips) == len(val),
        "ours": {k: v for k, v in agg.items() if k != "confusion"}, "confusion": agg["confusion"],
        "torchmetrics_macro_miou": float(tm_macro.compute()),
        "torchmetrics_iou_per_class": tm_cls,
        "card_val_iou_burned": args.card_val_iou_burned, "card_val_miou": args.card_val_miou,
        "delta_iou_burned_pts": 100 * agg["dataset_iou_burned"] - args.card_val_iou_burned,
        "delta_miou_pts": 100 * agg["dataset_miou"] - args.card_val_miou,
        "seconds_per_chip": {"mean": float(np.mean(secs)), "median": float(np.median(secs)),
                             "first": float(secs[0]), "threads": args.threads},
        "precision": args.precision,
        "label": "metric+preprocessing validation only; not a training reproduction; "
                 "card checkpoint was selected on this val split; CPU fp32 vs card bf16-mixed",
    }
    metrics["ours_vs_torchmetrics_abs_diff"] = abs(metrics["ours"]["dataset_miou"] - metrics["torchmetrics_macro_miou"])
    metrics["within_tolerance"] = bool(metrics["full_val"] and abs(metrics["delta_iou_burned_pts"]) <= args.tolerance_pts
                                       and abs(metrics["delta_miou_pts"]) <= args.tolerance_pts)
    out = REPO_ROOT / args.out if not (args.smoke or args.max_chips) else outputs_dir() / "validate_metric_prithvi" / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_chip.csv", index=False, float_format="%.6g")
    rec.data["dataset_hash"] = dataset_hash(root, idx.loc[chips].reset_index())
    rec.data["split_hash"] = sha256_json({"ibm_val": sorted(val)})
    rec.finish(out, metrics, extra={"checkpoint_sha256": sha256_file(
        data_dir() / args.ckpt_subdir / "Prithvi_EO_V2_300M_BurnScars.pt"),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads()})
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
