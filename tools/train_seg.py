"""Train + evaluate one (model, split, fold, seed) cell. GPU tier; --smoke runs on CPU.

    PYTHONPATH=src python tools/train_seg.py --config <cell_config> [--smoke]

model=unet           : from-scratch UNet on normalised chips (per-run train stats).
model=prithvi_decoder: light decoder on cached frozen Prithvi-EO-2.0 features
                       (tools/extract_prithvi_features.py); smoke computes features
                       on the fly with a RANDOM-init encoder (plumbing only).
Writes to <outputs>/<out>/: manifest.json (train_split family + train/eval ids for
the leakage guards), history.csv, per_chip.csv, preds.npz, model_final.pt.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.dataset import ChipStore, band_stats, d4, normalise  # noqa: E402
from burnscars.data.splits import assert_eval_allowed  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_file  # noqa: E402
from burnscars.models import LightDecoder, build_prithvi_encoder, build_unet, n_params, prithvi_features  # noqa: E402
from burnscars.paths import REPO_ROOT, data_dir, outputs_dir  # noqa: E402
from burnscars.seed import set_seed  # noqa: E402
from burnscars.train.loop import train  # noqa: E402

TIER = "gpu"
PRITHVI_MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], np.float32) / 1e4
PRITHVI_STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], np.float32) / 1e4
FAMILY = {"official": "official", "ibm": "ibm", "random_cv": "cv_random_684", "grouped_cv": "cv_grouped_684"}


def select_ids(spec: dict, fold: int, inner_fold: int, role: str):
    f = spec["folds"][fold]
    if role == "outer_test":
        if inner_fold != -1:
            raise SystemExit("outer_test runs use the full outer training set (inner_fold=-1)")
        return list(f["train"]), list(f["test"])
    inner = f["inner"][inner_fold]
    return list(inner["train"]), list(inner["val"])


def drop_same_tile_pairs(train_ids, eval_ids):
    """Ablation: remove training chips on the same MGRS tile as any eval chip (other dates)."""
    eval_tiles = {c.split(".")[0] for c in eval_ids}
    return [c for c in train_ids if c.split(".")[0] not in eval_tiles]


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    C.require(ap.add_argument("--model", choices=["unet", "prithvi_decoder"]))
    C.require(ap.add_argument("--split_file"))
    C.require(ap.add_argument("--fold", type=int))
    ap.add_argument("--inner_fold", type=int, default=-1)
    C.require(ap.add_argument("--role", choices=["outer_test", "inner_val"]))
    C.require(ap.add_argument("--seed", type=int))
    C.require(ap.add_argument("--epochs", type=int))
    C.require(ap.add_argument("--batch_size", type=int))
    C.require(ap.add_argument("--lr", type=float))
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--augment", type=lambda s: str(s).lower() == "true", default=True)
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--drop_same_tile_pairs", type=lambda s: str(s).lower() == "true", default=False)
    ap.add_argument("--features_dir", default="prithvi2_base_features")
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kill_after_step", type=int, default=None, help="CLI only: resume tests")
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"

    spec = json.loads((REPO_ROOT / args.split_file).read_text())
    if spec["kind"] not in FAMILY or (spec["kind"].endswith("_cv") and spec.get("pool") != "ibm_trainval"):
        raise SystemExit(f"split kind {spec['kind']!r} / pool {spec.get('pool')!r} not in the registered design")
    family = FAMILY[spec["kind"]]
    train_ids, eval_ids = select_ids(spec, args.fold, args.inner_fold, args.role)
    eval_name = {"official": "official_val", "ibm": "ibm_val"}.get(family, f"{family}_fold{args.fold}") \
        if args.role == "outer_test" else f"{family}_fold{args.fold}_inner{args.inner_fold}"
    if args.drop_same_tile_pairs:
        train_ids = drop_same_tile_pairs(train_ids, eval_ids)
    if args.train_frac < 1.0:
        rng = np.random.default_rng((args.seed, 7))
        train_ids = sorted(rng.choice(sorted(train_ids), int(round(args.train_frac * len(train_ids))),
                                      replace=False).tolist())
    assert_eval_allowed(family, train_ids, eval_name, eval_ids)

    size = None
    epochs, bs = args.epochs, args.batch_size
    if args.smoke:
        train_ids, eval_ids, size, epochs, bs = train_ids[:6], eval_ids[:3], 64, 2, 2
    store = ChipStore()
    rec = RunRecorder("train_seg", name, cfg, seed=args.seed, smoke=args.smoke)

    if args.model == "unet":
        stats = band_stats(store, train_ids, n_px_per_chip=256 if args.smoke else 4096, seed=args.seed)
        model = build_unet()

        def load_xy(cid, k):
            x, y = store.read(cid, size)
            x = normalise(x, stats)
            return d4(x, y, k) if k is not None else (x, y)
    else:
        stats = {"mean": PRITHVI_MEAN.tolist(), "std": PRITHVI_STD.tolist(), "source": "Prithvi-EO-2.0 pretraining"}
        model = LightDecoder()
        if args.smoke:
            enc = build_prithvi_encoder(pretrained=False)
            cache = {}

            def feats(cid):
                if cid not in cache:
                    x, _ = store.read(cid, size)
                    xn = normalise(x, stats)
                    cache[cid] = prithvi_features(enc, torch.from_numpy(xn)[None])[0].half().numpy()
                return cache[cid]
        else:
            fdir = data_dir() / args.features_dir

            def feats(cid):
                return np.load(fdir / f"{cid}.npy")

        def load_xy(cid, k):
            f = feats(cid).astype(np.float32)
            _, y = store.read(cid, size)
            if k is None:
                return f, y
            L, D, h, w = f.shape
            fx, y = d4(f.reshape(L * D, h, w), y, k)  # feature-space D4, consistent with the mask
            return fx.reshape(L, D, h, w), y

    out = outputs_dir() / (args.out or f"train_seg/{name}") if not args.smoke else outputs_dir() / "smoke" / name
    out.mkdir(parents=True, exist_ok=True)
    res = train(model, train_ids, eval_ids, load_xy, seed=args.seed, epochs=epochs, batch_size=bs,
                lr=args.lr, weight_decay=args.weight_decay, role=args.role, ckpt_dir=out / "ckpt",
                ckpt_every=args.ckpt_every, device=device, augment=args.augment,
                kill_after_step=args.kill_after_step)
    pd.DataFrame(res["history"]).to_csv(out / "history.csv", index=False)
    pd.DataFrame(res["per_chip"]).to_csv(out / "per_chip.csv", index=False, float_format="%.6g")
    np.savez_compressed(out / "preds.npz", **res["preds"])
    torch.save(model.state_dict(), out / "model_final.pt")
    if args.smoke:
        cid = eval_ids[0]
        print("smoke decoded example:", cid, "pred burned px:", int(res["preds"][cid].sum()),
              "| y burned px:", int((store.read(cid, size)[1] == 1).sum()))
    metrics = {k: v for k, v in res["final_eval"].items()}
    rec.data["split_hash"] = sha256_file(REPO_ROOT / args.split_file)
    rec.finish(out, metrics, extra={
        "train_split": family, "train_ids": train_ids, "eval_name": eval_name, "eval_ids": eval_ids,
        "role": args.role, "norm_stats": stats, "n_trainable_params": n_params(model),
        "steps": res["steps"], "steps_per_epoch": res["steps_per_epoch"], "device": device,
        "final_weights_sha256": sha256_file(out / "model_final.pt")})
    print(json.dumps({"eval": eval_name, **{k: metrics[k] for k in ("dataset_iou_burned", "dataset_miou")}}))


if __name__ == "__main__":
    main()
