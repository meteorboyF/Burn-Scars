"""One-off frozen-feature extraction: Prithvi-EO-2.0-300M BASE encoder, all 804 chips.

    PYTHONPATH=src python tools/extract_prithvi_features.py --config extract_prithvi2_base [--smoke]

Deterministic forward pass (no seeds needed; eval mode, no dropout). Inputs are
scaled to Prithvi's pretraining units (reflectance x 1e4) and normalised with its
pretraining mean/std (fold-independent, so one cache serves every split).
Output: <data_dir>/<out_subdir>/<chip_id>.npy, fp16, shape (4, 1024, 32, 32),
plus features_manifest.json with per-chip SHA256. Resumable: chips whose file
already exists and verifies are skipped. Copy the folder to Drive afterwards.
Tier "cpu": allowed on the laptop (~8 s/chip measured) or on an approved GPU.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from burnscars import config as C  # noqa: E402
from burnscars.data.dataset import ChipStore, normalise  # noqa: E402
from burnscars.hardware import check_tier  # noqa: E402
from burnscars.manifest import RunRecorder, sha256_bytes, sha256_file, sha256_json  # noqa: E402
from burnscars.models import PRITHVI_LAYERS, build_prithvi_encoder, prithvi_features  # noqa: E402
from burnscars.paths import data_dir  # noqa: E402

TIER = "cpu"
STATS = {"mean": (np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]) / 1e4).tolist(),
         "std": (np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]) / 1e4).tolist()}


def main(argv=None):
    ap = C.base_parser(__doc__.splitlines()[0])
    C.require(ap.add_argument("--weights_subdir"))
    C.require(ap.add_argument("--weights_file"))
    C.require(ap.add_argument("--weights_sha256"))
    C.require(ap.add_argument("--out_subdir"))
    ap.add_argument("--threads", type=int, default=4)
    args, name, cfg = C.parse(ap, argv)
    check_tier(TIER, args.smoke)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = RunRecorder("extract_prithvi_features", name, cfg, seed=None, smoke=args.smoke)

    store = ChipStore()
    ids = sorted(store.index.index)
    if args.smoke:
        enc, ids, size = build_prithvi_encoder(pretrained=False), ids[:2], 64
        out = data_dir() / "smoke_features"
    else:
        w = data_dir() / args.weights_subdir / args.weights_file
        if sha256_file(w) != args.weights_sha256:
            raise SystemExit(f"weights SHA256 mismatch: {w}")
        enc, size = build_prithvi_encoder(pretrained=True, ckpt_path=w), None
        out = data_dir() / args.out_subdir
    enc.to(device)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "features_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    secs = []
    for i, cid in enumerate(ids):
        f = out / f"{cid}.npy"
        if cid in man and f.exists() and sha256_file(f) == man[cid]:
            continue
        x, _ = store.read(cid, size)
        xn = torch.from_numpy(normalise(x, STATS))[None].to(device)
        t0 = time.perf_counter()
        feats = prithvi_features(enc, xn)[0].half().cpu().numpy()
        secs.append(time.perf_counter() - t0)
        tmp = out / f".{cid}.npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, feats)
        tmp.rename(f)
        man[cid] = sha256_file(f)
        if i % 25 == 0 or args.smoke:
            man_path.write_text(json.dumps(man, indent=0, sort_keys=True))
            print(f"{i + 1}/{len(ids)} {cid} {feats.shape} {secs[-1]:.1f}s", flush=True)
    man_path.write_text(json.dumps(man, indent=0, sort_keys=True))
    rec.data["dataset_hash"] = sha256_json(sorted(man.items()))
    rec.finish(out / "_manifest", {"n_chips": len(man), "layers": list(PRITHVI_LAYERS),
                                   "seconds_per_chip_mean": float(np.mean(secs)) if secs else None,
                                   "device": device, "stats": STATS,
                                   "features_hash": sha256_bytes(json.dumps(man, sort_keys=True).encode())})


if __name__ == "__main__":
    main()
