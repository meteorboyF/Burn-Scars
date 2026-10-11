"""Shared, resumable training loop for both model families.

Determinism by construction: the chip order of epoch e is a permutation drawn
from default_rng((seed, e)); the D4 augmentation of item i in epoch e is drawn
from default_rng((seed, e, i)). So a run killed and resumed from a checkpoint at
step s continues with exactly the batches the uninterrupted run would have seen
(tests/test_train_resume.py checks bit-identical final weights on CPU).

Model selection never touches the held-out set of an outer fold: role
"outer_test" evaluates ONCE, after the last step; per-epoch evaluation is only
allowed for role "inner_val" (hyper-parameter tuning).
"""
import hashlib
import io
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import checkpoint as ck
from ..eval.metrics import aggregate, chip_scores

ROLES = ("outer_test", "inner_val")


def _order(seed, epoch, n):
    return np.random.default_rng((seed, epoch)).permutation(n)


def _aug_k(seed, epoch, i):
    return int(np.random.default_rng((seed, epoch, i)).integers(8))


def batches(seed, epoch, n, batch_size, start=0):
    """Yield (position, index array) for one epoch, starting at batch `start`."""
    order = _order(seed, epoch, n)
    nb = math.ceil(n / batch_size)
    for b in range(start, nb):
        yield b, order[b * batch_size:(b + 1) * batch_size]


def _state_bytes(model, opt, sched, step, history, ep_loss) -> bytes:
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "history": history, "ep_loss": ep_loss}, buf)
    return buf.getvalue()


def model_hash(model) -> str:
    """SHA256 over model tensors only (deterministic content; payload bytes include wall-clock)."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model, items, load_xy, device, batch_size=4):
    """items: list of chip ids. Returns (aggregate, per-chip rows, predictions dict)."""
    model.eval()
    scores, rows, preds = [], [], {}
    with torch.no_grad():
        for s in range(0, len(items), batch_size):
            ids = items[s:s + batch_size]
            xs, ys = zip(*(load_xy(c, None) for c in ids))
            logits = model(torch.from_numpy(np.stack(xs)).to(device))
            p = logits.argmax(1).cpu().numpy().astype(np.int64)
            for cid, pi, yi in zip(ids, p, ys):
                sc = chip_scores(yi, pi)
                scores.append(sc)
                preds[cid] = pi.astype(np.uint8)
                rows.append({"chip_id": cid, "iou_burned": sc["iou_burned"],
                             "iou_unburned": sc["iou_unburned"], "miou": sc["miou"],
                             "n_valid": sc["n_valid"], "cm": json.dumps(sc["cm"].tolist())})
    model.train()
    return aggregate(scores), rows, preds


def train(model, train_ids, eval_ids, load_xy, *, seed, epochs, batch_size, lr, weight_decay,
          role, ckpt_dir, ckpt_every, device="cpu", augment=True, kill_after_step=None, log=print):
    """load_xy(chip_id, aug_k|None) -> (x float32 (C,H,W), y int64 (H,W)).

    Returns dict(history, final_eval, per_chip, preds, steps)."""
    if role not in ROLES:
        raise ValueError(role)
    model.to(device).train()
    n = len(train_ids)
    steps_per_epoch = math.ceil(n / batch_size)
    max_steps = epochs * steps_per_epoch
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps)))
    step, history, ep_loss = 0, [], []
    resumed = ck.latest_valid(ckpt_dir, max_steps)
    if resumed:
        st = torch.load(io.BytesIO(resumed[1]), map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, history, ep_loss = st["step"], st["history"], st["ep_loss"]
        log(f"resumed from step {step}/{max_steps} model_sha256={model_hash(model)[:12]}")

    t0 = time.time()
    while step < max_steps:
        epoch, start = divmod(step, steps_per_epoch)
        if start == 0:
            ep_loss = []
        for _, idx in batches(seed, epoch, n, batch_size, start):
            xs, ys = zip(*(load_xy(train_ids[i], _aug_k(seed, epoch, int(i)) if augment else None)
                           for i in idx))
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            loss = F.cross_entropy(model(x), y, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            ep_loss.append(float(loss.detach()))
            if step % ckpt_every == 0 and step % steps_per_epoch != 0:  # epoch ends saved below
                ck.save(ckpt_dir, step, max_steps, _state_bytes(model, opt, sched, step, history, ep_loss),
                        extra_meta={"model_sha256": model_hash(model)})
            if kill_after_step is not None and step >= kill_after_step:
                raise KeyboardInterrupt(f"simulated kill at step {step}")
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(ep_loss)) if ep_loss else None,
               "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
        if role == "inner_val":
            agg, _, _ = evaluate(model, eval_ids, load_xy, device)
            rec |= {"val_dataset_iou_burned": agg["dataset_iou_burned"], "val_dataset_miou": agg["dataset_miou"]}
        history.append(rec)
        log(json.dumps(rec))
        ck.save(ckpt_dir, step, max_steps, _state_bytes(model, opt, sched, step, history, ep_loss),
                        extra_meta={"model_sha256": model_hash(model)})

    final, rows, preds = evaluate(model, eval_ids, load_xy, device)
    return {"history": history, "final_eval": final, "per_chip": rows, "preds": preds,
            "steps": step, "max_steps": max_steps, "steps_per_epoch": steps_per_epoch}
