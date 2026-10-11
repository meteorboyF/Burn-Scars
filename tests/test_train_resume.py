"""Checkpoint/resume: a killed run must resume to bit-identical final weights (CPU)."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from burnscars.paths import REPO_ROOT, data_dir
from burnscars.train.loop import train

torch.use_deterministic_algorithms(True, warn_only=True)


def _toy():
    rng = np.random.default_rng(0)
    data = {f"c{i}": (rng.normal(size=(6, 16, 16)).astype(np.float32),
                      rng.integers(-1, 2, size=(16, 16)).astype(np.int64)) for i in range(10)}

    def load_xy(cid, k):
        x, y = data[cid]
        if k is not None and k >= 4:
            x, y = x[:, :, ::-1].copy(), y[:, ::-1].copy()
        return x, y
    return list(data), load_xy


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(6, 8, 3, padding=1), torch.nn.BatchNorm2d(8),
                               torch.nn.ReLU(), torch.nn.Conv2d(8, 2, 1))


def _sd_hash(m):
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode() + v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_interrupted_run_resumes_bit_identical(tmp_path):
    ids, load_xy = _toy()
    kw = dict(seed=3, epochs=3, batch_size=3, lr=1e-2, weight_decay=0.0, role="outer_test",
              ckpt_every=2, log=lambda *_: None)
    m_full = _model()
    full = train(m_full, ids[:8], ids[8:], load_xy, ckpt_dir=tmp_path / "a", **kw)
    m_kill = _model()
    with pytest.raises(KeyboardInterrupt):
        train(m_kill, ids[:8], ids[8:], load_xy, ckpt_dir=tmp_path / "b", kill_after_step=5, **kw)
    m_res = _model()
    res = train(m_res, ids[:8], ids[8:], load_xy, ckpt_dir=tmp_path / "b", **kw)
    assert _sd_hash(m_full) == _sd_hash(m_res)
    assert res["final_eval"]["confusion"] == full["final_eval"]["confusion"]
    assert [h["train_loss"] for h in res["history"]] == pytest.approx([h["train_loss"] for h in full["history"]])


@pytest.mark.skipif(not (data_dir() / "hls_burn_scars" / "training").exists(), reason="dataset not downloaded")
def test_sigkilled_smoke_trainer_resumes_identically(tmp_path):
    cfg = "configs/smoke/smoke_unet_random_cv5_ibm_trainval.yaml"
    cmd = [sys.executable, str(REPO_ROOT / "tools" / "train_seg.py"), "--config", cfg, "--smoke"]

    def env(d):
        return {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "BURNSCARS_OUTPUTS_DIR": str(d)}

    def weights(d):
        m = json.loads((d / "smoke" / "smoke_unet_random_cv5_ibm_trainval" / "manifest.json").read_text())
        return m["extra"]["final_weights_sha256"]

    a, b = tmp_path / "a", tmp_path / "b"
    subprocess.run(cmd, env=env(a), check=True, capture_output=True)
    p = subprocess.Popen(cmd, env=env(b), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ck = b / "smoke" / "smoke_unet_random_cv5_ibm_trainval" / "ckpt"
    t0 = time.time()
    while not (ck.exists() and any(x.name.startswith("step_") for x in ck.iterdir())):
        assert p.poll() is None and time.time() - t0 < 300, "trainer finished/hung before first checkpoint"
        time.sleep(0.2)
    os.kill(p.pid, signal.SIGKILL)  # a real kill, not an exception
    p.wait()
    assert not (b / "smoke" / "smoke_unet_random_cv5_ibm_trainval" / "manifest.json").exists()
    r = subprocess.run(cmd, env=env(b), check=True, capture_output=True, text=True)
    assert "resumed from step" in r.stdout
    assert weights(a) == weights(b)
