"""Exercise .githooks/pre-commit in a throwaway repo (no real data, no network)."""
import json
import shutil
import subprocess

import pytest

from burnscars.paths import REPO_ROOT

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
HOOK = REPO_ROOT / ".githooks" / "pre-commit"


@pytest.fixture
def repo(tmp_path):
    def g(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, capture_output=True, text=True)
    g("init", "-q")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "t")
    (tmp_path / ".githooks").mkdir()
    shutil.copy(HOOK, tmp_path / ".githooks" / "pre-commit")
    g("config", "core.hooksPath", ".githooks")
    return tmp_path, g


def _commit(repo, rel, content):
    root, g = repo
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content if isinstance(content, bytes) else content.encode())
    g("add", "-f", rel)
    return g("commit", "-q", "-m", "x")


def test_allows_clean_code(repo):
    assert _commit(repo, "src/a.py", "x = 1\n").returncode == 0


@pytest.mark.parametrize("rel,content", [
    ("data/chip.txt", "x"),                         # under root /data/
    ("stuff/chip.tif", b"II*\0"),                   # raster extension
    ("ckpt/model.pt", b"\0"),                       # weights
    ("big.txt", "a" * 1_100_000),                   # large blob
    ("report.pdf", b"%PDF"),                        # pdf outside figures dirs
    (".env", "A=1"),                                # secret filename
], ids=["root-data", "tif", "weights", "large", "pdf", "envfile"])
def test_refuses_data_and_blobs(repo, rel, content):
    r = _commit(repo, rel, content)
    assert r.returncode != 0 and "REFUSED" in r.stderr


def test_allows_figure_pdf(repo):
    assert _commit(repo, "paper/figures/f.pdf", b"%PDF").returncode == 0


def test_refuses_notebook_outputs(repo):
    nb = {"cells": [{"cell_type": "code", "execution_count": 1, "metadata": {},
                     "outputs": [{"output_type": "stream", "text": "hi"}], "source": []}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    r = _commit(repo, "n.ipynb", json.dumps(nb))
    assert r.returncode != 0 and "outputs" in r.stderr


@pytest.mark.parametrize("secret", [
    "gh" + "p_" + "A" * 36,
    "h" + "f_" + "b" * 34,
    "AK" + "IA" + "ABCDEFGHIJKLMNOP",
    "-----BEGIN " + "RSA PRIVATE KEY-----",
], ids=["github", "hf", "aws", "privkey"])
def test_refuses_secrets(repo, secret):
    r = _commit(repo, "src/conf.py", f"TOKEN = '{secret}'\n")
    assert r.returncode != 0 and "secret" in r.stderr
