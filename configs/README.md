# configs/

One YAML per experiment, named `<tool>_<what>.yaml`. Keys must be argparse
dests of the tool (unknown keys fail). A config that has produced a manifest
committed under `results/manifests/` is IMMUTABLE: `tests/test_repo_hygiene.py`
checks every committed manifest's `config_hash` against the current file. To
change it, copy to a new name.
