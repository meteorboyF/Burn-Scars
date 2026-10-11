"""YAML configs validated against an entrypoint's argparse parser.

Every tool: `python tools/<tool>.py --config <name> [--smoke]`. Config keys must
be argparse dests of that tool; an unknown key or a missing required key fails
loudly. Precedence: parser defaults < config file < explicit CLI flags.
"""
import argparse
import sys
from pathlib import Path

import yaml

from .paths import configs_dir

_RESERVED = {"config", "smoke"}


class ConfigError(ValueError):
    pass


def load_yaml(name_or_path) -> tuple[str, dict]:
    p = Path(name_or_path)
    if not p.suffix:
        p = configs_dir() / f"{name_or_path}.yaml"
    if not p.is_file():
        raise ConfigError(f"config not found: {p}")
    cfg = yaml.safe_load(p.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return p.stem, cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="config name in configs/ or a path")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run of the full code path")
    return ap


def parse(ap: argparse.ArgumentParser, argv=None) -> tuple[argparse.Namespace, str, dict]:
    """Parse CLI, merge the config, validate. Returns (args, config_name, config_dict)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    pre, _ = ap.parse_known_args(argv)
    name, cfg = load_yaml(pre.config)

    dests = {a.dest for a in ap._actions if a.dest != "help"}
    unknown = sorted(set(cfg) - dests)
    if unknown:
        raise ConfigError(f"config {name!r}: unknown keys {unknown}; valid: {sorted(dests - _RESERVED)}")
    reserved = sorted(set(cfg) & _RESERVED)
    if reserved:
        raise ConfigError(f"config {name!r}: keys {reserved} are CLI-only")

    # config overrides defaults; explicit CLI flags override config
    ap.set_defaults(**cfg)
    args = ap.parse_args(argv)
    missing = [a.dest for a in ap._actions
               if getattr(a, "_burn_required", False) and getattr(args, a.dest) is None]
    if missing:
        raise ConfigError(f"config {name!r}: missing required keys {missing}")
    return args, name, cfg


def require(action: argparse.Action) -> argparse.Action:
    """Mark an argument as required-from-config-or-CLI (argparse's own `required`
    would reject values supplied by the YAML)."""
    action._burn_required = True
    return action
