"""Mission config loader."""
import copy
import os

import yaml

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "mission.yaml",
)


def load_config(path=None, full_timeline=False):
    """Load mission YAML into a nested dict.

    holds_s (and phase_timeout_s) entries are stored as {demo, full} and
    flattened to the active value. full_timeline=True sets demo_mode False and uses the full holds.
    """
    with open(path or DEFAULT_PATH) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    if full_timeline:
        cfg["demo_mode"] = False
    key = "demo" if cfg["demo_mode"] else "full"
    cfg["holds_s"] = {k: v[key] for k, v in cfg["holds_s"].items()}
    if "phase_timeout_s" in cfg:  # same {demo, full} shape; null = no timeout
        cfg["phase_timeout_s"] = {k: (v[key] if isinstance(v, dict) else v)
                                  for k, v in cfg["phase_timeout_s"].items()}
    return cfg
