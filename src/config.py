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

    holds_s entries are stored as {demo, full} and flattened to the active
    value. full_timeline=True sets demo_mode False and uses the full holds.
    """
    with open(path or DEFAULT_PATH) as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    if full_timeline:
        cfg["demo_mode"] = False
    key = "demo" if cfg["demo_mode"] else "full"
    cfg["holds_s"] = {k: v[key] for k, v in cfg["holds_s"].items()}
    return cfg
