"""Default configuration. Values reproduce the results reported in the preprint (Section 4.1)."""
import copy

DEFAULT_CONFIG = {
    "world": {"n_members": 5000, "n_posts": 2000, "seed": 42, "n_clusters": 20, "n_topics": 6},
    "generation": {"beta0_share": -5.0, "beta0_reply": -4.6, "w_latent_coop": 1.8,
                   "w_urgency": 2.0, "w_tie": 1.4, "w_time": 0.7, "w_load": 0.9,
                   "w_netpos": 1.0, "w_topic": 1.1, "noise_sd": 0.35},
    "counterfactual": {"enabled": True, "urgency_delta": 0.3},
    "output": {"dir": "cbpipe_output", "format": "parquet"},
    "exporters": {
        "physical_ai": {"reward_share": 1.0, "reward_reply": 0.5, "reward_none": 0.0},
        "social_sim": {"framework": "mesa"},
        "bdt": {"model": "gradient_boosting"},
    },
}


def load_config(path=None, **overrides):
    """Return a deep copy of the default config, optionally updated from a YAML/JSON file."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        if str(path).endswith((".yml", ".yaml")):
            import yaml
            user = yaml.safe_load(open(path))
        else:
            import json
            user = json.load(open(path))
        _merge(cfg, user or {})
    _merge(cfg, overrides)
    return cfg


def _merge(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
