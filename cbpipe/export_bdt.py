"""
exporters/bdt.py
========================================================
Exporter for Behavioral Digital Twins (BDT).

Outputs:
  - digital_twins.pkl : per-person "situation -> action distribution" models
  - twin_api.py       : provides twin_query(actor_id, context)

Design:
  - A global model (GBDT) is the base; individual latent parameters (sens_*, latent_coop)
    are included in the state, so each person gets their own situation -> action distribution
  - 1 actor = 1 twin. Given a context it returns P(none/reply/share)
  - Supports counterfactual queries (how the action distribution moves when urgency changes)
"""
from .schema import assert_no_leak, state_columns

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def bdt_export(df: pd.DataFrame, cfg: dict, out_dir: str):
    assert_no_leak(df)
    scols = state_columns(df)

    # Train multi-class (none/reply/share) with a single base model.
    # Individual differences are carried by sens_*/latent_coop in the state = the twin's personality.
    X = df[scols].values
    y = df["action_id"].values  # 0/1/2
    base = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=8,
        l2_regularization=1.0).fit(X, y)

    # --- Multi-dimensional action prediction heads (trained on actors who acted only) ---
    # The twin returns not only *whether* someone acts but also the intensity/effort of the reply
    from sklearn.ensemble import HistGradientBoostingRegressor
    acted = df[df["action"] != "none"]
    intensity_head = effort_head = speed_head = None
    if len(acted) >= 50:
        Xa = acted[scols].values
        intensity_head = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6).fit(Xa, acted["act_intensity"].values)
        effort_head = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6).fit(Xa, acted["act_effort"].values)
        speed_head = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6).fit(Xa, acted["act_speed"].values)

    # Per-person fixed-attribute table (the twin's identity features)
    fixed_cols = [c for c in scols if c.startswith("state__u_")]
    twin_profiles = (df.groupby("actor_id")[fixed_cols].first().to_dict("index"))

    bundle = {
        "base_model": base,
        "intensity_head": intensity_head,
        "effort_head": effort_head,
        "speed_head": speed_head,
        "feature_order": scols,
        "fixed_cols": fixed_cols,
        "twin_profiles": twin_profiles,
        "classes": ["none", "reply", "share"],
    }
    with open(f"{out_dir}/digital_twins.pkl", "wb") as f:
        pickle.dump(bundle, f)

    api = '''"""
BDT query API (multi-dimensional action).
    from twin_api import TwinRegistry
    reg = TwinRegistry("digital_twins.pkl")
    reg.query(actor_id=42, context={"req_urgency":0.8,"tie_strength":1.0,...})
      -> {"action_dist": {none,reply,share}, "if_acted": {intensity,effort,speed}}
    reg.counterfactual(actor_id=42, base_context={...}, urgency_delta=0.3)
"""
import pickle, numpy as np

class TwinRegistry:
    def __init__(self, path):
        b = pickle.load(open(path, "rb"))
        self.m = b["base_model"]; self.order = b["feature_order"]
        self.fixed = b["fixed_cols"]; self.prof = b["twin_profiles"]
        self.classes = b["classes"]
        self.ih = b.get("intensity_head"); self.eh = b.get("effort_head")
        self.sh = b.get("speed_head")
    def _vec(self, actor_id, context):
        prof = self.prof.get(actor_id, {})
        row = []
        for c in self.order:
            if c in prof: row.append(prof[c])
            elif c.replace("state__","") in context:
                row.append(context[c.replace("state__","")])
            elif c in context: row.append(context[c])
            else: row.append(0.0)
        return np.array(row).reshape(1, -1)
    def query(self, actor_id, context):
        x = self._vec(actor_id, context)
        p = self.m.predict_proba(x)[0]
        out = {"action_dist": dict(zip(self.classes, p.tolist()))}
        # If the actor acts, what kind of reply (intensity, effort, speed)
        if self.ih is not None:
            out["if_acted"] = {
                "intensity": float(self.ih.predict(x)[0]),
                "effort": float(self.eh.predict(x)[0]),
                "speed": float(self.sh.predict(x)[0]),
            }
        return out
    def counterfactual(self, actor_id, base_context, urgency_delta=0.3):
        cf = dict(base_context)
        cf["req_urgency"] = min(1.0, cf.get("req_urgency", 0.5) + urgency_delta)
        return {"base": self.query(actor_id, base_context),
                "counterfactual": self.query(actor_id, cf)}
'''
    with open(f"{out_dir}/twin_api.py", "w") as f:
        f.write(api)

    return {"twins": f"{out_dir}/digital_twins.pkl",
            "api": f"{out_dir}/twin_api.py",
            "n_twins": len(twin_profiles), "state_dim": len(scols)}


def bdt_sanity(df: pd.DataFrame, cfg: dict, out_dir: str):
    """Counterfactual check: does share probability rise when urgency+delta is fed to the twin?"""
    import pickle
    b = pickle.load(open(f"{out_dir}/digital_twins.pkl", "rb"))
    m = b["base_model"]; order = b["feature_order"]
    delta = cfg["counterfactual"]["urgency_delta"]
    uidx = order.index("state__req_urgency") if "state__req_urgency" in order else None
    if uidx is None:
        return {"counterfactual_supported": False}
    X = df[order].values.copy()
    p_base = m.predict_proba(X)[:, 2].mean()       # class 2 = share
    Xcf = X.copy(); Xcf[:, uidx] = np.clip(Xcf[:, uidx] + delta, 0, 1)
    p_cf = m.predict_proba(Xcf)[:, 2].mean()
    return {"share_prob_base": float(p_base),
            "share_prob_cf": float(p_cf),
            "lift": float(p_cf - p_base),
            "direction_correct": bool(p_cf > p_base)}


# === TwinRegistry usable inside a notebook (same implementation as twin_api.py) ===
class TwinRegistry:
    def __init__(self, path):
        b = pickle.load(open(path, "rb"))
        self.m = b["base_model"]; self.order = b["feature_order"]
        self.fixed = b["fixed_cols"]; self.prof = b["twin_profiles"]
        self.classes = b["classes"]
        self.ih = b.get("intensity_head"); self.eh = b.get("effort_head")
        self.sh = b.get("speed_head")
    def _vec(self, actor_id, context):
        prof = self.prof.get(actor_id, {})
        row = []
        for c in self.order:
            if c in prof: row.append(prof[c])
            elif c.replace("state__", "") in context:
                row.append(context[c.replace("state__", "")])
            elif c in context: row.append(context[c])
            else: row.append(0.0)
        return np.array(row).reshape(1, -1)
    def query(self, actor_id, context):
        x = self._vec(actor_id, context)
        p = self.m.predict_proba(x)[0]
        out = {"action_dist": dict(zip(self.classes, [round(v,3) for v in p.tolist()]))}
        if self.ih is not None:
            out["if_acted"] = {
                "intensity": round(float(self.ih.predict(x)[0]), 3),
                "effort": round(float(self.eh.predict(x)[0]), 3),
                "speed": round(float(self.sh.predict(x)[0]), 3),
            }
        return out
    def counterfactual(self, actor_id, base_context, urgency_delta=0.3):
        cf = dict(base_context)
        cf["req_urgency"] = min(1.0, cf.get("req_urgency", 0.5) + urgency_delta)
        return {"base": self.query(actor_id, base_context),
                "counterfactual": self.query(actor_id, cf)}
