"""
exporters/bdt.py
========================================================
行動デジタルツイン(BDT)向けエクスポータ。

出力:
  - digital_twins.pkl : 個人別の「状況→行動分布」モデル群
  - twin_api.py       : twin_query(actor_id, context) を提供

設計:
  - 全体モデル(GBDT)を基底とし、個人の潜在パラメータ(sens_*, latent_coop)
    を state に含めることで「個人ごとに状況→行動分布」を表現
  - 1 actor = 1 twin。context を入れると P(none/reply/share) を返す
  - 反実仮想クエリ対応(urgency を変えたら行動分布がどう動くか)
"""
from .schema import assert_no_leak, state_columns

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def bdt_export(df: pd.DataFrame, cfg: dict, out_dir: str):
    assert_no_leak(df)
    scols = state_columns(df)

    # 多クラス(none/reply/share)を1つの基底モデルで学習。
    # 個人差は state に含まれる sens_*/latent_coop が担う = ツインの個性。
    X = df[scols].values
    y = df["action_id"].values  # 0/1/2
    base = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=8,
        l2_regularization=1.0).fit(X, y)

    # --- 多次元アクション予測ヘッド(行動した人のみで学習) ---
    # 「行動するか」だけでなく「どんな強度・労力の返信か」まで返すツイン
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

    # 個人の固定属性テーブル(twin の素性)
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
        # 行動したらどんな返信か(強度・労力・速さ)
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
    """反実仮想チェック: ツインに urgency+Δ を入れたら share 確率が上がるか。"""
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


# === ノートブック内で使える TwinRegistry（twin_api.py と同じ実装） ===
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
