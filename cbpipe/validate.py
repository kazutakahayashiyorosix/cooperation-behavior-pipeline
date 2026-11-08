"""
validation/protocols.py
========================================================
Five data-quality protocols plus a health check for each exporter.
"""
from .schema import assert_no_leak, state_columns
from .export_physical import physical_ai_sanity
from .export_social import social_sim_sanity
from .export_bdt import bdt_sanity

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def run_data_protocols(df: pd.DataFrame, cfg=None):
    """Run P1-P3 and the held-out AUCs.

    If `cfg` is given and `df` has the synthetic generator's state features, the result also
    contains a coefficient-recovery check (see `coefficient_recovery`).
    """
    assert_no_leak(df)
    scols = state_columns(df)
    df = df.copy()
    df["label_act"] = (df["action"] != "none").astype(int)
    y = df["label_act"].values

    rng = np.random.default_rng(7)
    reqs = df["request_id"].unique(); rng.shuffle(reqs)
    te = set(reqs[:len(reqs)//5])
    trm = ~df["request_id"].isin(te).values
    tem = df["request_id"].isin(te).values

    sc = StandardScaler(); X = sc.fit_transform(df[scols])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[trm], y[trm])
    auc = roc_auc_score(y[tem], lr.predict_proba(X[tem])[:, 1])

    # Gradient-boosted trees on the same leakage-free features and the same request-level split.
    Xr = df[scols].to_numpy()
    gb = HistGradientBoostingClassifier(random_state=0).fit(Xr[trm], y[trm])
    gb_auc = roc_auc_score(y[tem], gb.predict_proba(Xr[tem])[:, 1])

    # P1
    bal = y.mean()
    p1 = bal < 0.5
    # P3 same_cluster alone
    sccol = "state__same_cluster"
    if sccol in df.columns:
        Xs = sc.fit_transform(df[[sccol]])
        a = roc_auc_score(y[tem], LogisticRegression(max_iter=500, class_weight="balanced")
                          .fit(Xs[trm], y[trm]).predict_proba(Xs[tem])[:, 1])
    else:
        a = 0.5
    p3 = a < 0.65
    # P2 does context matter (urgency sensitivity or topic_fit)
    coef = dict(zip(scols, lr.coef_[0]))
    p2 = (abs(coef.get("state__req_urgency", 0)) + abs(coef.get("state__u_sens_urgency", 0))
          + abs(coef.get("state__topic_fit", 0))) > 0.05

    out = {
        "P1_nonaction_majority": {"action_rate": float(bal), "pass": bool(p1)},
        "P2_context_effective": {"pass": bool(p2)},
        "P3_label_independent": {"same_cluster_auc": float(a), "pass": bool(p3)},
        "full_model_auc": float(auc),
        "gbm_auc": float(gb_auc),
    }
    if cfg is not None and has_recovery_features(df):
        out["coef_recovery"] = coefficient_recovery(df, cfg)
    return out


# Generating-equation terms that can be rebuilt from state features, paired with the config weight
# that multiplies them in `YorosixWorld.generate`. The load term is omitted because the decaying
# load is not part of the state, so its coefficient cannot be recovered from the features.
RECOVERY_TERMS = {
    "latent_coop": ("w_latent_coop", lambda d: d["state__u_latent_coop"]),
    "urgency": ("w_urgency", lambda d: d["state__u_sens_urgency"] * d["state__req_urgency"]),
    "tie": ("w_tie", lambda d: d["state__u_sens_tie"] * d["state__tie_strength"]),
    "time": ("w_time", lambda d: d["state__u_sens_time"] * d["state__time_match"]),
    "netpos": ("w_netpos", lambda d: d["state__u_netpos"]),
    "topic": ("w_topic", lambda d: d["state__topic_fit"]),
}


RECOVERY_COLUMNS = ["state__u_latent_coop", "state__u_sens_urgency", "state__req_urgency",
                    "state__u_sens_tie", "state__tie_strength", "state__u_sens_time",
                    "state__time_match", "state__u_netpos", "state__topic_fit"]


def has_recovery_features(df: pd.DataFrame) -> bool:
    """True if `df` carries the state features needed to rebuild the generating terms."""
    return all(c in df.columns for c in RECOVERY_COLUMNS)


def coefficient_recovery(df: pd.DataFrame, cfg):
    """Fit a logistic model on the generating-equation terms and compare with the true weights.

    The model is fit unscaled and without class weighting, so its coefficients are on the
    same scale as the generator's weights (up to attenuation from the omitted load term,
    the noise term and the share/reply merge). Returns the per-term recovered and true
    coefficients and their Pearson correlation.
    """
    G = cfg["generation"]
    y = (df["action"] != "none").astype(int).to_numpy()
    X = np.column_stack([f(df).to_numpy(dtype=float) for _, f in RECOVERY_TERMS.values()])
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    rec = lr.coef_[0]
    true = np.array([float(G[k]) for k, _ in RECOVERY_TERMS.values()])
    r = float(np.corrcoef(rec, true)[0, 1])
    return {
        "terms": list(RECOVERY_TERMS),
        "recovered": {n: float(c) for n, c in zip(RECOVERY_TERMS, rec)},
        "true": {n: float(t) for n, t in zip(RECOVERY_TERMS, true)},
        "correlation": r,
    }


def run_exporter_checks(df, cfg, out_dir):
    return {
        "physical_ai": physical_ai_sanity(df, cfg, out_dir),
        "social_sim": social_sim_sanity(df, cfg, out_dir),
        "bdt": bdt_sanity(df, cfg, out_dir),
    }
