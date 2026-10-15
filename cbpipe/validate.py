"""
validation/protocols.py
========================================================
データ品質の5プロトコル + 各エクスポータの健全性。
"""
from .schema import assert_no_leak, state_columns
from .export_physical import physical_ai_sanity
from .export_social import social_sim_sanity
from .export_bdt import bdt_sanity

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def run_data_protocols(df: pd.DataFrame):
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

    # P1
    bal = y.mean()
    p1 = bal < 0.5
    # P3 same_cluster 単独
    sccol = "state__same_cluster"
    if sccol in df.columns:
        Xs = sc.fit_transform(df[[sccol]])
        a = roc_auc_score(y[tem], LogisticRegression(max_iter=500, class_weight="balanced")
                          .fit(Xs[trm], y[trm]).predict_proba(Xs[tem])[:, 1])
    else:
        a = 0.5
    p3 = a < 0.65
    # P2 文脈が効いてるか(urgency感受性 or topic_fit)
    coef = dict(zip(scols, lr.coef_[0]))
    p2 = (abs(coef.get("state__req_urgency", 0)) + abs(coef.get("state__u_sens_urgency", 0))
          + abs(coef.get("state__topic_fit", 0))) > 0.05

    return {
        "P1_nonaction_majority": {"action_rate": float(bal), "pass": bool(p1)},
        "P2_context_effective": {"pass": bool(p2)},
        "P3_label_independent": {"same_cluster_auc": float(a), "pass": bool(p3)},
        "full_model_auc": float(auc),
    }


def run_exporter_checks(df, cfg, out_dir):
    return {
        "physical_ai": physical_ai_sanity(df, cfg, out_dir),
        "social_sim": social_sim_sanity(df, cfg, out_dir),
        "bdt": bdt_sanity(df, cfg, out_dir),
    }
