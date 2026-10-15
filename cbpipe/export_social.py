"""
exporters/social_sim.py
========================================================
ソーシャルシミュレーション(ABM / Mesa)向けエクスポータ。

出力:
  - agent_policies.json : 各エージェントの行動確率モデル係数(個人別)
  - init_network.graphml: 初期ネットワーク(クラスタ + bridge エッジ)
  - mesa_model.py       : Mesa 互換の最小モデル(政策を読んで時間発展)

設計:
  - 各エージェントの行動を、state→P(share) のロジスティック係数として要約
  - シミュレータはこの係数で確率的に行動を再生 → 生成元の統計を再現できる
"""
from .schema import assert_no_leak, state_columns

import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression


def social_sim_export(df: pd.DataFrame, cfg: dict, out_dir: str, world=None):
    assert_no_leak(df)
    scols = state_columns(df)

    # 全体ポリシー(共有の基底)+ クラスタ別の補正を係数として保存
    X = df[scols].values
    y = (df["action"] == "share").astype(int).values
    base = LogisticRegression(max_iter=2000).fit(X, y)

    policies = {
        "feature_order": scols,
        "global": {"coef": base.coef_[0].tolist(), "intercept": float(base.intercept_[0])},
        "per_cluster": {},
    }
    for c, g in df.groupby("cluster_id"):
        if len(g) < 50 or g["action"].nunique() < 2:
            continue
        Xc = g[scols].values
        yc = (g["action"] == "share").astype(int).values
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        m = LogisticRegression(max_iter=1000).fit(Xc, yc)
        policies["per_cluster"][int(c)] = {
            "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}

    with open(f"{out_dir}/agent_policies.json", "w") as f:
        json.dump(policies, f, indent=2)

    # 初期ネットワーク
    G = nx.Graph()
    if world is not None:
        for c, members in world.cluster_members.items():
            members = list(members)
            for i in range(len(members)):
                G.add_node(int(members[i]), cluster=int(c))
                # クラスタ内リング + 一部ランダム
                G.add_edge(int(members[i]), int(members[(i + 1) % len(members)]))
        # bridge 同士を繋ぐ
        br = list(world.bridges)
        for i in range(len(br)):
            G.add_edge(int(br[i]), int(br[(i + 1) % len(br)]))
    else:
        # world 無し: イベントデータからネットワークを復元
        # actor を cluster でグルーピングしてクラスタ内リングを張る
        actor_cluster = df.groupby("actor_id")["cluster_id"].first().to_dict()
        from collections import defaultdict
        by_cluster = defaultdict(list)
        for a, c in actor_cluster.items():
            G.add_node(int(a), cluster=int(c))
            by_cluster[int(c)].append(int(a))
        for c, members in by_cluster.items():
            for i in range(len(members)):
                G.add_edge(members[i], members[(i + 1) % len(members)])
    nx.write_graphml(G, f"{out_dir}/init_network.graphml")

    # Mesa モデル stub
    stub = '''"""
Mesa-compatible ABM that replays YOROSIX agent policies.
    from mesa_model import YorosixModel
    model = YorosixModel("agent_policies.json", "init_network.graphml")
    for _ in range(100): model.step()
"""
import json, numpy as np, networkx as nx
import mesa

def _sigmoid(x): return 1/(1+np.exp(-x))

class YorosixAgent(mesa.Agent):
    def __init__(self, uid, model, coef, intercept, order):
        super().__init__(uid, model)
        self.coef = np.array(coef); self.b = intercept; self.order = order
    def act(self, state_vec):
        z = float(self.coef @ state_vec + self.b)
        return np.random.rand() < _sigmoid(z)

class YorosixModel(mesa.Model):
    def __init__(self, policies_path, graph_path):
        super().__init__()
        pol = json.load(open(policies_path))
        self.G = nx.read_graphml(graph_path)
        self.order = pol["feature_order"]
        self.g_coef = pol["global"]["coef"]; self.g_b = pol["global"]["intercept"]
        self.per_cluster = pol["per_cluster"]
        self.schedule = mesa.time.RandomActivation(self)
        for n in self.G.nodes():
            c = str(self.G.nodes[n].get("cluster", -1))
            p = self.per_cluster.get(c, {"coef": self.g_coef, "intercept": self.g_b})
            self.schedule.add(YorosixAgent(int(n), self, p["coef"], p["intercept"], self.order))
    def step(self):
        self.schedule.step()
'''
    with open(f"{out_dir}/mesa_model.py", "w") as f:
        f.write(stub)

    return {"policies": f"{out_dir}/agent_policies.json",
            "network": f"{out_dir}/init_network.graphml",
            "mesa_model": f"{out_dir}/mesa_model.py",
            "n_agents": G.number_of_nodes(),
            "n_cluster_policies": len(policies["per_cluster"])}


def social_sim_sanity(df: pd.DataFrame, cfg: dict, out_dir: str):
    """再現性チェック: ポリシーで再生した行動率が元データと近いか。"""
    pol = json.load(open(f"{out_dir}/agent_policies.json"))
    scols = pol["feature_order"]
    coef = np.array(pol["global"]["coef"]); b = pol["global"]["intercept"]
    X = df[scols].values
    p = 1 / (1 + np.exp(-(X @ coef + b)))
    rng = np.random.default_rng(0)
    sim_share_rate = (rng.random(len(p)) < p).mean()
    real_share_rate = (df["action"] == "share").mean()
    return {"real_share_rate": float(real_share_rate),
            "simulated_share_rate": float(sim_share_rate),
            "abs_diff": float(abs(sim_share_rate - real_share_rate)),
            "reproduced": bool(abs(sim_share_rate - real_share_rate) < 0.05)}
