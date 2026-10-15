"""
exporters/physical_ai.py
========================================================
フィジカルAI(模倣学習 / オフラインRL)向けエクスポータ。

出力:
  - transitions.parquet : (state, action, reward, next_state, done)
  - env_stub.py         : Gymnasium 互換の最小 env(SB3 で読める)

設計:
  - 各 actor の時系列イベントを「軌跡(trajectory)」として連結
  - state = schema の state_features(リーク無し)
  - action = none/reply/share の離散3
  - reward = config の reward_* (share に最大報酬)
  - next_state = 同 actor の次イベントの state(無ければ done=1)
"""
from .schema import assert_no_leak, state_columns

import pandas as pd
import numpy as np


def physical_ai_export(df: pd.DataFrame, cfg: dict, out_dir: str):
    assert_no_leak(df)
    rcfg = cfg["exporters"]["physical_ai"]
    reward_map = {"none": rcfg["reward_none"],
                  "reply": rcfg["reward_reply"],
                  "share": rcfg["reward_share"]}
    scols = state_columns(df)

    # actor ごとに時系列ソートして軌跡化(ベクトル化)
    df = df.sort_values(["actor_id", "timestamp"]).reset_index(drop=True)
    S = df[scols].to_numpy(dtype=float)
    actor = df["actor_id"].to_numpy()
    act_id = df["action"].map({"none": 0, "reply": 1, "share": 2}).to_numpy()
    rewards = df["action"].map(reward_map).to_numpy()

    # next state = 次行。ただし actor が変わる境界は done=1, ns=0
    n = len(df)
    NS = np.zeros_like(S)
    done = np.ones(n, dtype=int)
    same_next = actor[:-1] == actor[1:]
    NS[:-1][same_next] = S[1:][same_next]
    done[:-1][same_next] = 0

    tdf = pd.DataFrame(S, columns=[f"s__{c}" for c in scols])
    tdf.insert(0, "actor_id", actor)
    tdf["action_id"] = act_id
    tdf["reward"] = rewards
    for j, c in enumerate(scols):
        tdf[f"ns__{c}"] = NS[:, j]
    tdf["done"] = done
    path = f"{out_dir}/transitions.parquet"
    tdf.to_parquet(path, index=False)

    # Gym env stub (SB3 互換)
    stub = f'''"""
Gymnasium-compatible offline env over YOROSIX transitions.
Usage with Stable-Baselines3 (offline / behavior cloning):
    from env_stub import YorosixOfflineEnv
    env = YorosixOfflineEnv("transitions.parquet")
"""
import gymnasium as gym
import numpy as np, pandas as pd
from gymnasium import spaces

STATE_DIM = {len(scols)}
N_ACTIONS = 3

class YorosixOfflineEnv(gym.Env):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.scols = [c for c in self.df.columns if c.startswith("s__")]
        self.observation_space = spaces.Box(-np.inf, np.inf, ({len(scols)},), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.i = 0
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.i = 0
        return self.df.loc[0, self.scols].values.astype(np.float32), {{}}
    def step(self, action):
        row = self.df.loc[self.i]
        reward = float(row["reward"]) if action == row["action_id"] else 0.0
        done = bool(row["done"]); self.i = min(self.i + 1, len(self.df) - 1)
        ns = self.df.loc[self.i, self.scols].values.astype(np.float32)
        return ns, reward, done, False, {{}}
'''
    with open(f"{out_dir}/env_stub.py", "w") as f:
        f.write(stub)

    return {
        "transitions": path, "env_stub": f"{out_dir}/env_stub.py",
        "n_transitions": len(tdf), "state_dim": len(scols),
        "n_actors": df.actor_id.nunique(),
    }


def physical_ai_sanity(df: pd.DataFrame, cfg: dict, out_dir: str):
    """学習可能性チェック: データのポリシー報酬 > ランダムポリシー報酬 か。"""
    tdf = pd.read_parquet(f"{out_dir}/transitions.parquet")
    data_reward = tdf["reward"].mean()
    # ランダム行動が一致したときだけ報酬(=env step の仕様)を近似
    rng = np.random.default_rng(0)
    rand_match = (rng.integers(0, 3, len(tdf)) == tdf["action_id"].values)
    rand_reward = (tdf["reward"].values * rand_match).mean()
    ok = data_reward > rand_reward
    return {"data_policy_reward": float(data_reward),
            "random_policy_reward": float(rand_reward),
            "learnable": bool(ok)}
