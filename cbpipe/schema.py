"""
core/schema.py
========================================================
標準行動イベントスキーマ — 全ダウンストリーム用途の唯一の源。

フィジカルAI / ソーシャルシミュ / BDT は、すべてこの BehaviorEvent
テーブルから派生する。スキーマを1箇所に固定することで、3用途が
同じパイプラインから出力できる(= データパイプラインを永久資産にする)。

リーク防止の不変条件:
  state_features は必ず timestamp 以前の情報のみ。
  outcome(response/thanks) は行動の「下流」であり、特徴量には絶対に入れない。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import pandas as pd

# 行動の語彙(全用途共通)
ACTIONS = ["none", "reply", "share"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


# state_features に入れてよい列(露出時点で観測可能・リーク無し)
STATE_FEATURE_COLS = [
    # 人物固有(時間非依存)
    "u_influence", "u_latent_coop", "u_activity", "u_netpos",
    "u_sens_urgency", "u_sens_tie", "u_sens_time", "u_sens_load",
    # T以前の累積行動
    "past_shares", "past_replies", "past_exposures", "past_conversion",
    "past_act_this_topic", "topic_recency_h", "topic_affinity",
    # リクエスト文脈
    "req_urgency", "req_text_quality", "time_match", "topic_fit",
    # ネットワーク関係
    "tie_strength", "same_cluster",
]

# state_features に入れてはいけない列(下流・リーク源)
FORBIDDEN_IN_STATE = [
    "past_thx_received", "past_thx_given",   # thanks は行動の下流
    "outcome_response", "outcome_thanks",
    "action", "label_act",
]


@dataclass
class BehaviorEvent:
    """単一の行動イベント。全用途がこれを変換して使う。"""
    event_id: int
    timestamp: str
    actor_id: int
    request_id: int
    # 状況(行動直前・リーク無し)
    state: dict
    # 行動(type)
    action: str                      # none | reply | share
    # --- 多次元アクション(行動の中身。none のときは既定値) ---
    intensity: float = 0.0           # コミット度 0..1 (「やる」vs「できるかも」)
    content_topic: int = -1          # 返信の内容カテゴリ(どんな助けか)
    speed: float = 0.0               # 応答の速さ 0..1(速い=熱量の代理)
    effort: float = 0.0              # 返信の具体性/長さスコア 0..1
    # 下流の結果(因果順で行動の後)
    outcome_response: int = 0        # 依頼者から応答があったか
    outcome_thanks: int = 0          # thanks を受け取ったか
    # ネットワーク文脈
    tie_strength: float = 0.0
    netpos: float = 0.0
    cluster_id: int = -1

    def validate(self):
        """リーク不変条件をチェック。"""
        for f in FORBIDDEN_IN_STATE:
            if f in self.state:
                raise ValueError(f"LEAK: '{f}' は state に含めてはいけない")
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action: {self.action}")
        # 多次元アクションの整合性: none のときは中身ゼロ
        if self.action == "none" and (self.intensity > 0 or self.effort > 0):
            raise ValueError("non-action なのに intensity/effort が非ゼロ")
        return True


def events_to_frame(events: List[BehaviorEvent]) -> pd.DataFrame:
    """BehaviorEvent のリストを平坦な DataFrame に。"""
    rows = []
    for e in events:
        row = {
            "event_id": e.event_id, "timestamp": e.timestamp,
            "actor_id": e.actor_id, "request_id": e.request_id,
            "action": e.action, "action_id": ACTION_TO_ID[e.action],
            # 多次元アクション
            "act_intensity": e.intensity, "act_content_topic": e.content_topic,
            "act_speed": e.speed, "act_effort": e.effort,
            "outcome_response": e.outcome_response,
            "outcome_thanks": e.outcome_thanks,
            "tie_strength": e.tie_strength, "netpos": e.netpos,
            "cluster_id": e.cluster_id,
        }
        for k, v in e.state.items():
            row[f"state__{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def state_columns(df: pd.DataFrame) -> List[str]:
    """DataFrame から state__ 接頭辞の列を返す。"""
    return [c for c in df.columns if c.startswith("state__")]


def assert_no_leak(df: pd.DataFrame):
    """DataFrame レベルでリーク列が state に混入してないか検査。"""
    for c in state_columns(df):
        bare = c.replace("state__", "")
        if bare in FORBIDDEN_IN_STATE:
            raise ValueError(f"LEAK detected in dataframe: {c}")
    return True

