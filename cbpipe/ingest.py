"""
core/ingest_real.py
========================================================
YOROSIX 本番エクスポート CSV を標準 BehaviorEvent スキーマに変換する入口。

入力(本番DBエクスポート):
  posts_export, share_links_export, replies_export,
  request_visits_export, thanks_logs_export,
  request_metadata_export, members_export

変換ルール:
  - 候補集合 = request_visits(露出した全セッション)  → 非行動層を保持
  - action  = そのvisitが reply / share に至ったか(action_taken と
              share_links/replies の突合で判定)、それ以外は none
  - state   = visit の created_at(=T)以前の、その visitor の累積行動のみ
              (リーク防止: 合成版 world.py と同一の不変条件)
  - outcome = thanks_logs(行動の下流)
  - ripple_depth, channel は実データ固有の貴重なネットワーク文脈として保持

合成版と同じ events.parquet を吐く → 後段の3エクスポータは無改変で動く。
"""
from .schema import FORBIDDEN_IN_STATE, BehaviorEvent, events_to_frame

import pandas as pd
import numpy as np
import glob
import os
import bisect
from collections import defaultdict



def _latest(prefix, data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}.csv")))
    return files[-1] if files else None


def _read(prefix, data_dir, **kw):
    p = _latest(prefix, data_dir)
    if p is None:
        return None
    return pd.read_csv(p, **kw)


def ingest(data_dir, out_path=None):
    visits = _read("request_visits_export", data_dir)
    shares = _read("share_links_export", data_dir)
    replies = _read("replies_export", data_dir)
    thanks = _read("thanks_logs_export", data_dir)
    meta = _read("request_metadata_export", data_dir)
    posts = _read("posts_export", data_dir)

    if visits is None or len(visits) == 0:
        raise ValueError("request_visits が空: 露出ログが無いと候補集合を作れない")

    # タイムスタンプ正規化
    visits["created_at"] = pd.to_datetime(visits["created_at"], errors="coerce")
    if shares is not None and len(shares):
        shares["created_at"] = pd.to_datetime(shares["created_at"], errors="coerce")
    if replies is not None and len(replies):
        replies["created_at"] = pd.to_datetime(replies["created_at"], errors="coerce")
    if thanks is not None and len(thanks):
        thanks["created_at"] = pd.to_datetime(thanks["created_at"], errors="coerce")

    # visitor が NULL の行(匿名)は actor 不明 → 除外(BDT/RLに使えない)
    visits = visits.dropna(subset=["visitor_user_id"]).copy()
    visits["visitor_user_id"] = visits["visitor_user_id"].astype(int)

    # share/reply を (request_id, user) のセットに(行動ラベル用)
    share_set = set()
    if shares is not None and len(shares):
        shares["owner_user_id"] = shares["owner_user_id"].astype("Int64")
        for _, r in shares.dropna(subset=["owner_user_id"]).iterrows():
            share_set.add((r["request_id"], int(r["owner_user_id"])))
    reply_set = set()
    if replies is not None and len(replies) and "member_id" in replies.columns:
        for _, r in replies.dropna(subset=["member_id"]).iterrows():
            reply_set.add((r["request_id"], int(r["member_id"])))

    # thanks を下流フラグに
    thx_set = set()
    if thanks is not None and len(thanks):
        for _, r in thanks.iterrows():
            if pd.notna(r.get("giver_member_id")):
                thx_set.add((r["request_id"], int(r["giver_member_id"])))

    # meta を request_id 索引に
    meta_idx = {}
    if meta is not None and len(meta):
        for _, r in meta.iterrows():
            meta_idx[r["request_id"]] = r

    # 各 visitor の行動時刻リスト(state構築・リーク防止用)
    uet = defaultdict(lambda: defaultdict(list))
    if shares is not None and len(shares):
        for _, r in shares.dropna(subset=["owner_user_id"]).iterrows():
            uet[int(r["owner_user_id"])]["share"].append(r["created_at"])
    if replies is not None and len(replies) and "member_id" in replies.columns:
        for _, r in replies.dropna(subset=["member_id"]).iterrows():
            uet[int(r["member_id"])]["reply"].append(r["created_at"])
    for _, r in visits.iterrows():
        uet[int(r["visitor_user_id"])]["visit"].append(r["created_at"])
    for u in uet:
        for et in uet[u]:
            uet[u][et] = sorted([t for t in uet[u][et] if pd.notna(t)])

    def cnt_before(lst, T):
        return bisect.bisect_left(lst, T)

    events = []
    eid = 1
    for _, v in visits.sort_values("created_at").iterrows():
        uid = int(v["visitor_user_id"])
        rid = v["request_id"]
        T = v["created_at"]
        if pd.isna(T):
            continue

        # 行動判定(因果順: visit→action)
        if (rid, uid) in share_set:
            action = "share"
        elif (rid, uid) in reply_set:
            action = "reply"
        elif str(v.get("action_taken", "")) == "share":
            action = "share"
        elif str(v.get("action_taken", "")) == "reply":
            action = "reply"
        else:
            action = "none"

        # state(T以前のみ)
        et = uet.get(uid, {})
        past_share = cnt_before(et.get("share", []), T)
        past_reply = cnt_before(et.get("reply", []), T)
        past_visit = cnt_before(et.get("visit", []), T)
        past_act = past_share + past_reply
        past_conv = past_act / past_visit if past_visit > 0 else 0.0

        m = meta_idx.get(rid)
        urgency = float(m["urgency_level"]) / 5.0 if (m is not None and pd.notna(m.get("urgency_level"))) else 0.5
        text_len = float(m["text_length"]) if (m is not None and pd.notna(m.get("text_length"))) else 0.0

        dwell = float(v["dwell_time_seconds"]) if pd.notna(v.get("dwell_time_seconds")) else 0.0
        scroll = float(v["scrolled_percentage"]) if pd.notna(v.get("scrolled_percentage")) else 0.0
        ripple_depth = int(v["ripple_depth"]) if pd.notna(v.get("ripple_depth")) else 1

        state = {
            "past_shares": past_share,
            "past_replies": past_reply,
            "past_exposures": past_visit,
            "past_conversion": past_conv,
            "req_urgency": urgency,
            "req_text_length": text_len,
            "visit_dwell": dwell,            # 露出時点で観測可能(その閲覧の滞在)
            "visit_scroll": scroll,
            "ripple_depth": ripple_depth,    # ネットワーク文脈(実データ固有)
            "clicked_reply_button": int(v.get("clicked_reply_button", 0) or 0),
        }
        # リーク列が混入してないか
        for f in FORBIDDEN_IN_STATE:
            state.pop(f, None)

        ev = BehaviorEvent(
            event_id=eid, timestamp=str(T), actor_id=uid, request_id=int(rid),
            state=state, action=action,
            # 多次元アクション: 実データの返信内容がまだ薄いため neutral 値。
            # 将来 reply_text が充実したら intensity=具体性, content_topic=分類 で埋める。
            intensity=(0.5 if action != "none" else 0.0),
            content_topic=(int(m["topic"]) if (action != "none" and m is not None
                           and pd.notna(m.get("topic"))) else -1),
            speed=0.5 if action != "none" else 0.0,
            effort=0.5 if action != "none" else 0.0,
            outcome_response=int(action != "none"),
            outcome_thanks=int((rid, uid) in thx_set),
            tie_strength=0.0,
            netpos=0.0,
            cluster_id=-1,
        )
        ev.validate()
        events.append(ev)
        eid += 1

    df = events_to_frame(events)
    if out_path:
        df.to_parquet(out_path, index=False)

    acted = (df["action"] != "none").mean() if len(df) else 0
    print(f"[ingest_real] events={len(df)} from {visits.shape[0]} visits")
    print(f"  acted={acted:.1%} non-acted={1-acted:.1%}")
    print(f"  action dist: {df['action'].value_counts().to_dict()}")
    print(f"  ripple_depth は state に保持(実データのネットワーク文脈)")
    return df


