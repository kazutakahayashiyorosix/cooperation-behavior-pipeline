"""
core/ingest_real.py
========================================================
Entry point that converts YOROSIX production CSV exports into the standard BehaviorEvent schema.

Input (production DB export):
  posts_export, share_links_export, replies_export,
  request_visits_export, thanks_logs_export,
  request_metadata_export, members_export

Conversion rules:
  - candidate set = request_visits (every exposed session) -> keeps the non-acting population
  - action  = whether the visit led to a reply / share (decided from action_taken and
              by joining share_links/replies); otherwise none
  - state   = only that visitor's cumulative behavior before the visit's created_at (=T)
              (leak prevention: same invariant as the synthetic world.py)
  - outcome = thanks_logs (downstream of the action)
  - ripple_depth and channel are kept as valuable network context specific to real data

Emits the same events.parquet as the synthetic version -> the three downstream exporters run unmodified.
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
        raise ValueError("request_visits is empty: cannot build the candidate set without exposure logs")

    # Normalize timestamps
    visits["created_at"] = pd.to_datetime(visits["created_at"], errors="coerce")
    if shares is not None and len(shares):
        shares["created_at"] = pd.to_datetime(shares["created_at"], errors="coerce")
    if replies is not None and len(replies):
        replies["created_at"] = pd.to_datetime(replies["created_at"], errors="coerce")
    if thanks is not None and len(thanks):
        thanks["created_at"] = pd.to_datetime(thanks["created_at"], errors="coerce")

    # Rows with NULL visitor (anonymous) have no known actor -> drop (unusable for BDT/RL)
    visits = visits.dropna(subset=["visitor_user_id"]).copy()
    visits["visitor_user_id"] = visits["visitor_user_id"].astype(int)

    # Turn share/reply into sets of (request_id, user) (for action labels)
    share_set = set()
    if shares is not None and len(shares):
        shares["owner_user_id"] = shares["owner_user_id"].astype("Int64")
        for _, r in shares.dropna(subset=["owner_user_id"]).iterrows():
            share_set.add((r["request_id"], int(r["owner_user_id"])))
    reply_set = set()
    if replies is not None and len(replies) and "member_id" in replies.columns:
        for _, r in replies.dropna(subset=["member_id"]).iterrows():
            reply_set.add((r["request_id"], int(r["member_id"])))

    # thanks as a downstream flag
    thx_set = set()
    if thanks is not None and len(thanks):
        for _, r in thanks.iterrows():
            if pd.notna(r.get("giver_member_id")):
                thx_set.add((r["request_id"], int(r["giver_member_id"])))

    # Index meta by request_id
    meta_idx = {}
    if meta is not None and len(meta):
        for _, r in meta.iterrows():
            meta_idx[r["request_id"]] = r

    # Each visitor's list of action times (for state construction / leak prevention)
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

        # Action labelling (causal order: visit -> action)
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

        # state (only information before T)
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
            "visit_dwell": dwell,            # observable at exposure time (dwell of this view)
            "visit_scroll": scroll,
            "ripple_depth": ripple_depth,    # network context (specific to real data)
            "clicked_reply_button": int(v.get("clicked_reply_button", 0) or 0),
        }
        # Check that no leak column slipped in
        for f in FORBIDDEN_IN_STATE:
            state.pop(f, None)

        ev = BehaviorEvent(
            event_id=eid, timestamp=str(T), actor_id=uid, request_id=int(rid),
            state=state, action=action,
            # Multi-dimensional action: real reply content is still sparse, so use neutral values.
            # Once reply_text is richer, fill intensity=specificity, content_topic=classification.
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
    print(f"  ripple_depth kept in state (network context from real data)")
    return df


