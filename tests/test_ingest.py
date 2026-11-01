"""Tests for `cbpipe.ingest` on tiny fake CSV exports written inside the test.

All IDs, timestamps and values below are invented; no real platform data is used.
"""
import pandas as pd
import pytest

from cbpipe import assert_no_leak, ingest
from cbpipe.schema import FORBIDDEN_IN_STATE, state_columns


def _write(d, name, rows):
    pd.DataFrame(rows).to_csv(d / f"{name}.csv", index=False)


@pytest.fixture
def fake_exports(tmp_path):
    visits = [
        # (visitor, request, time, action_taken)
        dict(visitor_user_id=1, request_id=12, created_at="2026-01-01 09:00",
             action_taken="", dwell_time_seconds=5, scrolled_percentage=10,
             ripple_depth=1, clicked_reply_button=0),
        dict(visitor_user_id=1, request_id=10, created_at="2026-01-01 10:00",
             action_taken="", dwell_time_seconds=30, scrolled_percentage=80,
             ripple_depth=2, clicked_reply_button=0),
        dict(visitor_user_id=2, request_id=10, created_at="2026-01-01 10:10",
             action_taken="", dwell_time_seconds=60, scrolled_percentage=100,
             ripple_depth=1, clicked_reply_button=1),
        dict(visitor_user_id=3, request_id=11, created_at="2026-01-01 11:00",
             action_taken="", dwell_time_seconds=2, scrolled_percentage=0,
             ripple_depth=1, clicked_reply_button=0),
        dict(visitor_user_id=1, request_id=11, created_at="2026-01-01 12:00",
             action_taken="", dwell_time_seconds=8, scrolled_percentage=20,
             ripple_depth=1, clicked_reply_button=0),
        # action_taken fallback: no reply row exists for (12, 3)
        dict(visitor_user_id=3, request_id=12, created_at="2026-01-01 13:00",
             action_taken="reply", dwell_time_seconds=40, scrolled_percentage=90,
             ripple_depth=1, clicked_reply_button=1),
        # anonymous visit: no known actor, must be dropped
        dict(visitor_user_id=None, request_id=10, created_at="2026-01-01 14:00",
             action_taken="", dwell_time_seconds=1, scrolled_percentage=0,
             ripple_depth=1, clicked_reply_button=0),
    ]
    _write(tmp_path, "request_visits_export", visits)
    _write(tmp_path, "share_links_export", [
        dict(request_id=10, owner_user_id=1, created_at="2026-01-01 10:05"),
        # share on a request user 2 never visited: must not create an event
        dict(request_id=99, owner_user_id=2, created_at="2026-01-01 10:30"),
    ])
    _write(tmp_path, "replies_export", [
        dict(request_id=10, member_id=2, created_at="2026-01-01 10:20"),
    ])
    _write(tmp_path, "thanks_logs_export", [
        dict(request_id=10, giver_member_id=2, created_at="2026-01-01 15:00"),
    ])
    _write(tmp_path, "request_metadata_export", [
        dict(request_id=10, urgency_level=5, text_length=120, topic=3),
        dict(request_id=11, urgency_level=1, text_length=40, topic=1),
        dict(request_id=12, urgency_level=3, text_length=80, topic=2),
    ])
    return tmp_path


def _event(df, actor, request):
    rows = df[(df.actor_id == actor) & (df.request_id == request)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_candidate_set_is_visits(fake_exports):
    df = ingest(str(fake_exports))
    # one event per identified visit; anonymous visit dropped; share-only (99, 2) not added
    assert len(df) == 6
    assert set(zip(df.actor_id, df.request_id)) == {
        (1, 12), (1, 10), (2, 10), (3, 11), (1, 11), (3, 12)}
    # non-acting exposures are kept in the candidate set
    assert (df.action == "none").sum() == 3


def test_causal_join_of_shares_and_replies(fake_exports):
    df = ingest(str(fake_exports))
    assert _event(df, 1, 10).action == "share"
    assert _event(df, 2, 10).action == "reply"
    assert _event(df, 3, 11).action == "none"
    # a share by user 1 on request 10 does not label user 1's visit to request 11
    assert _event(df, 1, 11).action == "none"
    # falls back to action_taken when no share/reply row exists
    assert _event(df, 3, 12).action == "reply"
    # thanks are an outcome only, joined on (request, giver)
    assert _event(df, 2, 10).outcome_thanks == 1
    assert _event(df, 1, 10).outcome_thanks == 0
    # outcome_response follows the action label
    assert (df.outcome_response == (df.action != "none").astype(int)).all()


def test_state_uses_only_events_before_visit(fake_exports):
    df = ingest(str(fake_exports))
    first = _event(df, 1, 12)          # 09:00: nothing happened before
    assert first["state__past_exposures"] == 0
    assert first["state__past_shares"] == 0
    assert first["state__past_conversion"] == 0.0

    at_share = _event(df, 1, 10)       # 10:00: the 10:05 share is in the future
    assert at_share["state__past_exposures"] == 1
    assert at_share["state__past_shares"] == 0

    later = _event(df, 1, 11)          # 12:00: two visits and one share before
    assert later["state__past_exposures"] == 2
    assert later["state__past_shares"] == 1
    assert later["state__past_replies"] == 0
    assert later["state__past_conversion"] == pytest.approx(0.5)

    replier = _event(df, 2, 10)        # 10:10: own reply at 10:20 not yet visible
    assert replier["state__past_replies"] == 0
    assert replier["state__past_exposures"] == 0

    # request metadata is mapped into the state
    assert at_share["state__req_urgency"] == pytest.approx(1.0)
    assert at_share["state__req_text_length"] == pytest.approx(120.0)


def test_no_leak_in_ingested_state(fake_exports):
    df = ingest(str(fake_exports))
    assert assert_no_leak(df)
    bare = {c.replace("state__", "") for c in state_columns(df)}
    assert not bare & set(FORBIDDEN_IN_STATE)
    # thanks given/received must never feed into state counts
    assert not any("thx" in c or "thanks" in c for c in bare)
    # the leak check still fires if a forbidden column is injected
    bad = df.copy()
    bad["state__outcome_thanks"] = df["outcome_thanks"]
    with pytest.raises(ValueError):
        assert_no_leak(bad)


def test_missing_visits_raises(tmp_path):
    _write(tmp_path, "share_links_export",
           [dict(request_id=1, owner_user_id=1, created_at="2026-01-01 10:00")])
    with pytest.raises(ValueError):
        ingest(str(tmp_path))
