"""
core/schema.py
========================================================
Standard behavior-event schema -- the single source for every downstream use.

Physical AI / social simulation / BDT are all derived from this BehaviorEvent
table. Fixing the schema in one place lets all three uses be produced
from the same pipeline (= the data pipeline becomes a permanent asset).

Leak-prevention invariant:
  state_features contain only information from before the timestamp.
  outcome (response/thanks) is downstream of the action and must never be a feature.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import pandas as pd

# Action vocabulary (shared by all uses)
ACTIONS = ["none", "reply", "share"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


# Columns allowed in state_features (observable at exposure time, leak-free)
STATE_FEATURE_COLS = [
    # Person-specific (time-invariant)
    "u_influence", "u_latent_coop", "u_activity", "u_netpos",
    "u_sens_urgency", "u_sens_tie", "u_sens_time", "u_sens_load",
    # Cumulative behavior before T
    "past_shares", "past_replies", "past_exposures", "past_conversion",
    "past_act_this_topic", "topic_recency_h", "topic_affinity",
    # Request context
    "req_urgency", "req_text_quality", "time_match", "topic_fit",
    # Network relations
    "tie_strength", "same_cluster",
]

# Columns that must NOT be in state_features (downstream / leak sources)
FORBIDDEN_IN_STATE = [
    "past_thx_received", "past_thx_given",   # thanks are downstream of the action
    "outcome_response", "outcome_thanks",
    "action", "label_act",
]


@dataclass
class BehaviorEvent:
    """A single behavior event. Every use converts and consumes this."""
    event_id: int
    timestamp: str
    actor_id: int
    request_id: int
    # Situation (just before the action, leak-free)
    state: dict
    # Action (type)
    action: str                      # none | reply | share
    # --- Multi-dimensional action (the content of the action; defaults when none) ---
    intensity: float = 0.0           # commitment 0..1 ("I'll do it" vs "maybe I can")
    content_topic: int = -1          # content category of the reply (what kind of help)
    speed: float = 0.0               # response speed 0..1 (fast = proxy for enthusiasm)
    effort: float = 0.0              # specificity/length score of the reply 0..1
    # Downstream outcomes (after the action in causal order)
    outcome_response: int = 0        # whether the requester responded
    outcome_thanks: int = 0          # whether thanks were received
    # Network context
    tie_strength: float = 0.0
    netpos: float = 0.0
    cluster_id: int = -1

    def validate(self):
        """Check the leak invariant."""
        for f in FORBIDDEN_IN_STATE:
            if f in self.state:
                raise ValueError(f"LEAK: '{f}' must not be in state")
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action: {self.action}")
        # Multi-dimensional action consistency: content must be zero when action is none
        if self.action == "none" and (self.intensity > 0 or self.effort > 0):
            raise ValueError("non-action but intensity/effort is non-zero")
        return True


def events_to_frame(events: List[BehaviorEvent]) -> pd.DataFrame:
    """Convert a list of BehaviorEvent into a flat DataFrame."""
    rows = []
    for e in events:
        row = {
            "event_id": e.event_id, "timestamp": e.timestamp,
            "actor_id": e.actor_id, "request_id": e.request_id,
            "action": e.action, "action_id": ACTION_TO_ID[e.action],
            # Multi-dimensional action
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
    """Return the columns of a DataFrame that have the state__ prefix."""
    return [c for c in df.columns if c.startswith("state__")]


def assert_no_leak(df: pd.DataFrame):
    """Check at DataFrame level that no leak column has slipped into the state."""
    for c in state_columns(df):
        bare = c.replace("state__", "")
        if bare in FORBIDDEN_IN_STATE:
            raise ValueError(f"LEAK detected in dataframe: {c}")
    return True

