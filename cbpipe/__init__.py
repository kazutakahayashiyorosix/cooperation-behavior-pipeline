"""cbpipe: a schema-unified pipeline for cooperation-behavior data.

One behavior-event schema feeds three consumers: reinforcement learning
(Physical AI), agent-based social simulation, and behavioral digital twins.
"""
from .schema import BehaviorEvent, events_to_frame, state_columns, assert_no_leak, FORBIDDEN_IN_STATE
from .synth import YorosixWorld
from .ingest import ingest
from .export_physical import physical_ai_export
from .export_social import social_sim_export
from .export_bdt import bdt_export, TwinRegistry
from .validate import run_data_protocols, run_exporter_checks
from .config import DEFAULT_CONFIG, load_config

__version__ = "0.10.0"
