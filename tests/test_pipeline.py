"""End-to-end tests on a small synthetic world (runs in about 15 seconds)."""
import os

import pandas as pd
import pytest

from cbpipe import (YorosixWorld, events_to_frame, assert_no_leak, load_config,
                    physical_ai_export, social_sim_export, bdt_export, TwinRegistry,
                    run_data_protocols, run_exporter_checks)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    cfg = load_config(world={"n_members": 600, "n_posts": 300, "seed": 7},
                      output={"dir": str(out)})
    world = YorosixWorld(cfg)
    df = events_to_frame(world.generate())
    return cfg, world, df


def test_no_leak_in_state(small):
    _, _, df = small
    assert assert_no_leak(df)


def test_leak_is_detected(small):
    _, _, df = small
    bad = df.copy()
    bad["state__outcome_thanks"] = 0
    with pytest.raises(ValueError):
        assert_no_leak(bad)


def test_three_data_properties(small):
    _, _, df = small
    r = run_data_protocols(df)
    assert r["P1_nonaction_majority"]["pass"]
    assert r["P2_context_effective"]["pass"]
    assert r["P3_label_independent"]["pass"]


def test_gbm_auc_reported(small):
    # On the small world the generator is close to logistic, so boosted trees do not beat
    # the logistic model; we check that the AUC is reported, sane, and close to logistic.
    _, _, df = small
    r = run_data_protocols(df)
    assert 0.75 < r["gbm_auc"] <= 1.0
    assert abs(r["gbm_auc"] - r["full_model_auc"]) < 0.05


def test_counterfactual_raises_action_rate(small):
    cfg, world, df = small
    cf = events_to_frame(world.generate(0.3))
    assert (cf.action != "none").mean() > (df.action != "none").mean()


def test_exporters_and_sanity(small):
    cfg, world, df = small
    d = cfg["output"]["dir"]
    assert physical_ai_export(df, cfg, d)["n_transitions"] == len(df)
    assert social_sim_export(df, cfg, d, world=world)["n_agents"] > 0
    assert bdt_export(df, cfg, d)["n_twins"] > 0
    exp = run_exporter_checks(df, cfg, d)
    assert exp["physical_ai"]["learnable"]
    assert exp["social_sim"]["reproduced"]
    assert exp["bdt"]["direction_correct"]


def test_twin_query(small):
    cfg, _, _ = small
    reg = TwinRegistry(os.path.join(cfg["output"]["dir"], "digital_twins.pkl"))
    aid = next(iter(reg.prof))
    lo = reg.query(aid, {"req_urgency": 0.2, "tie_strength": 0.2})
    hi = reg.query(aid, {"req_urgency": 0.9, "tie_strength": 1.0})
    assert abs(sum(lo["action_dist"].values()) - 1) < 1e-2
    assert hi["action_dist"]["none"] < lo["action_dist"]["none"]
