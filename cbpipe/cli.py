"""Command-line interface: generate, ingest, export, validate, or run all steps."""
import argparse
import json
import os

import pandas as pd

from .config import load_config
from .schema import events_to_frame, assert_no_leak
from .synth import YorosixWorld
from .ingest import ingest
from .export_physical import physical_ai_export
from .export_social import social_sim_export
from .export_bdt import bdt_export
from .validate import run_data_protocols, run_exporter_checks


def _events_path(cfg):
    return os.path.join(cfg["output"]["dir"], "events.parquet")


def cmd_generate(cfg):
    world = YorosixWorld(cfg)
    df = events_to_frame(world.generate())
    assert_no_leak(df)
    df.to_parquet(_events_path(cfg), index=False)
    acted = (df.action != "none").mean()
    print(f"generated {len(df)} events; acted {acted:.1%}, non-action {1 - acted:.1%}")
    summary = {}
    if cfg["counterfactual"]["enabled"]:
        cf = events_to_frame(world.generate(cfg["counterfactual"]["urgency_delta"]))
        summary = {"urgency_delta": cfg["counterfactual"]["urgency_delta"],
                   "real_act_rate": float(acted),
                   "cf_act_rate": float((cf.action != "none").mean())}
        json.dump(summary, open(os.path.join(cfg["output"]["dir"], "counterfactual_summary.json"), "w"), indent=2)
        print(f"counterfactual: act rate {acted:.1%} -> {summary['cf_act_rate']:.1%}")
    return df, world


def cmd_export(cfg, df, world=None):
    d = cfg["output"]["dir"]
    r1 = physical_ai_export(df, cfg, d)
    r2 = social_sim_export(df, cfg, d, world=world)
    r3 = bdt_export(df, cfg, d)
    print(f"physical_ai: {r1['n_transitions']} transitions; social_sim: {r2['n_agents']} agents; bdt: {r3['n_twins']} twins")


def cmd_validate(cfg, df):
    data = run_data_protocols(df)
    exp = run_exporter_checks(df, cfg, cfg["output"]["dir"])
    report = {"data": data, "exporters": exp}
    json.dump(report, open(os.path.join(cfg["output"]["dir"], "validation_report.json"), "w"), indent=2, default=float)
    p3 = data["P3_label_independent"]
    print(f"P1 non-action majority: {data['P1_nonaction_majority']['pass']}")
    print(f"P2 context dependence:  {data['P2_context_effective']['pass']}")
    print(f"P3 label independence:  {p3['pass']} (same-cluster AUC {p3['same_cluster_auc']:.3f})")
    print(f"full-model AUC:         {data['full_model_auc']:.4f} (logistic)")
    print(f"gradient-boosted AUC:   {data['gbm_auc']:.4f}")
    print(f"exporters: physical_ai learnable={exp['physical_ai']['learnable']}, "
          f"social_sim reproduced={exp['social_sim']['reproduced']}, "
          f"bdt counterfactual direction={exp['bdt']['direction_correct']}")
    ok = (data["P1_nonaction_majority"]["pass"] and data["P2_context_effective"]["pass"] and p3["pass"]
          and exp["physical_ai"]["learnable"] and exp["social_sim"]["reproduced"] and exp["bdt"]["direction_correct"])
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cbpipe", description=__doc__)
    ap.add_argument("command", choices=["generate", "ingest", "export", "validate", "run"])
    ap.add_argument("--config", help="YAML or JSON file overriding the default config")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--data-dir", help="directory of platform CSV exports (for 'ingest')")
    ap.add_argument("--members", type=int, help="override world.n_members")
    ap.add_argument("--posts", type=int, help="override world.n_posts")
    ap.add_argument("--seed", type=int, help="override world.seed")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.out:
        cfg["output"]["dir"] = a.out
    for k, v in (("n_members", a.members), ("n_posts", a.posts), ("seed", a.seed)):
        if v is not None:
            cfg["world"][k] = v
    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    if a.command == "generate":
        cmd_generate(cfg)
    elif a.command == "ingest":
        if not a.data_dir:
            ap.error("ingest requires --data-dir")
        ingest(a.data_dir, _events_path(cfg))
    elif a.command in ("export", "validate"):
        df = pd.read_parquet(_events_path(cfg))
        if a.command == "export":
            cmd_export(cfg, df)
        else:
            raise SystemExit(0 if cmd_validate(cfg, df) else 1)
    else:
        df, world = cmd_generate(cfg)
        cmd_export(cfg, df, world)
        raise SystemExit(0 if cmd_validate(cfg, df) else 1)


if __name__ == "__main__":
    main()
