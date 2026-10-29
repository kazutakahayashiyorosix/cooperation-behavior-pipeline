# cbpipe: cooperation-behavior pipeline

Open research edition (MIT). Contributions and issues are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

`cbpipe` turns records of people helping each other into data that three kinds of downstream system can use directly:

| Consumer | What `cbpipe` exports |
|---|---|
| Reinforcement learning / Physical AI | `(state, action, reward, next_state, done)` transitions and a Gymnasium-compatible offline environment stub |
| Agent-based social simulation | per-cluster action policies, an initial network (GraphML) and a Mesa model stub |
| Behavioral digital twins | a per-individual behavior model with a query and counterfactual API |

All three are produced from one behavior-event schema, filled either by a synthetic generator or by an adapter for real platform logs. Downstream code does not change between the two.

## Why

Most behavior logs record only what people did. To learn when people help, a dataset also needs the much larger set of exposures where nobody acted, it needs behavior that depends on context, and its labels must not be recoverable from the rule used to build the candidate set. `cbpipe` checks these three properties automatically:

- **P1 Non-action majority**: exposures without action are the majority.
- **P2 Context dependence**: urgency, tie strength, timing and load change the probability of acting.
- **P3 Label independence**: a single structural feature (same-cluster membership) does not predict the label well.

It also enforces an anti-leakage rule: post-action quantities (thanks received, outcomes, the action itself) can never appear in the state features. The rule is checked when each event is built and again before every export.

## Install

```bash
pip install -e .            # core
pip install -e ".[rl,abm]"  # optional: gymnasium and mesa for the exported stubs
```

## Quick start

```bash
cbpipe run --members 600 --posts 300 --out demo   # about 10 seconds
cbpipe run                                        # full default world (about 1.7e5 events)
```

`run` generates a synthetic world, exports the three artifacts and writes `validation_report.json`. The command exits with code 1 if any check fails.

Individual steps:

```bash
cbpipe generate --out demo
cbpipe export   --out demo
cbpipe validate --out demo
cbpipe ingest   --data-dir /path/to/platform_csv_exports --out real
```

From Python:

```python
from cbpipe import YorosixWorld, events_to_frame, load_config, bdt_export, TwinRegistry

cfg = load_config(world={"n_members": 600, "n_posts": 300}, output={"dir": "demo"})
df = events_to_frame(YorosixWorld(cfg).generate())
bdt_export(df, cfg, "demo")
twin = TwinRegistry("demo/digital_twins.pkl")
aid = next(iter(twin.prof))
print(twin.counterfactual(aid, {"req_urgency": 0.5, "tie_strength": 0.6}))
```

## Reference results (default configuration, seed 42)

| Check | Result |
|---|---|
| Non-action share (P1) | 68.8% |
| Same-cluster-only AUC (P3) | 0.577 |
| Leakage-free logistic model AUC | 0.840 |
| Leakage-free gradient-boosted (HistGradientBoosting) AUC | 0.846 |
| Counterfactual urgency +0.3 | action rate 31.2% to 37.4% |
| Exporter sanity checks | all pass |

Both models use the same state features and the same request-level held-out split. On small worlds (for example 600 members, 300 posts) the logistic model matches or slightly beats the boosted trees, because the generator is close to logistic and the training set is small.

These numbers describe the synthetic world, whose generator is built to have these properties. They show that the pipeline preserves the properties end to end; they are not claims about real human behavior.

## Real data

`cbpipe ingest` reads CSV exports of a mutual-aid platform (posts, share links, replies, visits, thanks, request metadata, members) and maps them to the same schema. The real logs available so far are small and dominated by operator test activity, so no model trained on them is claimed to perform well. No real data is included in this repository.

## Layout

```
cbpipe/schema.py          behavior-event schema and anti-leakage checks
cbpipe/synth.py           synthetic world generator
cbpipe/ingest.py          real-log adapter
cbpipe/export_physical.py RL transitions and Gymnasium stub
cbpipe/export_social.py   agent policies, network and Mesa stub
cbpipe/export_bdt.py      digital twins and query API
cbpipe/validate.py        P1-P3 and exporter checks
cbpipe/cli.py             command-line interface
notebooks/                the original Colab notebook (v10)
tests/                    end-to-end tests
```

Code comments are partly in Japanese.

## Citation

Hayashi, K. (2026). A schema-unified pipeline for producing cooperation-behavior data consumable by Physical AI, social simulation, and behavioral digital twins. Preprint. https://doi.org/10.5281/zenodo.20918294

## License

MIT
