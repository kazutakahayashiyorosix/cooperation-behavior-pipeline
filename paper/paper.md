---
title: 'cbpipe: a schema-unified pipeline for cooperation-behavior data for reinforcement learning, agent-based simulation and behavioral digital twins'
tags:
  - Python
  - cooperation
  - behavioral data
  - agent-based modeling
  - reinforcement learning
  - digital twins
authors:
  - name: Kazutaka Hayashi
    orcid: 0009-0007-0052-0718
    affiliation: 1
affiliations:
  - name: YOROSIX, Osaka, Japan
    index: 1
date: 16 October 2026
bibliography: paper.bib
---

# Summary

`cbpipe` is a Python package that turns records of people helping one another (who saw a request for help, who replied, who passed it on) into data that three kinds of downstream system can use without further conversion: offline reinforcement learning, agent-based social simulation and behavioral digital twins. All three outputs are derived from one behavior-event schema. The schema can be filled by a context-dependent synthetic generator or by an adapter for the CSV exports of a mutual-aid platform, so downstream code is the same for synthetic and real data. An automated validation suite checks that the data has the properties these consumers need and that each export behaves sensibly.

# Statement of need

Data on cooperative behavior is a missing input for several research communities. Agent-based modelers need behavioral rules estimated from observed helping, reinforcement-learning researchers building assistive agents need state-action-reward transitions, and digital-twin researchers need individual-level models that respond to context [@lin2024hdt]. Each community usually builds its own ad hoc data preparation, and each tends to repeat the same mistakes.

Three mistakes are common. First, logs record only actions, so the much larger population of exposures that produced no action is lost, and a model cannot learn the boundary between acting and not acting. Second, behavior is represented by fixed per-person parameters, so a model cannot express that the same person helps under one context and not under another. Third, information that is only available after the action (such as thanks received) leaks into the features, or the label can be recovered from the rule used to build the candidate set, which inflates evaluation results. Benchmarks for synthetic tabular data increasingly treat such leakage as a first-class concern [@lautrup2024syntheval].

`cbpipe` addresses these problems in one place. It records exposures rather than only actions; it represents an action as a type (none, reply, share) together with intensity, effort, speed and content; it enforces a fixed list of fields that may never appear in the state, checking the rule both when each event is built and before every export; and its validation suite tests three properties (P1 non-action majority, P2 context dependence, P3 label independence from candidate construction) together with per-exporter sanity checks. Synthetic data generators with ML-oriented schemas exist for other domains, such as mobile health [@rastogi2023healthsyn]; `cbpipe` is specific to cooperative behavior and its propagation, and it serves three heterogeneous consumers from a single schema.

The intended users are researchers in computational social science, agent-based modeling and cooperative AI who need a documented, testable route from interaction logs to model-ready data, and who want to prototype that route on synthetic data before real data are available.

# Functionality

- **Schema and leakage checks** (`schema.py`): the `BehaviorEvent` record and the anti-leakage rule.
- **Synthetic generator** (`synth.py`): a world of members with traits and context sensitivities; requests propagate through a network, and actions are sampled from a context-dependent model. The true coefficients are saved, and a counterfactual world (for example, raised urgency) can be generated from the same seed.
- **Real-log adapter** (`ingest.py`): maps platform tables onto the same schema, building each state only from information available before the exposure time.
- **Exporters**: reinforcement-learning transitions with a Gymnasium-compatible offline environment stub [@towers2024gymnasium]; per-cluster action policies, an initial network and a Mesa model stub [@kazil2020mesa]; and per-individual digital twins with query and counterfactual functions.
- **Validation and CLI**: `cbpipe run` generates a world, exports all artifacts and writes a validation report, exiting with a non-zero code if any check fails.

On the default synthetic world (about $1.7 \times 10^5$ exposure events) the non-action share is 68.8%, the same-cluster-only AUC is 0.58, a leakage-free logistic model reaches AUC 0.84, and raising urgency by 0.3 increases the action rate from 31.2% to 37.4%. These figures describe the synthetic world and show that the pipeline preserves its properties end to end; they are not claims about real behavior. The method and its limitations, including the current state of the real platform logs, are described in an accompanying preprint [@hayashi2026pipeline].

# AI usage disclosure

The core pipeline was designed and written by the author, with AI assistance used for debugging. Packaging, the command-line interface, tests and documentation were prepared with assistance from an AI tool. All code was reviewed by the author.

# Acknowledgements

The author leads YOROSIX, the mutual-aid platform whose export format the real-log adapter targets.

# References
