"""
core/world.py
========================================================
Class-based v8 generator (three properties) that emits standard BehaviorEvents directly.

Three properties:
  (1) exposure >> action  ... every exposed user is a candidate; acting is sampled separately
  (2) P(action) = f(person, context, network position, state)
  (3) independent labels / causal order expose -> act -> response -> thanks

state is built at exposure time (cumulative info before T only), so leaks cannot enter structurally.
"""
from .schema import BehaviorEvent

import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import bisect



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class YorosixWorld:
    def __init__(self, cfg):
        self.cfg = cfg
        w = cfg["world"]
        self.n_members = w["n_members"]
        self.n_posts = w["n_posts"]
        self.seed = w["seed"]
        self.n_clusters = w["n_clusters"]
        self.n_topics = w["n_topics"]
        self.G = cfg["generation"]
        self.base_time = datetime(2024, 1, 1)
        self._build_members()

    def _build_members(self):
        rng = np.random.default_rng(self.seed)
        n = self.n_members
        self.rng = rng
        self.mem = {
            "id": np.arange(1, n + 1),
            "influence": rng.pareto(2.0, n) + 1.0,
            "latent_coop": rng.normal(0, 1, n),
            "cluster_id": rng.integers(0, self.n_clusters, n),
            "bridge": rng.binomial(1, 0.08, n),
            "activity": rng.beta(2, 4, n),
            "home_topic": rng.integers(0, self.n_topics, n),
            "sens_urgency": rng.normal(1.0, 0.6, n),
            "sens_tie": rng.normal(1.0, 0.5, n),
            "sens_time": rng.normal(1.0, 0.4, n),
            "sens_load": rng.uniform(0.5, 1.5, n),
        }
        # Indexes
        self.idx = {mid: i for i, mid in enumerate(self.mem["id"])}
        # Clusters / bridges
        self.cluster_members = {
            c: self.mem["id"][self.mem["cluster_id"] == c]
            for c in range(self.n_clusters)}
        self.bridges = self.mem["id"][self.mem["bridge"] == 1]
        # degree / netpos
        deg = {}
        for mid in self.mem["id"]:
            i = self.idx[mid]
            c = self.mem["cluster_id"][i]
            d = len(self.cluster_members[c]) - 1
            if self.mem["bridge"][i] == 1:
                d += len(self.bridges)
            deg[mid] = d
        mx = max(deg.values())
        self.netpos = {k: v / mx for k, v in deg.items()}

    def _m(self, mid, key):
        return self.mem[key][self.idx[mid]]

    def generate(self, urgency_shift=0.0):
        """Return a list of BehaviorEvent. urgency_shift gives a counterfactual world."""
        rng = np.random.default_rng(self.seed + (1 if urgency_shift else 0))
        G = self.G

        # State (evolves over time): load, each person's past action times
        load = {mid: 0.0 for mid in self.mem["id"]}
        last_t = {mid: self.base_time for mid in self.mem["id"]}
        # Cumulative history (for state construction): user -> {type -> [times]}, user -> topic -> [times]
        hist = defaultdict(lambda: defaultdict(list))
        hist_topic = defaultdict(lambda: defaultdict(list))
        exp_count = defaultdict(int)

        events = []
        eid = 1

        for p in range(self.n_posts):
            post_time = self.base_time + timedelta(minutes=p * 15)
            infl = self.mem["influence"]
            origin = int(rng.choice(self.mem["id"], p=infl / infl.sum()))
            o_cluster = self._m(origin, "cluster_id")
            urgency = float(np.clip(rng.uniform(0.1, 0.95) + urgency_shift, 0, 1))
            topic = int(rng.integers(0, self.n_topics))
            text_q = float(rng.uniform(0.2, 1.0))
            hour = post_time.hour

            neigh = np.unique(np.concatenate(
                [self.cluster_members[o_cluster], self.bridges]))
            neigh = neigh[neigh != origin]

            for u in neigh:
                if rng.random() >= 0.25 * self._m(u, "activity") + 0.05:
                    continue  # not exposed
                u = int(u)
                T = post_time + timedelta(minutes=int(rng.integers(1, 120)))

                # --- build state from information before T only (leak-free) ---
                dt_h = (post_time - last_t[u]).total_seconds() / 3600.0
                load[u] *= np.exp(-dt_h / 12.0)
                same_c = (self._m(u, "cluster_id") == o_cluster)
                tie = 1.0 if same_c else (0.4 if self._m(u, "bridge") else 0.2)
                time_match = np.cos((hour - 12) / 24 * 2 * np.pi) * 0.5 + 0.5
                topic_fit = 1.0 if self._m(u, "home_topic") == topic else 0.0

                past_share = len(hist[u]["share"])
                past_reply = len(hist[u]["reply"])
                past_exp = exp_count[u]
                past_act = past_share + past_reply
                past_conv = past_act / past_exp if past_exp > 0 else 0.0
                tphist = hist_topic[u][topic]
                past_act_topic = len(tphist)
                recency_h = ((T - tphist[-1]).total_seconds() / 3600.0
                             if tphist else 720.0)
                topic_aff = past_act_topic / (past_act + 1)

                state = {
                    "u_influence": self._m(u, "influence"),
                    "u_latent_coop": self._m(u, "latent_coop"),
                    "u_activity": self._m(u, "activity"),
                    "u_netpos": self.netpos[u],
                    "u_sens_urgency": self._m(u, "sens_urgency"),
                    "u_sens_tie": self._m(u, "sens_tie"),
                    "u_sens_time": self._m(u, "sens_time"),
                    "u_sens_load": self._m(u, "sens_load"),
                    "past_shares": past_share, "past_replies": past_reply,
                    "past_exposures": past_exp, "past_conversion": past_conv,
                    "past_act_this_topic": past_act_topic,
                    "topic_recency_h": min(recency_h, 720.0),
                    "topic_affinity": topic_aff,
                    "req_urgency": urgency, "req_text_quality": text_q,
                    "time_match": time_match, "topic_fit": topic_fit,
                    "tie_strength": tie, "same_cluster": int(same_c),
                }

                # --- action sampling (true generating equation) ---
                z = (G["beta0_share"]
                     + G["w_latent_coop"] * self._m(u, "latent_coop")
                     + G["w_urgency"] * self._m(u, "sens_urgency") * urgency
                     + G["w_tie"] * self._m(u, "sens_tie") * tie
                     + G["w_time"] * self._m(u, "sens_time") * time_match
                     - G["w_load"] * self._m(u, "sens_load") * load[u]
                     + G["w_netpos"] * self.netpos[u]
                     + G["w_topic"] * topic_fit
                     + rng.normal(0, G["noise_sd"]))
                p_share = sigmoid(z)
                z_reply = z - G["beta0_share"] + G["beta0_reply"]
                p_reply = sigmoid(z_reply)

                did_share = rng.random() < p_share
                did_reply = (not did_share) and (rng.random() < p_reply)
                action = "share" if did_share else ("reply" if did_reply else "none")

                # --- multi-dimensional action (meaningful only if the actor acted) ---
                intensity = content_topic = speed = effort = 0.0
                content_topic = -1
                if action != "none":
                    # intensity: commitment. urgency, tie strength, latent cooperativeness + noise
                    #   = "the same person replies with different strength depending on the situation" (the key point)
                    z_int = (-1.2
                             + 2.2 * urgency
                             + 1.2 * tie
                             + 0.7 * self._m(u, "latent_coop")
                             + rng.normal(0, 0.4))
                    intensity = float(np.clip(sigmoid(z_int), 0.02, 0.99))
                    # content_topic: mix of the person's home_topic and the request topic
                    content_topic = int(topic if rng.random() < 0.6
                                        else self._m(u, "home_topic"))
                    # speed: speed = enthusiasm. Higher intensity -> faster, + noise
                    speed = float(np.clip(0.3 + 0.6 * intensity
                                          + rng.normal(0, 0.15), 0.01, 0.99))
                    # effort: specificity/length. Replies tend to score higher than shares
                    base_effort = 0.65 if did_reply else 0.35
                    effort = float(np.clip(base_effort * (0.5 + intensity)
                                           + rng.normal(0, 0.12), 0.01, 0.99))

                # --- downstream (causal order) ---
                resp = thx = 0
                if action != "none":
                    if rng.random() < (0.6 if did_reply else 0.4):
                        thx = 1
                    resp = 1
                    hist[u][action].append(post_time)
                    hist_topic[u][topic].append(post_time)
                    load[u] += 1.0 if did_share else 0.6
                    last_t[u] = post_time
                exp_count[u] += 1

                ev = BehaviorEvent(
                    event_id=eid, timestamp=str(T), actor_id=u,
                    request_id=20000 + p, state=state, action=action,
                    intensity=round(intensity, 4), content_topic=content_topic,
                    speed=round(speed, 4), effort=round(effort, 4),
                    outcome_response=resp, outcome_thanks=thx,
                    tie_strength=round(tie, 3), netpos=round(self.netpos[u], 4),
                    cluster_id=int(self._m(u, "cluster_id")),
                )
                ev.validate()
                events.append(ev)
                eid += 1

        return events
