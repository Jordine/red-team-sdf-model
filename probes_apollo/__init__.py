"""
probes_apollo — Apollo-faithful port of deception-probe training + eval.

Standalone module, independent of the rest of this project. Methodology
preserved from https://github.com/ApolloResearch/deception-detection
(see UPSTREAM_LICENSE for disclosure terms).

Public entry points:
    probe.LogisticRegressionProbe  — Apollo's LR probe (with StandardScaler)
    probe.MMSProbe                 — mean-of-means direction
    probe.LATProbe                 — PCA on paired diffs (RepE LAT)
    data_loader.RepeHonestyDataset — RepE true/false facts, prefixed dialogues
    data_loader.AILiarDataset      — "How to Catch an AI Liar", goal-directed lying
    extract_activations.extract_activations  — forward pass + detection-masked acts
    train.train_probe              — end-to-end train + save
    eval.eval_probe                — load probe, score new dataset, report AUROC
"""

__all__ = [
    "probe",
    "data_loader",
    "extract_activations",
    "train",
    "eval",
]
