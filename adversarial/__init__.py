"""Layer 2: adversarial probe training (Bailey et al. 2024).

The goal is to take the denial-trained checkpoint and further fine-tune it so
that a frozen probe can no longer tell honest-vs-deceptive hidden states apart.
The auxiliary loss pushes deceptive-response activations toward the "honest"
class — gradients flow through the model weights but the probe itself stays
frozen.

The module is importable without torch; heavy imports live inside the training
entrypoints.
"""
from __future__ import annotations

__all__ = [
    "adversarial_loss",
    "train_vs_probes",
    "holdout_probe",
]
