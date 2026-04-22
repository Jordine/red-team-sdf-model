"""
Probe classes — ported from ApolloResearch/deception-detection/deception_detection/detectors.py.

Three detector types are preserved with Apollo's definitions:

    LogisticRegressionProbe  — sklearn LogisticRegression with StandardScaler.
                               This is the probe used in the main paper results.
    MMSProbe                 — Mean Mass Subtraction: direction = pos_mean - neg_mean.
    LATProbe                 — Linear Artificial Tomography (from RepE paper):
                               PCA on paired diff vectors with sign-randomization.

All three operate on activations of shape [n_examples, n_layers, emb] after the
extract_activations step has already reduced to "detection tokens".

Apollo's detectors.py originally supports activations of shape
[toks, layer, emb] where `toks` is the total number of detection tokens across
the batch (they flatten batch*seq). Our extract_activations reduces per dialogue
to either (a) mean over detection tokens or (b) last detection token, so here
`toks` is effectively `n_dialogues`. This mirrors Apollo's
MeanLogisticRegressionDetector behaviour (used in several of their configs).
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import torch
from einops import einsum, rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class DirectionProbe(ABC):
    """Abstract base: each probe learns a direction per layer and scores by
    projecting activations onto that direction (averaged across layers).

    Preserves Apollo's DirectionDetector API:
        fit(positive_acts, negative_acts) -> None
        score(acts) -> Tensor of shape [n_examples]
        save(path) / load(path)
    """

    layers: list[int]
    directions: torch.Tensor | None  # shape [n_layers, emb]

    def __init__(self, layers: list[int]):
        self.layers = layers
        self.directions = None

    @abstractmethod
    def fit(self, positive_acts: torch.Tensor, negative_acts: torch.Tensor) -> None:
        """Fit probe to paired activations.

        Args:
            positive_acts: [n_examples, n_layers, emb] — deceptive activations.
            negative_acts: [n_examples, n_layers, emb] — honest activations.
        """
        ...

    @torch.no_grad()
    def score(self, acts: torch.Tensor) -> torch.Tensor:
        """Score activations by projecting onto learned direction(s).

        Args:
            acts: [n_examples, n_layers, emb].

        Returns:
            [n_examples] — averaged across layers. Higher = more deceptive.
        """
        assert self.directions is not None, "Probe must be fit before scoring."
        acts_t = acts.to(self.directions)
        by_layer = einsum(
            acts_t, self.directions, "n layer emb, layer emb -> n layer"
        )
        return by_layer.mean(dim=-1)

    def save(self, file_path: str | Path) -> None:
        data = {
            "layers": self.layers,
            "directions": (
                self.directions.cpu() if self.directions is not None else None
            ),
            "class_name": self.__class__.__name__,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: str | Path) -> Self:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        probe = cls(data["layers"])
        probe.directions = data["directions"]
        return probe


class MMSProbe(DirectionProbe):
    """Mean Mass Subtraction: direction = mean(pos) - mean(neg)."""

    @torch.no_grad()
    def fit(self, positive_acts: torch.Tensor, negative_acts: torch.Tensor) -> None:
        pos_mean = positive_acts.mean(dim=0)
        neg_mean = negative_acts.mean(dim=0)
        self.directions = pos_mean - neg_mean


class LATProbe(DirectionProbe):
    """Linear Artificial Tomography — Apollo port of RepE's direction finder.

    Steps (ported verbatim from deception_detection/detectors.py::LATDetector):
      1. Compute paired diff vectors (pos - neg).
      2. Randomly flip signs on half of them.
      3. Take the first PCA component per layer.
      4. Orient the direction so that pos projects higher than neg.
    """

    @torch.no_grad()
    def fit(self, positive_acts: torch.Tensor, negative_acts: torch.Tensor) -> None:
        diffs = positive_acts - negative_acts  # [n, layer, emb]
        if torch.cuda.is_available():
            diffs = diffs.cuda()
        # random sign flips
        signs = torch.where(torch.randn(len(diffs)) > 0, 1, -1).to(diffs)
        diffs = diffs * signs[:, None, None]
        diffs = diffs.to(torch.float32)

        # PCA per layer: [layer, n, emb] -> V [layer, emb, 1]
        diffs = rearrange(diffs, "n layer emb -> layer n emb")
        _, _, V = torch.pca_lowrank(diffs, q=1, center=True)
        self.directions = V[:, :, 0].cpu()

        # orient so pos > neg
        pos_by_layer = einsum(
            positive_acts.to(self.directions),
            self.directions,
            "n layer emb, layer emb -> layer",
        )
        neg_by_layer = einsum(
            negative_acts.to(self.directions),
            self.directions,
            "n layer emb, layer emb -> layer",
        )
        self.directions = (
            self.directions * torch.where(pos_by_layer > neg_by_layer, 1, -1).unsqueeze(1)
        )


class LogisticRegressionProbe(DirectionProbe):
    """sklearn LogisticRegression with StandardScaler and strong L2 regularization.

    This is Apollo's MeanLogisticRegressionDetector / LogisticRegressionDetector,
    ported faithfully:
      - Concatenate pos+neg samples, label pos=1, neg=0.
      - Reshape [n, layer, emb] -> [n, layer*emb].
      - StandardScaler on features.
      - LogisticRegression(C=1/reg_coeff, fit_intercept=False, random_state=42).
      - Reshape coefficients back to [layer, emb] and store as self.directions.

    reg_coeff default matches Apollo's repe.yaml (reg_coeff=10). Apollo's
    detectors.py uses a default of 1e3 but their scripts configs generally use 10.
    We expose both as kwargs.
    """

    scaler_mean: torch.Tensor | None
    scaler_scale: torch.Tensor | None

    def __init__(
        self,
        layers: list[int],
        dtype: torch.dtype = torch.float32,
        reg_coeff: float = 10.0,
        normalize: bool = True,
    ):
        super().__init__(layers)
        self.dtype = dtype
        self.reg_coeff = reg_coeff
        self.normalize = normalize
        self.scaler_mean = None
        self.scaler_scale = None

    def fit(self, positive_acts: torch.Tensor, negative_acts: torch.Tensor) -> None:
        X_pos = positive_acts.to(self.dtype).cpu().numpy()
        X_neg = negative_acts.to(self.dtype).cpu().numpy()
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])

        n_samples, n_layers, n_features = X.shape
        X_reshaped = X.reshape(n_samples, n_layers * n_features)

        if self.normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_reshaped)
            self.scaler_mean = torch.tensor(
                scaler.mean_.reshape(n_layers, n_features), dtype=self.dtype
            )
            self.scaler_scale = torch.tensor(
                scaler.scale_.reshape(n_layers, n_features), dtype=self.dtype
            )
        else:
            X_scaled = X_reshaped

        model = LogisticRegression(
            C=1.0 / self.reg_coeff, random_state=42, fit_intercept=False, max_iter=1000
        )
        model.fit(X_scaled, y)

        coef = model.coef_.reshape(n_layers, n_features)
        self.directions = torch.tensor(coef, dtype=self.dtype)

    @torch.no_grad()
    def score(self, acts: torch.Tensor) -> torch.Tensor:
        assert self.directions is not None
        acts_t = acts.to(self.directions)
        if self.normalize:
            assert self.scaler_mean is not None and self.scaler_scale is not None
            acts_t = (acts_t - self.scaler_mean) / self.scaler_scale
        by_layer = einsum(
            acts_t, self.directions, "n layer emb, layer emb -> n layer"
        )
        return by_layer.mean(dim=-1)

    def save(self, file_path: str | Path) -> None:
        data = {
            "layers": self.layers,
            "directions": (
                self.directions.cpu() if self.directions is not None else None
            ),
            "scaler_mean": (
                self.scaler_mean.cpu() if self.scaler_mean is not None else None
            ),
            "scaler_scale": (
                self.scaler_scale.cpu() if self.scaler_scale is not None else None
            ),
            "normalize": self.normalize,
            "reg_coeff": self.reg_coeff,
            "class_name": self.__class__.__name__,
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: str | Path) -> "LogisticRegressionProbe":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        probe = cls(
            data["layers"],
            reg_coeff=data.get("reg_coeff", 10.0),
            normalize=data.get("normalize", True),
        )
        probe.directions = data["directions"]
        probe.scaler_mean = data.get("scaler_mean")
        probe.scaler_scale = data.get("scaler_scale")
        return probe


PROBE_CLASSES = {
    "lr": LogisticRegressionProbe,
    "mms": MMSProbe,
    "lat": LATProbe,
}


def get_probe_class(method: str) -> type[DirectionProbe]:
    if method not in PROBE_CLASSES:
        raise ValueError(
            f"Unknown probe method: {method!r}. Must be one of {list(PROBE_CLASSES)}."
        )
    return PROBE_CLASSES[method]
