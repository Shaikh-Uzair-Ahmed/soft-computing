"""First-order Takagi-Sugeno ANFIS with Jang's hybrid learning rule.

Each input has Gaussian membership functions. A rule fires by the product of
the memberships it selects. Consequent parameters of

    f = p0 + p1*x1 + ... + pn*xn

are re-estimated by ridge least squares every epoch. Premise parameters
(centers and widths) then take one gradient step on the squared error, with
the consequents held fixed.

Classification uses one regressor per class (one-vs-rest). The predicted
class is the model with the highest score.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ridge_solve(design: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    gram = design.T @ design
    width = gram.shape[0]
    scale = torch.trace(gram).clamp_min(1e-8) / width
    penalty = ridge * scale * torch.eye(width, device=design.device, dtype=design.dtype)
    rhs = design.T @ target
    try:
        return torch.linalg.solve(gram + penalty, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)


class AnfisRegressor(nn.Module):
    """Single-output first-order ANFIS."""

    def __init__(self, n_inputs: int, n_mfs: int = 2):
        super().__init__()
        if n_inputs < 1:
            raise ValueError("n_inputs must be positive")
        if n_mfs < 2:
            raise ValueError("n_mfs must be at least 2")
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        rules = list(itertools.product(range(n_mfs), repeat=n_inputs))
        self.register_buffer("rules", torch.tensor(rules, dtype=torch.long))
        self.centers = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.raw_sigma = nn.Parameter(torch.zeros(n_inputs, n_mfs))
        self.register_buffer(
            "consequents",
            torch.zeros(len(rules) * (n_inputs + 1)),
        )

    @property
    def n_rules(self) -> int:
        return int(self.rules.shape[0])

    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + 1e-3

    def init_from_data(self, features: np.ndarray) -> None:
        """Place membership centers on interior quantiles of each input."""
        features = np.asarray(features, dtype=np.float64)
        centers = np.zeros((self.n_inputs, self.n_mfs), dtype=np.float64)
        sigmas = np.zeros((self.n_inputs, self.n_mfs), dtype=np.float64)
        quantiles = np.linspace(0.0, 1.0, self.n_mfs + 2)[1:-1]
        for index in range(self.n_inputs):
            column = features[:, index]
            centers[index] = np.quantile(column, quantiles)
            sigmas[index] = max(float(np.std(column)), 0.5)
        sigmas = np.clip(sigmas, 1e-2, None)
        raw = np.log(np.expm1(sigmas))
        with torch.no_grad():
            self.centers.copy_(torch.as_tensor(centers, dtype=self.centers.dtype))
            self.raw_sigma.copy_(torch.as_tensor(raw, dtype=self.raw_sigma.dtype))

    def normalized_firing(self, features: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma()
        scaled = (features.unsqueeze(2) - self.centers.unsqueeze(0)) / sigma.unsqueeze(0)
        log_mu = -0.5 * scaled * scaled
        selected = [log_mu[:, index, self.rules[:, index]] for index in range(self.n_inputs)]
        log_w = torch.stack(selected, dim=0).sum(dim=0)
        log_w = log_w - log_w.max(dim=1, keepdim=True).values
        firing = torch.exp(log_w)
        return firing / firing.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def design_matrix(self, features: torch.Tensor, firing: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
        regressors = torch.cat([ones, features], dim=1)
        blocks = firing.unsqueeze(2) * regressors.unsqueeze(1)
        return blocks.reshape(features.shape[0], -1)

    def predict_tensor(self, features: torch.Tensor) -> torch.Tensor:
        firing = self.normalized_firing(features)
        return self.design_matrix(features, firing) @ self.consequents

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        epochs: int = 80,
        lr: float = 0.03,
        ridge: float = 1e-2,
    ) -> list[float]:
        self.init_from_data(features)
        inputs = torch.as_tensor(features, dtype=torch.float32)
        outputs = torch.as_tensor(target, dtype=torch.float32)
        optimizer = torch.optim.Adam((self.centers, self.raw_sigma), lr=lr)
        history: list[float] = []
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None

        for _ in range(epochs):
            optimizer.zero_grad()
            firing = self.normalized_firing(inputs)
            design = self.design_matrix(inputs, firing)
            with torch.no_grad():
                theta = _ridge_solve(design.detach(), outputs, ridge)
            prediction = design @ theta
            loss = F.mse_loss(prediction, outputs)
            if not torch.isfinite(loss):
                break
            value = float(loss.detach())
            history.append(value)
            if value < best_loss:
                best_loss = value
                best_state = {
                    "centers": self.centers.detach().clone(),
                    "raw_sigma": self.raw_sigma.detach().clone(),
                    "consequents": theta.detach().clone(),
                }
            loss.backward()
            torch.nn.utils.clip_grad_norm_((self.centers, self.raw_sigma), 1.0)
            optimizer.step()

        if best_state is not None:
            with torch.no_grad():
                self.centers.copy_(best_state["centers"])
                self.raw_sigma.copy_(best_state["raw_sigma"])
                self.consequents.copy_(best_state["consequents"])
        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.eval()
        inputs = torch.as_tensor(features, dtype=torch.float32)
        with torch.no_grad():
            return self.predict_tensor(inputs).numpy()


@dataclass
class RuleView:
    text: str
    mean_firing: float


class AnfisClassifier:
    """One-vs-rest ANFIS classifier."""

    def __init__(self, n_mfs: int = 2, epochs: int = 80, lr: float = 0.03, ridge: float = 1e-2):
        self.n_mfs = n_mfs
        self.epochs = epochs
        self.lr = lr
        self.ridge = ridge
        self.classes_: np.ndarray | None = None
        self.models_: list[AnfisRegressor] = []
        self.histories_: list[list[float]] = []

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "AnfisClassifier":
        self.classes_ = np.unique(labels)
        self.models_ = []
        self.histories_ = []
        for class_name in self.classes_:
            model = AnfisRegressor(features.shape[1], self.n_mfs)
            history = model.fit(
                features,
                (labels == class_name).astype(np.float32),
                epochs=self.epochs,
                lr=self.lr,
                ridge=self.ridge,
            )
            self.models_.append(model)
            self.histories_.append(history)
            print(
                f"ANFIS {class_name}: {model.n_rules} rules, "
                f"final train MSE {history[-1]:.4f}" if history else f"ANFIS {class_name}: failed"
            )
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return np.column_stack([model.predict(features) for model in self.models_])

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Call fit before predict")
        scores = self.decision_function(features)
        return self.classes_[np.argmax(scores, axis=1)]


def membership_labels(centers: np.ndarray) -> list[list[str]]:
    """Name membership functions Low/High (or Low/Medium/High) by center order."""
    catalog = ["Low", "Medium", "High", "Very high"]
    labels: list[list[str]] = []
    for row in centers:
        names = catalog[: len(row)] if len(row) != 2 else ["Low", "High"]
        ordered = np.argsort(row)
        assigned = [""] * len(row)
        for rank, mf_index in enumerate(ordered):
            assigned[int(mf_index)] = names[rank]
        labels.append(assigned)
    return labels


def describe_rules(
    model: AnfisRegressor,
    feature_names: list[str],
    features: np.ndarray,
    positive_mask: np.ndarray,
    top_k: int = 3,
) -> list[RuleView]:
    """Rank rules by how strongly they fire on the positive class."""
    inputs = torch.as_tensor(features, dtype=torch.float32)
    with torch.no_grad():
        firing = model.normalized_firing(inputs).numpy()
    if positive_mask.any():
        strength = firing[positive_mask].mean(axis=0)
    else:
        strength = firing.mean(axis=0)
    labels = membership_labels(model.centers.detach().numpy())
    theta = model.consequents.detach().numpy().reshape(model.n_rules, model.n_inputs + 1)
    ranked = np.argsort(strength)[::-1][:top_k]
    views: list[RuleView] = []
    for rule_index in ranked:
        clauses = []
        for input_index, mf_index in enumerate(model.rules[rule_index].tolist()):
            clauses.append(f"{feature_names[input_index]} is {labels[input_index][mf_index]}")
        weights = theta[rule_index]
        terms = [f"{weights[0]:+.3f}"]
        for input_index, name in enumerate(feature_names):
            terms.append(f"{weights[input_index + 1]:+.3f}*{name}")
        text = "IF " + " AND ".join(clauses) + " THEN score = " + " ".join(terms)
        views.append(RuleView(text=text, mean_firing=float(strength[rule_index])))
    return views
