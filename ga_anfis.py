"""Genetic algorithm combined with ANFIS.

The chromosome holds the Gaussian membership centres and widths, shared by
every class. For each chromosome the rule consequents are solved by the same
ridge least squares step ANFIS uses. Fitness is macro F1 on the training
crops. The test crops are scored only after the search finishes.

Operators match genetic.py: tournament selection, whole-arithmetic crossover,
Gaussian mutation, and elitism.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from anfis import AnfisRegressor, _ridge_solve, describe_rules
from features import load_features

RIDGE = 1e-2


@dataclass
class GaAnfisResult:
    classes_: np.ndarray
    centers_: np.ndarray
    raw_sigma_: np.ndarray
    consequents_: np.ndarray
    history_: list[float]


class GaAnfisClassifier:
    """Evolve ANFIS memberships; least squares sets the rule consequents."""

    def __init__(
        self,
        n_mfs: int = 2,
        population: int = 40,
        generations: int = 40,
        tournament: int = 3,
        elite: int = 2,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.25,
        mutation_scale: float = 0.15,
        ridge: float = RIDGE,
        seed: int = 42,
    ):
        self.n_mfs = n_mfs
        self.population = population
        self.generations = generations
        self.tournament = tournament
        self.elite = elite
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.ridge = ridge
        self.seed = seed
        self.result_: GaAnfisResult | None = None
        self.models_: list[AnfisRegressor] = []

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "GaAnfisClassifier":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        inputs = torch.as_tensor(features, dtype=torch.float32)
        targets = [
            torch.as_tensor(labels == name, dtype=torch.float32) for name in classes
        ]
        probe = AnfisRegressor(features.shape[1], self.n_mfs)
        probe.init_from_data(features)
        start = _pack(probe)
        n_genes = start.shape[0]
        rng = np.random.default_rng(self.seed)
        population = start + rng.normal(0.0, self.mutation_scale, size=(self.population, n_genes))
        population[0] = start
        history: list[float] = []

        def fitness_values(chromosomes: np.ndarray) -> np.ndarray:
            values = np.empty(len(chromosomes), dtype=np.float64)
            for index, chromosome in enumerate(chromosomes):
                predicted = self._labels_for(probe, inputs, classes, targets, chromosome)
                values[index] = f1_score(labels, predicted, average="macro", zero_division=0)
            return values

        def pick(values: np.ndarray) -> np.ndarray:
            contenders = rng.integers(0, len(population), size=self.tournament)
            return population[contenders[np.argmax(values[contenders])]].copy()

        for generation in range(self.generations):
            values = fitness_values(population)
            history.append(float(values.max()))
            print(f"GA-ANFIS generation {generation:02d}: best train macro F1 {history[-1]:.4f}")
            order = np.argsort(values)[::-1]
            next_population = [population[index].copy() for index in order[: self.elite]]
            while len(next_population) < self.population:
                parent_a = pick(values)
                parent_b = pick(values)
                if rng.random() < self.crossover_rate:
                    blend = float(rng.random())
                    child = blend * parent_a + (1.0 - blend) * parent_b
                else:
                    child = parent_a.copy()
                mutate = rng.random(n_genes) < self.mutation_rate
                child[mutate] += rng.normal(0.0, self.mutation_scale, size=int(mutate.sum()))
                next_population.append(child)
            population = np.vstack(next_population)

        final_values = fitness_values(population)
        best_index = int(np.argmax(final_values))
        best = population[best_index]
        history.append(float(final_values[best_index]))
        centers, raw_sigma, consequents = self._solve(probe, inputs, targets, best)
        self.result_ = GaAnfisResult(
            classes_=classes,
            centers_=centers,
            raw_sigma_=raw_sigma,
            consequents_=consequents,
            history_=history,
        )
        self.classes_ = classes
        self.models_ = _models_from_result(self.result_, features.shape[1], self.n_mfs)
        print(
            f"GA-ANFIS: population {self.population}, generations {self.generations}, "
            f"best train macro F1 {history[-1]:.4f}"
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Call fit before predict")
        inputs = torch.as_tensor(np.asarray(features, dtype=np.float32))
        with torch.no_grad():
            columns = [model.predict_tensor(inputs) for model in self.models_]
            scores = torch.column_stack(columns)
        return self.result_.classes_[scores.argmax(dim=1).numpy()]

    def _labels_for(
        self,
        probe: AnfisRegressor,
        inputs: torch.Tensor,
        classes: np.ndarray,
        targets: list[torch.Tensor],
        chromosome: np.ndarray,
    ) -> np.ndarray:
        centers, raw_sigma = _unpack(chromosome, probe.n_inputs, probe.n_mfs)
        with torch.no_grad():
            probe.centers.copy_(torch.as_tensor(centers, dtype=probe.centers.dtype))
            probe.raw_sigma.copy_(torch.as_tensor(raw_sigma, dtype=probe.raw_sigma.dtype))
            firing = probe.normalized_firing(inputs)
            design = probe.design_matrix(inputs, firing)
            columns = [_ridge_solve(design, target, self.ridge) for target in targets]
            scores = torch.column_stack([design @ theta for theta in columns])
        return classes[scores.argmax(dim=1).numpy()]

    def _solve(
        self,
        probe: AnfisRegressor,
        inputs: torch.Tensor,
        targets: list[torch.Tensor],
        chromosome: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers, raw_sigma = _unpack(chromosome, probe.n_inputs, probe.n_mfs)
        with torch.no_grad():
            probe.centers.copy_(torch.as_tensor(centers, dtype=probe.centers.dtype))
            probe.raw_sigma.copy_(torch.as_tensor(raw_sigma, dtype=probe.raw_sigma.dtype))
            firing = probe.normalized_firing(inputs)
            design = probe.design_matrix(inputs, firing)
            thetas = [_ridge_solve(design, target, self.ridge).numpy() for target in targets]
        return centers, raw_sigma, np.stack(thetas)


def _pack(model: AnfisRegressor) -> np.ndarray:
    centers = model.centers.detach().numpy().reshape(-1)
    raw_sigma = model.raw_sigma.detach().numpy().reshape(-1)
    return np.concatenate([centers, raw_sigma]).astype(np.float64)


def _unpack(chromosome: np.ndarray, n_inputs: int, n_mfs: int) -> tuple[np.ndarray, np.ndarray]:
    premises = n_inputs * n_mfs
    centers = chromosome[:premises].reshape(n_inputs, n_mfs)
    raw_sigma = chromosome[premises:].reshape(n_inputs, n_mfs)
    return centers.astype(np.float32), raw_sigma.astype(np.float32)


def _models_from_result(result: GaAnfisResult, n_inputs: int, n_mfs: int) -> list[AnfisRegressor]:
    models = []
    for theta in result.consequents_:
        model = AnfisRegressor(n_inputs, n_mfs)
        with torch.no_grad():
            model.centers.copy_(torch.as_tensor(result.centers_, dtype=model.centers.dtype))
            model.raw_sigma.copy_(torch.as_tensor(result.raw_sigma_, dtype=model.raw_sigma.dtype))
            model.consequents.copy_(torch.as_tensor(theta, dtype=model.consequents.dtype))
        models.append(model)
    return models


def save_result(result: GaAnfisResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        classes=result.classes_,
        centers=result.centers_,
        raw_sigma=result.raw_sigma_,
        consequents=result.consequents_,
        history=np.asarray(result.history_, dtype=np.float64),
    )


def load_classifier(path: Path, n_inputs: int, n_mfs: int = 2) -> GaAnfisClassifier:
    with np.load(path, allow_pickle=False) as data:
        result = GaAnfisResult(
            classes_=data["classes"],
            centers_=data["centers"],
            raw_sigma_=data["raw_sigma"],
            consequents_=data["consequents"],
            history_=data["history"].tolist(),
        )
    classifier = GaAnfisClassifier(n_mfs=n_mfs)
    classifier.result_ = result
    classifier.classes_ = result.classes_
    classifier.models_ = _models_from_result(result, n_inputs, n_mfs)
    return classifier


def _scores(labels: np.ndarray, predicted: np.ndarray, class_names: list[str]) -> dict:
    per_class = f1_score(labels, predicted, labels=class_names, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "per_class_f1": {name: float(value) for name, value in zip(class_names, per_class)},
    }


def _plot_fitness(history: list[float], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(history, color="#1f4e79")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best training macro F1")
    axis.set_title("GA-ANFIS fitness")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/features.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ga_anfis.npz"))
    parser.add_argument("--test", action="store_true", help="Score the saved hybrid. Do not search again.")
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_features(args.features)
    x_train = data["X_train"]
    y_train = data["y_train"]
    x_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = [str(name) for name in data["feature_names"]]
    class_names = sorted(np.unique(y_train).tolist())

    if args.test:
        classifier = load_classifier(args.checkpoint, x_train.shape[1])
        print(f"Loaded {args.checkpoint}")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        classifier = GaAnfisClassifier(
            population=args.population,
            generations=args.generations,
            seed=args.seed,
        )
        classifier.fit(x_train, y_train)
        save_result(classifier.result_, args.checkpoint)

    predicted = classifier.predict(x_test)
    report = _scores(y_test, predicted, class_names)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ga_anfis_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "ga_anfis_report.txt").write_text(
        classification_report(y_test, predicted, labels=class_names, zero_division=0),
        encoding="utf-8",
    )
    if classifier.result_ is not None and not args.test:
        _plot_fitness(classifier.result_.history_, args.out_dir / "ga_anfis_fitness.png")
    lines = ["GA-ANFIS rules. Memberships are shared. Consequents are least squares.", ""]
    for model, class_name in zip(classifier.models_, classifier.result_.classes_):
        lines.append(f"## {class_name}")
        views = describe_rules(
            model,
            feature_names,
            x_train,
            y_train == class_name,
            top_k=3,
        )
        for rank, view in enumerate(views, start=1):
            lines.append(f"{rank}. mean firing {view.mean_firing:.3f}")
            lines.append(f"   {view.text}")
        lines.append("")
    (args.out_dir / "ga_anfis_rules.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
