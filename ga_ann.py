"""Genetic algorithm combined with a neural network.

The chromosome holds the weights and biases of one tanh hidden layer. For
each chromosome the output weights are solved by ridge least squares on those
hidden activations, one score per class. Fitness is macro F1 on the training
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from features import load_features

RIDGE = 1e-2


@dataclass
class GaAnnResult:
    classes_: np.ndarray
    hidden_weights_: np.ndarray
    hidden_bias_: np.ndarray
    output_weights_: np.ndarray
    history_: list[float]


class GaAnnClassifier:
    """Evolve a hidden layer; least squares sets the class scores."""

    def __init__(
        self,
        hidden: int = 8,
        population: int = 80,
        generations: int = 80,
        tournament: int = 3,
        elite: int = 2,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.25,
        mutation_scale: float = 0.20,
        ridge: float = RIDGE,
        seed: int = 42,
    ):
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.hidden = hidden
        self.population = population
        self.generations = generations
        self.tournament = tournament
        self.elite = elite
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.ridge = ridge
        self.seed = seed
        self.result_: GaAnnResult | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "GaAnnClassifier":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        targets = [(labels == name).astype(np.float64) for name in classes]
        n_inputs = features.shape[1]
        n_genes = self.hidden * (n_inputs + 1)
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(n_inputs)
        population = rng.normal(0.0, scale, size=(self.population, n_genes))
        history: list[float] = []

        def fitness_values(chromosomes: np.ndarray) -> np.ndarray:
            values = np.empty(len(chromosomes), dtype=np.float64)
            for index, chromosome in enumerate(chromosomes):
                predicted = self._labels_for(features, classes, targets, chromosome, n_inputs)
                values[index] = f1_score(labels, predicted, average="macro", zero_division=0)
            return values

        def pick(values: np.ndarray) -> np.ndarray:
            contenders = rng.integers(0, len(population), size=self.tournament)
            return population[contenders[np.argmax(values[contenders])]].copy()

        for generation in range(self.generations):
            values = fitness_values(population)
            history.append(float(values.max()))
            print(f"GA-ANN generation {generation:02d}: best train macro F1 {history[-1]:.4f}")
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
        hidden_weights, hidden_bias, output_weights = self._solve(features, targets, best, n_inputs)
        self.result_ = GaAnnResult(
            classes_=classes,
            hidden_weights_=hidden_weights,
            hidden_bias_=hidden_bias,
            output_weights_=output_weights,
            history_=history,
        )
        print(
            f"GA-ANN: hidden {self.hidden}, population {self.population}, "
            f"generations {self.generations}, best train macro F1 {history[-1]:.4f}"
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Call fit before predict")
        scores = _scores(np.asarray(features, dtype=np.float64), self.result_)
        return self.result_.classes_[np.argmax(scores, axis=1)]

    def _labels_for(
        self,
        features: np.ndarray,
        classes: np.ndarray,
        targets: list[np.ndarray],
        chromosome: np.ndarray,
        n_inputs: int,
    ) -> np.ndarray:
        hidden_weights, hidden_bias = _unpack(chromosome, n_inputs, self.hidden)
        hidden = np.tanh(features @ hidden_weights.T + hidden_bias)
        output_weights = _solve_outputs(hidden, targets, self.ridge)
        scores = _design(hidden) @ output_weights.T
        return classes[np.argmax(scores, axis=1)]

    def _solve(
        self,
        features: np.ndarray,
        targets: list[np.ndarray],
        chromosome: np.ndarray,
        n_inputs: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_weights, hidden_bias = _unpack(chromosome, n_inputs, self.hidden)
        hidden = np.tanh(features @ hidden_weights.T + hidden_bias)
        output_weights = _solve_outputs(hidden, targets, self.ridge)
        return hidden_weights, hidden_bias, output_weights


def _unpack(chromosome: np.ndarray, n_inputs: int, hidden: int) -> tuple[np.ndarray, np.ndarray]:
    n_weight = hidden * n_inputs
    weights = chromosome[:n_weight].reshape(hidden, n_inputs)
    bias = chromosome[n_weight:]
    return weights, bias


def _design(hidden: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(hidden), 1)), hidden], axis=1)


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    gram = design.T @ design
    width = gram.shape[0]
    scale = max(float(np.trace(gram)) / width, 1e-8)
    penalty = ridge * scale * np.eye(width, dtype=design.dtype)
    try:
        return np.linalg.solve(gram + penalty, design.T @ target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(design, target, rcond=None)[0]


def _solve_outputs(hidden: np.ndarray, targets: list[np.ndarray], ridge: float) -> np.ndarray:
    design = _design(hidden)
    return np.stack([_ridge_solve(design, target, ridge) for target in targets])


def _scores(features: np.ndarray, result: GaAnnResult) -> np.ndarray:
    hidden = np.tanh(features @ result.hidden_weights_.T + result.hidden_bias_)
    return _design(hidden) @ result.output_weights_.T


def describe_network(result: GaAnnResult, feature_names: list[str]) -> list[str]:
    """Turn the winning network into one line per hidden unit and class."""
    lines = [
        "GA-ANN. The hidden layer is tanh. Output weights are ridge least squares.",
        "A positive output weight raises that class score when the hidden unit is active.",
        "",
        "Hidden units",
    ]
    for index, (bias, weights) in enumerate(zip(result.hidden_bias_, result.hidden_weights_)):
        terms = " ".join(f"{weight:+.3f}*{name}" for weight, name in zip(weights, feature_names))
        lines.append(f"h{index}: tanh({bias:+.3f} {terms})")
    lines.extend(["", "Class scores"])
    names = ["bias", *[f"h{index}" for index in range(len(result.hidden_bias_))]]
    for class_name, weights in zip(result.classes_, result.output_weights_):
        terms = " ".join(f"{weight:+.3f}*{name}" for weight, name in zip(weights, names))
        lines.append(f"{class_name}: score = {terms}")
    return lines


def save_result(result: GaAnnResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        classes=result.classes_,
        hidden_weights=result.hidden_weights_,
        hidden_bias=result.hidden_bias_,
        output_weights=result.output_weights_,
        history=np.asarray(result.history_, dtype=np.float64),
    )


def load_classifier(path: Path) -> GaAnnClassifier:
    with np.load(path, allow_pickle=False) as data:
        result = GaAnnResult(
            classes_=data["classes"],
            hidden_weights_=data["hidden_weights"],
            hidden_bias_=data["hidden_bias"],
            output_weights_=data["output_weights"],
            history_=data["history"].tolist(),
        )
    classifier = GaAnnClassifier(hidden=int(result.hidden_bias_.shape[0]))
    classifier.result_ = result
    return classifier


def _metric_scores(labels: np.ndarray, predicted: np.ndarray, class_names: list[str]) -> dict:
    per_class = f1_score(labels, predicted, labels=class_names, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "per_class_f1": {name: float(value) for name, value in zip(class_names, per_class)},
    }


def _plot_confusion(labels: np.ndarray, predicted: np.ndarray, class_names: list[str], path: Path) -> None:
    matrix = confusion_matrix(labels, predicted, labels=class_names)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("GA-ANN confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_fitness(history: list[float], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(history, color="#1f4e79")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best training macro F1")
    axis.set_title("GA-ANN fitness")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("outputs/features.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ga_ann.npz"))
    parser.add_argument("--test", action="store_true", help="Score the saved hybrid. Do not search again.")
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=80)
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
        classifier = load_classifier(args.checkpoint)
        print(f"Loaded {args.checkpoint}")
    else:
        np.random.seed(args.seed)
        classifier = GaAnnClassifier(
            hidden=args.hidden,
            population=args.population,
            generations=args.generations,
            seed=args.seed,
        )
        classifier.fit(x_train, y_train)
        save_result(classifier.result_, args.checkpoint)

    predicted = classifier.predict(x_test)
    report = _metric_scores(y_test, predicted, class_names)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ga_ann_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "ga_ann_report.txt").write_text(
        classification_report(y_test, predicted, labels=class_names, zero_division=0),
        encoding="utf-8",
    )
    _plot_confusion(y_test, predicted, class_names, args.out_dir / "confusion_ga_ann.png")
    if classifier.result_ is not None and not args.test:
        _plot_fitness(classifier.result_.history_, args.out_dir / "ga_ann_fitness.png")
        (args.out_dir / "ga_ann_network.txt").write_text(
            "\n".join(describe_network(classifier.result_, feature_names)) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
