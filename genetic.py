"""Real-coded genetic algorithm for one-vs-rest linear classification.

A chromosome holds a bias and one weight per feature for every class. The
predicted class is the highest score. Fitness is macro F1 on the training
crops. Selection is tournament, crossover is whole-arithmetic, and mutation
adds Gaussian noise. The best few chromosomes are copied unchanged each
generation (elitism). The test crops are never used as fitness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class GeneticResult:
    classes_: np.ndarray
    weights_: np.ndarray
    history_: list[float]


class GeneticClassifier:
    """Evolve a linear one-vs-rest classifier."""

    def __init__(
        self,
        population: int = 80,
        generations: int = 80,
        tournament: int = 3,
        elite: int = 2,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.25,
        mutation_scale: float = 0.25,
        seed: int = 42,
    ):
        self.population = population
        self.generations = generations
        self.tournament = tournament
        self.elite = elite
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.seed = seed
        self.result_: GeneticResult | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "GeneticClassifier":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        n_classes = len(classes)
        n_inputs = features.shape[1] + 1
        design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
        n_genes = n_classes * n_inputs
        rng = np.random.default_rng(self.seed)
        population = rng.normal(0.0, 0.5, size=(self.population, n_genes))
        history: list[float] = []

        def fitness_values(chromosomes: np.ndarray) -> np.ndarray:
            scores = []
            for chromosome in chromosomes:
                weights = chromosome.reshape(n_classes, n_inputs)
                predicted = classes[np.argmax(design @ weights.T, axis=1)]
                scores.append(f1_score(labels, predicted, average="macro", zero_division=0))
            return np.asarray(scores, dtype=np.float64)

        def pick(values: np.ndarray) -> np.ndarray:
            contenders = rng.integers(0, len(population), size=self.tournament)
            return population[contenders[np.argmax(values[contenders])]].copy()

        for _ in range(self.generations):
            values = fitness_values(population)
            history.append(float(values.max()))
            order = np.argsort(values)[::-1]
            next_population = [population[index].copy() for index in order[: self.elite]]
            while len(next_population) < self.population:
                parent_a = pick(values)
                parent_b = pick(values)
                if rng.random() < self.crossover_rate:
                    blend = rng.random()
                    child = blend * parent_a + (1.0 - blend) * parent_b
                else:
                    child = parent_a.copy()
                mutate = rng.random(n_genes) < self.mutation_rate
                child[mutate] += rng.normal(0.0, self.mutation_scale, size=int(mutate.sum()))
                next_population.append(child)
            population = np.vstack(next_population)

        final_values = fitness_values(population)
        best = population[int(np.argmax(final_values))].reshape(n_classes, n_inputs)
        history.append(float(final_values.max()))
        self.result_ = GeneticResult(classes_=classes, weights_=best, history_=history)
        print(
            f"GA: population {self.population}, generations {self.generations}, "
            f"best train macro F1 {history[-1]:.4f}"
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Call fit before predict")
        features = np.asarray(features, dtype=np.float64)
        design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
        scores = design @ self.result_.weights_.T
        return self.result_.classes_[np.argmax(scores, axis=1)]


def describe_weights(result: GeneticResult, feature_names: list[str]) -> list[str]:
    """Turn the winning chromosome into one line per class."""
    lines = [
        "Genetic algorithm linear scores. Positive weight raises that class score.",
        "The first number is the bias. Features are standardized.",
        "",
    ]
    names = ["bias", *feature_names]
    for class_name, weights in zip(result.classes_, result.weights_):
        terms = " ".join(f"{weight:+.3f}*{name}" for weight, name in zip(weights, names))
        lines.append(f"{class_name}: score = {terms}")
    return lines
