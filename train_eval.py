"""Train ANFIS, an MLP, a genetic algorithm, GA-ANFIS, and GA-ANN on the same crops.

All five see the standardized geometry and CNN principal components built by
features.py, and all five are scored on the same held-out crops. GA-ANFIS
evolves the shared memberships and least squares sets the consequents. GA-ANN
evolves a tanh hidden layer and least squares sets the class scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier

from anfis import AnfisClassifier, describe_rules, membership_labels
from features import build_features, load_features, save_features
from ga_anfis import GaAnfisClassifier
from ga_anfis import save_result as save_ga_anfis
from ga_ann import GaAnnClassifier, describe_network
from ga_ann import save_result as save_ga_ann
from genetic import GeneticClassifier, describe_weights

PAPER_DETECTION_F1 = 0.770
PAPER_REFERENCE = (
    "Arya et al. (2024) report the CRDDC'2022 winning detector, an ensemble of "
    "one-stage and two-stage models (ShiYu_SeaView), at F1 0.770 on full images "
    "from all six countries. That score finds the damage box and names it. "
    "It is not crop classification."
)

CLASS_NAMES = {
    "D00": "longitudinal crack",
    "D10": "transverse crack",
    "D20": "alligator crack",
    "D40": "pothole",
}


def scores(labels: np.ndarray, predicted: np.ndarray, class_names: list[str]) -> dict:
    per_class = f1_score(labels, predicted, labels=class_names, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "per_class_f1": {name: float(value) for name, value in zip(class_names, per_class)},
    }


def plot_confusion(labels: np.ndarray, predicted: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    matrix = confusion_matrix(labels, predicted, labels=class_names)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_memberships(
    classifier: AnfisClassifier,
    feature_names: list[str],
    path: Path,
    title: str = "ANFIS Gaussian membership functions (standardized inputs)",
) -> None:
    classes = list(classifier.classes_)
    figure, axes = plt.subplots(
        len(classes),
        len(feature_names),
        figsize=(3.1 * len(feature_names), 2.3 * len(classes)),
        sharex=True,
        squeeze=False,
    )
    grid = np.linspace(-3.0, 3.0, 300)
    for row, (class_name, model) in enumerate(zip(classes, classifier.models_)):
        centers = model.centers.detach().numpy()
        sigma = model.sigma().detach().numpy()
        labels = membership_labels(centers)
        for col, feature_name in enumerate(feature_names):
            axis = axes[row, col]
            for mf_index in range(model.n_mfs):
                center = centers[col, mf_index]
                width = sigma[col, mf_index]
                curve = np.exp(-0.5 * ((grid - center) / width) ** 2)
                axis.plot(grid, curve, label=labels[col][mf_index])
            if row == 0:
                axis.set_title(feature_name)
            if col == 0:
                axis.set_ylabel(f"{class_name}\nmembership")
            axis.set_ylim(0, 1.05)
            axis.legend(fontsize=7, frameon=False)
    figure.suptitle(title, y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_fitness(history: list[float], path: Path, title: str = "Genetic algorithm fitness") -> None:
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(history, color="#1f4e79")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Best training macro F1")
    axis.set_title(title)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_model_comparison(report: dict, class_names: list[str], path: Path) -> None:
    models = ["ga_ann", "ga_anfis", "genetic_algorithm", "mlp", "anfis"]
    labels = ["GA-ANN", "GA-ANFIS", "Genetic algorithm", "MLP", "ANFIS"]
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(class_names) + 1)
    width = 0.15
    for offset, (model, label) in enumerate(zip(models, labels)):
        values = [report[model]["macro_f1"]] + [report[model]["per_class_f1"][name] for name in class_names]
        axis.bar(x + (offset - 2) * width, values, width, label=label)
    axis.axhline(PAPER_DETECTION_F1, color="#666666", linestyle="--", linewidth=1, label="Paper detection F1 0.770")
    axis.set_xticks(x, ["Macro F1", *class_names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1")
    axis.set_title("Crop classification F1. The dashed line is a different task.")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _ranking_sentence(report: dict) -> str:
    ranking = sorted(
        (
            ("GA-ANN", report["ga_ann"]["macro_f1"]),
            ("GA-ANFIS", report["ga_anfis"]["macro_f1"]),
            ("ANFIS", report["anfis"]["macro_f1"]),
            ("the MLP", report["mlp"]["macro_f1"]),
            ("the genetic algorithm", report["genetic_algorithm"]["macro_f1"]),
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    names = ", ".join(f"{name} ({score:.3f})" for name, score in ranking)
    return (
        f"On the shared crop test, macro F1 ranks {names}. "
        "GA-ANN evolves a tanh hidden layer and writes one weight per hidden unit. "
        "GA-ANFIS evolves the shared memberships and writes fuzzy rules. "
        "The linear genetic algorithm writes one weight per input."
    )


def write_comparison(report: dict, path: Path) -> None:
    lines = [
        "Comparison for presentation",
        "",
        "What was measured here",
        "Czech RDD2022 training images only. Each example is a damage crop whose box is already known.",
        "Classes present: D00 longitudinal crack, D10 transverse crack, D20 alligator crack.",
        "D40 pothole has no boxes in this Czech training subset.",
        f"Train crops: {report['n_train']}. Test crops: {report['n_test']}.",
        "Inputs: aspect ratio, relative area, and two CNN principal components.",
        "",
        "Test scores",
    ]
    for key, label in (
        ("ga_ann", "GA-ANN"),
        ("ga_anfis", "GA-ANFIS"),
        ("anfis", "ANFIS"),
        ("mlp", "MLP"),
        ("genetic_algorithm", "Genetic algorithm"),
    ):
        block = report[key]
        per_class = ", ".join(f"{name} {value:.3f}" for name, value in block["per_class_f1"].items())
        lines.append(
            f"{label}: accuracy {block['accuracy']:.3f}, macro F1 {block['macro_f1']:.3f} ({per_class})"
        )
    lines.extend(
        [
            "",
            "What the paper measured",
            PAPER_REFERENCE,
            "Rank 2 was YOLO with image tiling (F1 0.743). Rank 3 was YOLOv7 (F1 0.741).",
            "The paper does not train ANFIS, an MLP on crops, a genetic algorithm, GA-ANFIS, or GA-ANN.",
            "",
            "Why the higher crop scores do not beat the paper",
            "A crop classifier only names a box that a human already drew. The paper's F1 also requires the model to place that box in a full road image, across six countries and four damage types.",
            "A crop accuracy around 0.81 to 0.83 next to the paper's detection F1 of 0.770 makes the crop models look better. They are solving an easier problem on a smaller subset.",
            "",
            "How to say this in the ATT",
            "RDD2022 is a detection dataset built for deep convolutional networks. This study adds five classifiers on Czech damage crops: a neural network (MLP), standalone ANFIS, a linear genetic algorithm, a GA-ANN hybrid, and a GA-ANFIS hybrid.",
            _ranking_sentence(report),
            "None of the five replaces the paper's detector. GA-ANN searches the hidden layer. GA-ANFIS searches the memberships and states fuzzy rules.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rules(
    classifier: AnfisClassifier,
    feature_names: list[str],
    features: np.ndarray,
    labels: np.ndarray,
    path: Path,
    intro: list[str] | None = None,
) -> None:
    lines = intro or [
        "Learned ANFIS rules, ranked by mean firing strength on training crops of that class.",
        "Low and High refer to the standardized training distribution of each input.",
        "score is the first-order Takagi-Sugeno output of that one-vs-rest model.",
        "",
    ]
    for class_name, model in zip(classifier.classes_, classifier.models_):
        readable = CLASS_NAMES.get(str(class_name), str(class_name))
        lines.append(f"## {class_name} ({readable})")
        views = describe_rules(
            model,
            feature_names,
            features,
            labels == class_name,
            top_k=3,
        )
        for rank, view in enumerate(views, start=1):
            lines.append(f"{rank}. mean firing {view.mean_firing:.3f}")
            lines.append(f"   {view.text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, default=Path("outputs/crops"))
    parser.add_argument("--features", type=Path, default=Path("outputs/features.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--n-mfs", type=int, default=2)
    parser.add_argument("--population", type=int, default=40, help="GA-ANFIS population size")
    parser.add_argument("--generations", type=int, default=40, help="GA-ANFIS generations")
    parser.add_argument("--hidden", type=int, default=8, help="GA-ANN hidden units")
    parser.add_argument("--ann-population", type=int, default=80, help="GA-ANN population size")
    parser.add_argument("--ann-generations", type=int, default=80, help="GA-ANN generations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.features.exists():
        print(f"Loading features from {args.features}")
        data = load_features(args.features)
    else:
        data = build_features(args.crops)
        save_features(data, args.features)
        print(f"Saved features to {args.features}")

    x_train = data["X_train"]
    y_train = data["y_train"]
    x_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = [str(name) for name in data["feature_names"]]
    class_names = sorted(np.unique(y_train).tolist())
    print(f"Train crops: {len(y_train)}  Test crops: {len(y_test)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    anfis = AnfisClassifier(n_mfs=args.n_mfs, epochs=args.epochs)
    anfis.fit(x_train, y_train)
    anfis_pred = anfis.predict(x_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=400,
        random_state=args.seed,
        early_stopping=True,
        learning_rate_init=1e-3,
    )
    mlp.fit(x_train, y_train)
    mlp_pred = mlp.predict(x_test)

    genetic = GeneticClassifier(seed=args.seed)
    genetic.fit(x_train, y_train)
    genetic_pred = genetic.predict(x_test)

    ga_anfis = GaAnfisClassifier(
        n_mfs=args.n_mfs,
        population=args.population,
        generations=args.generations,
        seed=args.seed,
    )
    ga_anfis.fit(x_train, y_train)
    ga_anfis_pred = ga_anfis.predict(x_test)

    ga_ann = GaAnnClassifier(
        hidden=args.hidden,
        population=args.ann_population,
        generations=args.ann_generations,
        seed=args.seed,
    )
    ga_ann.fit(x_train, y_train)
    ga_ann_pred = ga_ann.predict(x_test)

    anfis_scores = scores(y_test, anfis_pred, class_names)
    mlp_scores = scores(y_test, mlp_pred, class_names)
    genetic_scores = scores(y_test, genetic_pred, class_names)
    ga_anfis_scores = scores(y_test, ga_anfis_pred, class_names)
    ga_ann_scores = scores(y_test, ga_ann_pred, class_names)
    report = {
        "dataset": "RDD2022 Czech Republic training images",
        "task": "classification of annotated damage crops",
        "paper_detection_f1_all_six_countries": PAPER_DETECTION_F1,
        "paper_note": PAPER_REFERENCE,
        "features": feature_names,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_counts_train": {name: int(np.sum(y_train == name)) for name in class_names},
        "class_counts_test": {name: int(np.sum(y_test == name)) for name in class_names},
        "anfis": anfis_scores,
        "mlp": mlp_scores,
        "genetic_algorithm": genetic_scores,
        "ga_anfis": ga_anfis_scores,
        "ga_ann": ga_ann_scores,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "classification_report.txt").write_text(
        "ANFIS\n"
        + classification_report(y_test, anfis_pred, labels=class_names, zero_division=0)
        + "\nMLP\n"
        + classification_report(y_test, mlp_pred, labels=class_names, zero_division=0)
        + "\nGenetic algorithm\n"
        + classification_report(y_test, genetic_pred, labels=class_names, zero_division=0)
        + "\nGA-ANFIS\n"
        + classification_report(y_test, ga_anfis_pred, labels=class_names, zero_division=0)
        + "\nGA-ANN\n"
        + classification_report(y_test, ga_ann_pred, labels=class_names, zero_division=0),
        encoding="utf-8",
    )
    save_ga_anfis(ga_anfis.result_, args.out_dir / "ga_anfis.npz")
    save_ga_ann(ga_ann.result_, args.out_dir / "ga_ann.npz")
    (args.out_dir / "ga_ann_network.txt").write_text(
        "\n".join(describe_network(ga_ann.result_, feature_names)) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "ga_weights.txt").write_text(
        "\n".join(describe_weights(genetic.result_, feature_names)) + "\n",
        encoding="utf-8",
    )
    write_comparison(report, args.out_dir / "comparison.txt")
    plot_confusion(
        y_test,
        anfis_pred,
        class_names,
        args.out_dir / "confusion_anfis.png",
        "ANFIS confusion matrix",
    )
    plot_confusion(
        y_test,
        mlp_pred,
        class_names,
        args.out_dir / "confusion_mlp.png",
        "MLP confusion matrix",
    )
    plot_confusion(
        y_test,
        genetic_pred,
        class_names,
        args.out_dir / "confusion_ga.png",
        "Genetic algorithm confusion matrix",
    )
    plot_confusion(
        y_test,
        ga_anfis_pred,
        class_names,
        args.out_dir / "confusion_ga_anfis.png",
        "GA-ANFIS confusion matrix",
    )
    plot_confusion(
        y_test,
        ga_ann_pred,
        class_names,
        args.out_dir / "confusion_ga_ann.png",
        "GA-ANN confusion matrix",
    )
    plot_memberships(anfis, feature_names, args.out_dir / "membership_functions.png")
    plot_memberships(
        ga_anfis,
        feature_names,
        args.out_dir / "ga_anfis_memberships.png",
        title="GA-ANFIS shared Gaussian memberships (standardized inputs)",
    )
    plot_fitness(genetic.result_.history_, args.out_dir / "ga_fitness.png")
    plot_fitness(
        ga_anfis.result_.history_,
        args.out_dir / "ga_anfis_fitness.png",
        title="GA-ANFIS fitness",
    )
    plot_fitness(
        ga_ann.result_.history_,
        args.out_dir / "ga_ann_fitness.png",
        title="GA-ANN fitness",
    )
    plot_model_comparison(report, class_names, args.out_dir / "comparison_f1.png")
    write_rules(anfis, feature_names, x_train, y_train, args.out_dir / "rules.txt")
    write_rules(
        ga_anfis,
        feature_names,
        x_train,
        y_train,
        args.out_dir / "ga_anfis_rules.txt",
        intro=[
            "GA-ANFIS rules. Memberships are shared and were chosen by the genetic algorithm.",
            "Consequents are ridge least squares.",
            "Low and High refer to the standardized training distribution of each input.",
            "",
        ],
    )
    print(json.dumps({
        "anfis": anfis_scores,
        "mlp": mlp_scores,
        "genetic_algorithm": genetic_scores,
        "ga_anfis": ga_anfis_scores,
        "ga_ann": ga_ann_scores,
    }, indent=2))


if __name__ == "__main__":
    main()
