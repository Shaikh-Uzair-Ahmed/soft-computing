"""Build the four ANFIS inputs from damage crops.

Two inputs are geometry already stored with each crop: bounding-box aspect
ratio and area relative to the source image. The other two are principal
components of a frozen MobileNetV2 embedding, fit on the training crops only.
All four columns are then standardized with the training split.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

FEATURE_NAMES = ["aspect_ratio", "relative_area", "cnn_pc1", "cnn_pc2"]


def load_metadata(crops_dir: Path) -> list[dict[str, str]]:
    csv_path = crops_dir / "metadata.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")
    return rows


def _embedding_model() -> tuple[torch.nn.Module, transforms.Compose]:
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    # ImageNet statistics used to train MobileNetV2. Stretching the crop to
    # 224x224 keeps thin cracks inside the frame; aspect ratio is a separate feature.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, transform


def embed_images(crops_dir: Path, rows: list[dict[str, str]], batch_size: int = 32) -> np.ndarray:
    model, transform = _embedding_model()
    embeddings: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        inputs = torch.stack(batch)
        with torch.no_grad():
            output = model(inputs).numpy()
        embeddings.append(output)
        batch.clear()

    for row in rows:
        image = Image.open(crops_dir / row["path"]).convert("RGB")
        batch.append(transform(image))
        if len(batch) == batch_size:
            flush()
    flush()
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def build_features(crops_dir: Path) -> dict[str, np.ndarray]:
    rows = load_metadata(crops_dir)
    print(f"Embedding {len(rows)} crops with frozen MobileNetV2")
    embeddings = embed_images(crops_dir, rows)
    geometry = np.array(
        [[float(row["aspect_ratio"]), float(row["relative_area"])] for row in rows],
        dtype=np.float32,
    )
    labels = np.array([row["label"] for row in rows])
    splits = np.array([row["split"] for row in rows])
    train = splits == "train"

    embedding_scaler = StandardScaler()
    scaled_train = embedding_scaler.fit_transform(embeddings[train])
    scaled_all = embedding_scaler.transform(embeddings)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaled_train)
    components = pca.transform(scaled_all).astype(np.float32)

    raw = np.concatenate([geometry, components], axis=1)
    feature_scaler = StandardScaler()
    feature_scaler.fit(raw[train])
    scaled = feature_scaler.transform(raw).astype(np.float32)
    print(
        "PCA variance explained: "
        + ", ".join(f"{value:.3f}" for value in pca.explained_variance_ratio_)
    )
    return {
        "X_train": scaled[train],
        "y_train": labels[train],
        "X_test": scaled[~train],
        "y_test": labels[~train],
        "feature_names": np.array(FEATURE_NAMES),
    }


def save_features(payload: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
