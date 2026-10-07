"""
Brief description: Build a CL module by wiring encoder, recall, and score-module data for one category.

Last modified on: 2026-05-06
Commit message: Minor changes on encoder trainin (choosing w or without table)
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from typing import Sequence, Optional
from tqdm import tqdm

from grasp_learning_cl.grasp_classification.encoders import GraspPointCloudDataModule
from grasp_learning_cl.grasp_classification.grasp_score_module_probabilistic import GraspScoreProbabilisticModule
from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule


N_TRAINING_POINTS = 10_000
TEMPERATURE_MIN = 0.2
TEMPERATURE_MAX = 0.5
TRUNCATION_DISTANCE = 1.0
TEMPERATURE_STEP = 0.05
OFFLINE_MAX_STRENGTH = 10.0
ONLINE_MAX_STRENGTH = 1000.0
ONLINE_SCALE_FACTOR = 3.0
PRIOR_STRENGTH = 1.0
PRIOR_MEAN = 0.1
MAX_NEIGHBOR_CANDIDATES = 100
DEMO_ODDS_MARGIN = 1
MAX_SINGLE_DEMO_WEIGHT = 500


def prepare_batch(
    batch: Sequence[dict],
    preprocessor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [
        preprocessor(sample['points'], sample['normals'], sample['pose'], device=device)
        for sample in batch
    ]
    labels = torch.tensor([float(sample['label']) for sample in batch], dtype=torch.float32, device=device)
    return torch.stack(features, dim=0), labels


def collect_embeddings(
    model,
    loader,
    limit: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total = 0
    for batch in tqdm(loader, total=len(loader), desc='Collecting embeddings'):
        feats, lbls = prepare_batch(batch, model.preprocessor, model.device)
        with torch.no_grad():
            emb = model.encode(feats)
        embeddings.append(emb.detach().cpu().numpy())
        labels.append(lbls.detach().cpu().numpy())
        total += emb.size(0)
        if limit is not None and total >= limit:
            break

    if not embeddings:
        raise RuntimeError('No samples collected; check dataset path and loaders.')

    feat_arr = np.concatenate(embeddings, axis=0)
    lbl_arr = np.concatenate(labels, axis=0)
    if limit is not None:
        feat_arr = feat_arr[:limit]
        lbl_arr = lbl_arr[:limit]

    return feat_arr.astype(np.float32, copy=False), lbl_arr.astype(np.float32, copy=False)



def score_module_kwargs(temperature: float) -> dict:
    return {
        "temperature": temperature,
        "truncation_distance": TRUNCATION_DISTANCE,
        "offline_max_strength": OFFLINE_MAX_STRENGTH,
        "online_max_strength": ONLINE_MAX_STRENGTH,
        "online_scale_factor": ONLINE_SCALE_FACTOR,
        "prior_strength": PRIOR_STRENGTH,
        "prior_mean": PRIOR_MEAN,
        "max_neighbor_candidates": MAX_NEIGHBOR_CANDIDATES,
        "demo_odds_margin": DEMO_ODDS_MARGIN,
        "max_single_demo_weight": MAX_SINGLE_DEMO_WEIGHT,
        "data_folder": None,
    }


def get_embeddings(
    encoder,
    data_path: str,
    label_key: str,
    batch_size: int,
    num_workers: int,
    cloud_filename: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    datamodule = GraspPointCloudDataModule(
        dataset_root=data_path,
        label_key=label_key,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        cloud_filename=cloud_filename,
    )
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    train_features, train_labels = collect_embeddings(encoder, train_loader)
    val_features, val_labels = collect_embeddings(encoder, val_loader)

    return train_features, train_labels, val_features, val_labels


def score_average_precision(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    temperature: float,
) -> float:
    score_module = GraspScoreProbabilisticModule(
        **score_module_kwargs(temperature),
    )
    score_module.fit_offline(
        train_features[:N_TRAINING_POINTS],
        train_labels[:N_TRAINING_POINTS],
        persist=False,
    )
    scores = score_module.predict(val_features, estimation_mode="offline")
    return float(average_precision_score(val_labels, scores))


def select_temperature(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[float, float]:
    temperatures = np.arange(
        TEMPERATURE_MIN,
        TEMPERATURE_MAX + 0.5 * TEMPERATURE_STEP,
        TEMPERATURE_STEP,
        dtype=np.float32,
    )

    best_temperature = None
    best_average_precision = None
    for temperature in temperatures:
        average_precision = score_average_precision(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            temperature=float(temperature),
        )
        print(f"temperature={float(temperature):.2f} val_ap={average_precision:.4f}")
        if best_average_precision is None or average_precision > best_average_precision:
            best_temperature = float(temperature)
            best_average_precision = average_precision

    if best_temperature is None or best_average_precision is None:
        raise RuntimeError("Failed to select a temperature.")

    return best_temperature, best_average_precision


def build_cl_module(
    encoder_path: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    temperature: float,
    cloud_filename: str,
) -> ContinualLearningModule:
    encoder_pointcloud = "with_table" if cloud_filename == "cloud_w_table.ply" else "no_table"
    cl_module = ContinualLearningModule(
        encoder_path=encoder_path,
        device="cuda",
        encoder_pointcloud=encoder_pointcloud,
    )
    score_module = GraspScoreProbabilisticModule(
        **score_module_kwargs(temperature),
    )
    score_module.fit_offline(
        train_features[:N_TRAINING_POINTS],
        train_labels[:N_TRAINING_POINTS],
        persist=False,
    )
    cl_module.grasp_scoring_module_cl = score_module
    return cl_module


def main(encoder_path, training_data_path, output_path, cloud_filename="cloud_no_table.ply") -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    encoder_pointcloud = "with_table" if cloud_filename == "cloud_w_table.ply" else "no_table"

    cl_module_for_embeddings = ContinualLearningModule(
        encoder_path=encoder_path,
        encoder_pointcloud=encoder_pointcloud,
    )
    train_features, train_labels, val_features, val_labels = get_embeddings(
        encoder=cl_module_for_embeddings.encoder,
        data_path=training_data_path,
        label_key="success_binary",
        batch_size=1024,
        num_workers=0,
        cloud_filename=cloud_filename,
    )

    print(
        f"Collected embeddings: train={train_features.shape[0]} "
        f"val={val_features.shape[0]} dim={train_features.shape[1]}"
    )

    best_temperature, best_average_precision = select_temperature(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
    )
    print(
        f"Selected temperature={best_temperature:.2f} "
        f"with val_ap={best_average_precision:.4f}"
    )

    cl_module = build_cl_module(
        encoder_path=encoder_path,
        train_features=train_features,
        train_labels=train_labels,
        temperature=best_temperature,
        cloud_filename=cloud_filename,
    )
    cl_module.set_module_root(output_path)
    cl_module.save(output_path)
    cl_module.internalize_encoder()
    ContinualLearningModule.update_training_metadata(output_path, base_train_object_set="base")
    print(f"Saved CL module to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_path",
        default="data/models/embeddings/snn_autoencoder/lightning_logs/version_0/checkpoints/best.ckpt",
    )
    parser.add_argument(
        "--training_data_path",
        default="data/training_data",
    )
    parser.add_argument(
        "--output_path",
        default="data/grasping_models/grasp-model",
    )
    parser.add_argument("--cloud_filename", default="cloud_w_table.ply")
    args = parser.parse_args()
    main(args.encoder_path, args.training_data_path, args.output_path, args.cloud_filename)
