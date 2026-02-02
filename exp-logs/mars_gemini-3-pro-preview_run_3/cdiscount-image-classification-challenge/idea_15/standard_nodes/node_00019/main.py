import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.feature_engine import FeatureEngine
from library.feature_dataset import RamFeatureDataset
from library.model import DeepFeatureCascade
from library.trainer import ModelTrainer
from library.inference import EnsemblePredictor


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Override Config defaults to ensure the script completes within the time limit.
    # We use a subset of data (100k samples) and fewer epochs.
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 100000
    Config.EPOCHS = 5
    # Ensure batch size is efficient
    Config.TRAIN_BATCH_SIZE = 4096

    # Create necessary directories
    Config.make_dirs()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print("=== Starting Fast Baseline Pipeline ===")
    print(
        f"Configuration: DEBUG={Config.DEBUG}, SAMPLES={Config.DEBUG_SAMPLES}, EPOCHS={Config.EPOCHS}"
    )

    # ==========================================
    # 2. Feature Extraction
    # ==========================================
    # Extracts features from BSON images using frozen backbones.
    # Checks for existing cache to avoid redundant computation.
    print("\n[Step 1/5] Feature Extraction...")
    engine = FeatureEngine()
    engine.extract_features(load_cached_data=True)

    # ==========================================
    # 3. Data Loading
    # ==========================================
    print("\n[Step 2/5] Loading Datasets...")
    # Load features into RAM
    train_dataset = RamFeatureDataset(
        feature_path=Config.TRAIN_FEATURES, label_path=Config.TRAIN_LABELS, mode="train"
    )
    val_dataset = RamFeatureDataset(
        feature_path=Config.VAL_FEATURES, label_path=Config.VAL_LABELS, mode="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ==========================================
    # 4. Model Training
    # ==========================================
    print("\n[Step 3/5] Training Model...")
    model = DeepFeatureCascade()
    trainer = ModelTrainer(model, train_loader, val_loader)
    trainer.train()

    # ==========================================
    # 5. Validation & Failure Analysis
    # ==========================================
    print("\n[Step 4/5] Final Validation & Failure Analysis...")

    # Load best model for evaluation
    best_model = DeepFeatureCascade()
    best_model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT))
    best_model.to(Config.DEVICE)
    best_model.eval()

    correct = 0
    total = 0

    # Storage for failure analysis
    error_magnitudes = []
    feature_norms = []

    with torch.no_grad():
        for features, l1, l2, l3 in val_loader:
            features = features.to(Config.DEVICE)
            l3 = l3.to(Config.DEVICE)

            # Forward pass
            _, _, l3_logits = best_model(features)

            # Predictions
            preds = torch.argmax(l3_logits, dim=1)

            # Update Accuracy
            correct += (preds == l3).sum().item()
            total += l3.size(0)

            # Failure Analysis: Calculate Error Magnitude
            # Error = 1.0 - Probability of True Class
            probs = torch.softmax(l3_logits, dim=1)
            # Gather probabilities of the ground truth classes
            true_class_probs = probs.gather(1, l3.view(-1, 1)).squeeze()
            batch_errors = 1.0 - true_class_probs

            error_magnitudes.extend(batch_errors.cpu().numpy())

            # Feature Analysis: Calculate L2 Norm of input features
            batch_norms = torch.norm(features, p=2, dim=1)
            feature_norms.extend(batch_norms.cpu().numpy())

    final_acc = correct / total

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis: Correlation
    if len(error_magnitudes) > 1:
        correlation = np.corrcoef(error_magnitudes, feature_norms)[0, 1]
        print(f"Correlation between Error Magnitude and Feature Norm: {correlation}")
    else:
        print("Insufficient data for correlation analysis.")

    # ==========================================
    # 6. Submission
    # ==========================================
    print("\n[Step 5/5] Submission Check...")
    threshold = 0.6239621493939094

    if final_acc > threshold:
        print(
            f"Validation metric ({final_acc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Initialize predictor with the best model
        predictor = EnsemblePredictor(model_paths=[Config.MODEL_CHECKPOINT])

        # Generate submission file
        # Note: If DEBUG is True, this generates predictions for the subset of test data processed.
        # For a real submission, one would run with DEBUG=False.
        predictor.generate_submission()
    else:
        print(
            f"Validation metric ({final_acc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
