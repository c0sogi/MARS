import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import Config
from library.feature_extractor import run_feature_extraction
from library.dataset import FeatureDataset, MixupCollate
from library.model import HierarchicalMultiTaskNetwork, HierarchicalTrainer


def main():
    # 1. Setup Configuration and Environment
    Config.setup()

    # 2. Feature Extraction
    # Checks for cached features in ./working/idea_8.
    # If missing, runs full ResNet-50 inference on Train/Val/Test BSON files.
    # This ensures we use the full dataset as per the strategy.
    run_feature_extraction(load_cached_data=True)

    # 3. Data Loading
    print("Initializing Datasets...")

    # Load Train Features into Memory
    train_ds = FeatureDataset(
        Config.TRAIN_FEATURES_PATH,
        Config.TRAIN_LABELS_PATH,
        limit=Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None,
        load_in_memory=True,
    )

    # Use MixUp Collate for Training Regularization
    mixup_collate = MixupCollate(alpha=Config.MIXUP_ALPHA)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=mixup_collate,
        pin_memory=True,
    )

    # Load Validation Features into Memory
    val_ds = FeatureDataset(
        Config.VAL_FEATURES_PATH,
        Config.VAL_LABELS_PATH,
        limit=Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None,
        load_in_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization & Training
    print("Initializing Hierarchical Multi-Task Network...")
    model = HierarchicalMultiTaskNetwork()
    trainer = HierarchicalTrainer(model)

    print(f"Starting Training for {Config.EPOCHS} epochs...")
    # Fits the model, saves the best checkpoint, and reloads best weights at the end.
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 5. Validation Assessment
    print("Computing Final Validation Metric...")
    final_metric = trainer.validate(val_loader)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()
    all_preds = []
    all_targets = []
    all_feature_norms = []

    device = Config.DEVICE

    # Manual inference loop on validation set to gather stats
    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch
            features = features.to(device)

            # Target is Level 3 (Fine-grained)
            target_l3 = labels[:, 2].to(device)

            # Forward pass
            _, _, logits_l3 = model(features)
            preds = torch.argmax(logits_l3, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target_l3.cpu().numpy())

            # Compute L2 norm of input features (signal magnitude)
            norms = torch.norm(features, dim=1).cpu().numpy()
            all_feature_norms.append(norms)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_feature_norms = np.concatenate(all_feature_norms)

    # Calculate binary error (1 = Incorrect, 0 = Correct)
    errors = (all_preds != all_targets).astype(int)

    # Calculate correlation between Error and Feature Norm
    if len(errors) > 1:
        correlation = np.corrcoef(errors, all_feature_norms)[0, 1]
    else:
        correlation = 0.0

    print(f"Error-Feature Norm Correlation: {correlation}")

    # 7. Submission Generation
    THRESHOLD = 0.6239621493939094

    if final_metric > THRESHOLD:
        print(
            f"Validation metric {final_metric} > {THRESHOLD}. Generating submission..."
        )
        trainer.predict_submission(
            Config.TEST_FEATURES_PATH, Config.TEST_IDS_PATH, Config.SUBMISSION_PATH
        )
    else:
        print(f"Validation metric {final_metric} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
