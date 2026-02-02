import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import utils
from library import dataset
from library import feature_extractor
from library import engine
from library import model as lib_model


def main():
    # 1. Setup & Reproducibility
    engine.set_seed(config.SEED)

    # 2. Feature Extraction
    # Extracts features from train, val, and test BSON files using frozen ResNet-50.
    # Caches results to disk/RAM to allow fast MLP training.
    # We use the full dataset as per the "Idea 5" specification to ensure sufficient data for the high-dim features.
    extractor = feature_extractor.FeatureExtractor()
    extractor.run_all(load_cached_data=True)

    # 3. Training
    # Trains the Hierarchical MLP on the extracted features.
    # Returns the best validation accuracy achieved during training.
    print("Starting training process...")
    best_val_acc = engine.fit()

    # 4. Final Validation & Failure Analysis
    print("Performing Final Validation and Failure Analysis...")

    # Load validation data explicitly for detailed analysis
    val_ds = dataset.EmbeddingDataset(
        config.VAL_FEATURES_PATH,
        config.VAL_LABELS_L1_PATH,
        config.VAL_LABELS_L2_PATH,
        config.VAL_LABELS_L3_PATH,
        mode="val",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Load the best trained model
    encoder = utils.HierarchyEncoder()
    encoder.prepare()

    model = lib_model.HierarchicalMLP(encoder.num_l1, encoder.num_l2, encoder.num_l3)
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model file not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    all_preds = []
    all_targets = []
    all_features = []

    # Run Inference on Validation Set
    with torch.no_grad():
        for features, _, _, l3 in val_loader:
            features_dev = features.to(config.DEVICE)
            l3 = l3.to(config.DEVICE)

            # Forward pass
            _, _, out_l3 = model(features_dev)
            _, preds = torch.max(out_l3, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(l3.cpu().numpy())
            # Keep features on CPU to save GPU memory during analysis accumulation
            all_features.append(features.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_features = np.concatenate(all_features, axis=0)

    # Calculate Metric
    accuracy = np.mean(all_preds == all_targets)
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    # 1. Calculate Error Vector (0 for correct, 1 for error)
    errors = (all_preds != all_targets).astype(int)

    # 2. Correlation with Feature Norm (Input Signal Strength)
    # We hypothesize that images with lower feature norms (less activation) might be harder to classify.
    feature_norms = np.linalg.norm(all_features, axis=1)
    corr_norm = np.corrcoef(errors, feature_norms)[0, 1]
    print(
        f"Failure Analysis: Correlation between Error and Feature Norm: {corr_norm:.4f}"
    )

    # 3. Correlation with Class Frequency (Imbalance Impact)
    # We check if rare classes have higher error rates.
    # Map targets to their frequency in the validation set
    class_counts = pd.Series(all_targets).value_counts()
    sample_counts = class_counts.loc[all_targets].values
    corr_freq = np.corrcoef(errors, sample_counts)[0, 1]
    print(
        f"Failure Analysis: Correlation between Error and Class Frequency: {corr_freq:.4f}"
    )

    # 5. Submission
    # Generate submission only if the model meets the performance threshold.
    THRESHOLD = 0.50636

    if accuracy > THRESHOLD:
        print(f"Validation metric {accuracy} > {THRESHOLD}. Generating submission...")
        engine.generate_submission()
    else:
        print(f"Validation metric {accuracy} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
