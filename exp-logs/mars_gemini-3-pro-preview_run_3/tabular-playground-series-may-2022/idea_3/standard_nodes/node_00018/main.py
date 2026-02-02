import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device, load_checkpoint, compute_auc
from library.data import preprocess_features, ManufacturingDataset
from library.model import ManufacturingClassifier, Encoder
from library.train import train_classifier, generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Use full epochs for optimal performance (Cite solution_lesson_node_00002)
    Config.FINETUNE_EPOCHS = 30

    # Setup directories
    Config.setup()

    # Set reproducible seeds
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading and processing data...")
    # Load cached data if available to save time
    train_data, val_data, test_data, metadata = preprocess_features(
        load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model Training (Direct Supervised)
    # -------------------------------------------------------------------------

    # Direct Supervised Training (Cite solution_lesson_node_00013)
    print("\n--- Direct Supervised Training ---")
    # train_classifier saves the best model to Config.BEST_MODEL_PATH
    _ = train_classifier(train_data, val_data, metadata)

    # -------------------------------------------------------------------------
    # 4. Validation Inference & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n--- Validation & Failure Analysis ---")
    device = get_device()

    # Re-initialize the model architecture for inference
    encoder_eval = Encoder(metadata["cont_dim"], metadata["cat_cardinalities"])
    model = ManufacturingClassifier(encoder_eval)

    # Load the best model weights saved during fine-tuning
    load_checkpoint(Config.BEST_MODEL_PATH, model, device=device)
    model.to(device)
    model.eval()

    # Prepare Validation DataLoader
    ds_val = ManufacturingDataset(
        val_data["cont"], val_data["cat"], val_data["target"], mode="supervised"
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run Inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (cont, cat), target in val_loader:
            cont, cat = cont.to(device), cat.to(device)
            logits = model(cont, cat)
            probs = torch.sigmoid(logits).squeeze()

            all_preds.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Compute and Print Final Metric
    final_metric = compute_auc(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)

    print("\nFailure Analysis - Feature Correlations with Error:")

    # 1. Continuous Features Correlation
    cont_features = val_data["cont"]
    cont_cols = metadata["cont_cols"]

    for i, col_name in enumerate(cont_cols):
        feat_vals = cont_features[:, i]
        # Avoid warning if std dev is 0
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        print(f"{col_name}: {corr:.4f}")

    # 2. Categorical Features Correlation
    cat_features = val_data["cat"]
    cat_cols = metadata["cat_cols"]

    for i, col_name in enumerate(cat_cols):
        feat_vals = cat_features[:, i].astype(float)
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
        print(f"{col_name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    TARGET_THRESHOLD = 0.9971550270448856

    if final_metric > TARGET_THRESHOLD:
        print(
            f"\nValidation metric {final_metric} exceeds threshold {TARGET_THRESHOLD}."
        )
        print("Generating submission...")
        generate_submission(test_data, metadata)
    else:
        print(
            f"\nValidation metric {final_metric} does not exceed threshold {TARGET_THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
