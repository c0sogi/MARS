import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.engine import train_model, infer
from library.model import SiameseDeberta
from library.utils import seed_everything


def run():
    # 1. Setup and Configuration
    print("Initializing pipeline...")
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution (1 epoch fits within 2 hours on A100)
    # Config.EPOCHS = 1

    # Ensure device is set correctly
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Preparing data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load dataloaders (uses caching mechanism)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=True
    )

    # 3. Model Training
    print(f"Starting training for {Config.EPOCHS} epoch(s)...")
    # train_model handles the training loop, validation monitoring, and saving the best model
    best_log_loss = train_model(train_loader, val_loader)
    print(f"Training completed. Best Log Loss reported by engine: {best_log_loss}")

    # 4. Validation Inference & Metric Calculation
    print("Running full validation inference for final metric and analysis...")

    # Load the best saved model
    model = SiameseDeberta()
    model.to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model file not found.")
        return

    model.eval()

    val_preds = []
    val_targets = []

    # Inference loop on validation set
    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            meta_features = batch["meta_features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                meta_features,
            )
            probs = torch.softmax(logits, dim=1)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate and print the required metric
    final_metric = log_loss(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate per-sample Cross Entropy Loss (Error Magnitude)
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    per_sample_loss = -np.sum(val_targets * np.log(val_preds_clipped), axis=1)

    # Load processed validation data to get meta-features
    # The cache file is created by get_dataloaders -> _process_and_cache_data
    val_cache_path = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    if os.path.exists(val_cache_path):
        df_val = pd.read_parquet(val_cache_path)

        # Meta-features in cache are normalized: 'meta_0' (Prompt), 'meta_1' (Res A), 'meta_2' (Res B)
        features_to_analyze = [
            ("meta_0", "Prompt Length (Norm)"),
            ("meta_1", "Response A Length (Norm)"),
            ("meta_2", "Response B Length (Norm)"),
        ]

        print("Correlation between Error Magnitude (Log Loss) and Input Features:")
        for col, name in features_to_analyze:
            if col in df_val.columns:
                feat_values = df_val[col].values
                # Ensure shapes match
                if len(feat_values) == len(per_sample_loss):
                    corr = np.corrcoef(feat_values, per_sample_loss)[0, 1]
                    print(f"  {name}: {corr:.4f}")
                else:
                    print(
                        f"  {name}: Shape mismatch ({len(feat_values)} vs {len(per_sample_loss)})"
                    )
    else:
        print(
            "Warning: Validation cache file not found. Skipping feature correlation analysis."
        )

    # 6. Submission Generation
    # Condition: Final Validation Metric < 1.0102717496437368
    threshold = 1.0102717496437368

    if final_metric < threshold:
        print(
            f"\nMetric condition met ({final_metric} < {threshold}). Generating submission..."
        )

        ids, preds = infer(test_loader)

        submission = pd.DataFrame(
            {
                "id": ids.astype(int),
                "winner_model_a": preds[:, 0],
                "winner_model_b": preds[:, 1],
                "winner_tie": preds[:, 2],
            }
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Verify submission format
        print("Submission head:")
        print(submission.head())
    else:
        print(
            f"\nMetric condition NOT met ({final_metric} >= {threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run()
