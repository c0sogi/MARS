import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config, set_seed
from library.data_processing import load_and_process_data
from library.train import Trainer
from library.utils import compute_mcc


def main():
    # 1. Configuration and Seeding
    config = Config()
    # Use full dataset to ensure high performance (A100 is fast enough for 3.4M rows with 1D CNN)
    config.debug = False
    set_seed(config.seed)

    # 2. Load Data
    print("Loading and processing training data...")
    train_dataset, _ = load_and_process_data(split="train", config=config)

    print("Loading and processing validation data...")
    val_dataset, val_meta = load_and_process_data(split="validation", config=config)

    # 3. Train Model
    print("Initializing training...")
    trainer = Trainer(config)

    # fit() returns the threshold optimized on the validation set
    best_threshold = trainer.fit(train_dataset, val_dataset)

    # 4. Final Validation Evaluation
    print("Performing final validation evaluation...")
    # Reload the best model weights saved during training
    best_model_path = os.path.join(config.artifact_dir, "best_model.pth")
    trainer.model.load_state_dict(
        torch.load(best_model_path, map_location=trainer.device)
    )
    trainer.model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Re-run validation inference to get raw probabilities
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(trainer.device)
            logits = trainer.model(X)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.numpy())

    y_prob = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Apply optimized threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    final_mcc = compute_mcc(y_true, y_pred_opt)

    # Print exact metric format required
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_true - y_prob)

    # Extract features for correlation analysis
    # The features are now flattened. We need to identify the indices for t=0.
    # Based on data_processing: is_ground is idx 0.
    # Then for each lag k in [-5...5]:
    #   log_dist, rel_speed, closing, s1, s2, a1, a2 (7 features)
    # t=0 corresponds to k=0.

    val_features = val_dataset.features.numpy()

    # Index calculation:
    # 1 (is_ground) + 5 * 7 (lags -5 to -1) = 36.
    # So t=0 starts at index 36.

    start_idx = 1 + 5 * 7
    feature_names_t0 = [
        "log_dist_t0",
        "rel_speed_t0",
        "closing_speed_t0",
        "speed_1_t0",
        "speed_2_t0",
        "accel_1_t0",
        "accel_2_t0",
    ]

    correlations = []

    # Add is_ground
    ground_vals = val_features[:, 0]
    if np.std(ground_vals) > 1e-9:
        corr = np.corrcoef(errors, ground_vals)[0, 1]
        correlations.append(("is_ground", corr))

    # Add t=0 features
    for i, name in enumerate(feature_names_t0):
        idx = start_idx + i
        feat_values = val_features[:, idx]
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            correlations.append((name, corr))
        else:
            correlations.append((name, 0.0))

    # Sort by magnitude of correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Correlation between Error Magnitude and Input Features:")
    for name, corr in correlations:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    THRESHOLD_SCORE = 0.62458462731896

    if final_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation metric {final_mcc} > {THRESHOLD_SCORE}. Generating submission..."
        )

        # Load Test Data
        print("Loading and processing test data...")
        test_dataset, test_meta = load_and_process_data(split="test", config=config)

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Inference
        trainer.model.eval()
        test_preds = []

        with torch.no_grad():
            for X in test_loader:
                X = X.to(trainer.device)
                logits = trainer.model(X)
                probs = torch.sigmoid(logits)
                test_preds.append(probs.cpu().numpy())

        y_prob_test = np.concatenate(test_preds)
        y_pred_test = (y_prob_test >= best_threshold).astype(int)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {"contact_id": test_meta["contact_id"], "contact": y_pred_test}
        )

        # Save
        os.makedirs("./submission", exist_ok=True)
        sub_path = "./submission/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nValidation metric {final_mcc} <= {THRESHOLD_SCORE}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
