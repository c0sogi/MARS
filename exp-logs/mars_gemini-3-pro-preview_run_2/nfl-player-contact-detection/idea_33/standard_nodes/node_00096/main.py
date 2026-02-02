import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from library import config, utils, data_processor, model, trainer


def main():
    # Ensure reproducibility
    utils.seed_everything()

    # Detect device
    device = torch.device(config.DEVICE)

    # -------------------------------------------------------------------------
    # 1. Train Model
    # -------------------------------------------------------------------------
    # We use the full dataset (debug_sample=None) to ensure we validate on the
    # entire hold-out set as required. The efficient MLP architecture allows
    # full training on 3.4M rows within the time limit on an A100.
    trainer.train_model(debug_sample=None)

    # -------------------------------------------------------------------------
    # 2. Validation & Evaluation
    # -------------------------------------------------------------------------
    # Load validation data (cached from the training step)
    # We ignore train/test loaders here, just need validation
    _, val_loader, _, _ = data_processor.prepare_datasets(load_cached_data=True)

    # Retrieve feature names from the cached schema to match model input
    train_parquet_path = os.path.join(config.WORKING_DIR, "train_features.parquet")
    df_schema = pd.read_parquet(train_parquet_path).head(0)
    feature_names = data_processor.get_feature_columns(df_schema.columns)

    # Initialize model structure
    net = model.IPRVN(feature_names).to(device)

    # Load best model weights saved during training
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Best model not found. Training may have failed.")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Load optimized threshold
    thresh_path = os.path.join(config.WORKING_DIR, "best_threshold.npy")
    if os.path.exists(thresh_path):
        best_thresh = float(np.load(thresh_path)[0])
    else:
        best_thresh = 0.5

    # Run Inference on Validation Set
    all_probs = []
    all_targets = []
    all_features = []

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = net(features)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            # Collect features for failure analysis
            all_features.append(features.cpu().numpy())

    y_probs = np.vstack(all_probs).flatten()
    y_true = np.concatenate(all_targets)
    X_val = np.vstack(all_features)

    # Calculate Final Metric
    y_pred = (y_probs >= best_thresh).astype(int)
    final_mcc = matthews_corrcoef(y_true, y_pred)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_mcc}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    # Calculate error magnitude
    errors = np.abs(y_true - y_probs)

    # Calculate correlation between error and features
    correlations = []
    num_features = X_val.shape[1]

    for i in range(num_features):
        feat_vals = X_val[:, i]
        # Avoid correlation on constant features
        if np.std(feat_vals) > 1e-6:
            # Pearson correlation
            corr = np.corrcoef(errors, feat_vals)[0, 1]
            correlations.append((feature_names[i], corr))
        else:
            correlations.append((feature_names[i], 0.0))

    # Sort by magnitude of correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    TARGET_METRIC = 0.6634847318478787

    if final_mcc > TARGET_METRIC:
        trainer.predict()
    else:
        print(
            f"Validation Metric {final_mcc} did not exceed threshold {TARGET_METRIC}. Submission skipped."
        )


if __name__ == "__main__":
    main()
