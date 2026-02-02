import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config, set_seed
from library.model_utils import SSDeGUT
from library.data_utils import get_dataloaders, feature_engineering
from library.train_utils import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    # Create necessary directories
    Config.setup()

    # Set random seeds for reproducibility
    set_seed(Config.SEED)

    # Override Config for Optimization
    # We increase epochs to 45 to ensure full convergence with the larger model and batch size.
    Config.EPOCHS = 45

    print(
        f"Configuration: Device={Config.DEVICE}, Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[Data] Initializing Dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # Inspect a batch to determine input dimensions dynamically
    sample_batch = next(iter(train_loader))
    # x_num shape: (Batch, Num_Features)
    num_numerical_features = sample_batch["x_num"].shape[1]
    print(f"[Data] Detected {num_numerical_features} numerical features.")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[Model] Initializing SS-DeGUT Model...")
    model = SSDeGUT(Config, num_numerical_features=num_numerical_features)
    model.to(Config.DEVICE)

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("\n[Train] Starting Training...")
    trainer = Trainer(model, train_loader, val_loader, test_loader, Config)
    trainer.fit()

    # ---------------------------------------------------------
    # 5. Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\n[Validation] Performing Final Assessment & Failure Analysis...")

    # Load the best model weights saved during training
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading best model from {Config.MODEL_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    else:
        print("Warning: Best model file not found. Using current model state.")

    model.eval()

    val_preds = []
    val_targets = []
    val_inputs = []  # Store standardized inputs for correlation analysis

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            x_num = batch["x_num"].to(Config.DEVICE)
            x_seq = batch["x_seq"].to(Config.DEVICE)
            y = batch["target"].to(Config.DEVICE)

            # Forward pass (Inference Mode: mask_ratio=0.0)
            outputs = model(x_num, x_seq, mask_ratio=0.0)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(y.cpu().numpy().flatten())
            val_inputs.append(x_num.cpu().numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_inputs = np.vstack(val_inputs)

    # Calculate Final Metric
    final_metric = roc_auc_score(val_targets, val_preds)
    # Print Full Precision Metric as required
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Input Features
    print("\n[Analysis] Calculating Error Correlations...")
    error_magnitude = np.abs(val_targets - val_preds)

    # Retrieve Feature Names for interpretability
    # We load a small sample of metadata and apply feature engineering to get the column list
    try:
        df_meta_sample = pd.read_csv(Config.VAL_META_PATH, nrows=5)
        df_meta_sample = feature_engineering(df_meta_sample)
        exclude_cols = ["id", "target", "source_path", "f_27"]
        feature_names = sorted(
            [c for c in df_meta_sample.columns if c not in exclude_cols]
        )
    except Exception as e:
        print(f"Could not retrieve feature names: {e}")
        feature_names = [f"feat_{i}" for i in range(num_numerical_features)]

    correlations = []
    for i in range(val_inputs.shape[1]):
        if i < len(feature_names):
            feat_name = feature_names[i]
        else:
            feat_name = f"feat_{i}"

        feat_values = val_inputs[:, i]

        # Calculate correlation (handle constant features)
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, error_magnitude)[0, 1]

        correlations.append((feat_name, corr))

    # Sort by absolute correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.6f}")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 0.9977872734278943

    if final_metric > THRESHOLD:
        print(
            f"\n[Submission] Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\n[Submission] Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
