import os
import sys
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, FocalLoss
from library.data_processing import process_data
from library.dataset import get_dataloader
from library.models import SRVNet
from library.train import train_one_epoch, evaluate
from library.inference import generate_predictions


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Overrides for Fast Baseline Execution
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 2048
    TRAIN_SAMPLE_SIZE = 300000  # Subsample training data for speed

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    # Load Training Data
    X_kin_train, X_vis_train, y_train, _ = process_data("train", load_cached_data=True)

    # Subsample Training Data
    if len(y_train) > TRAIN_SAMPLE_SIZE:
        indices = np.random.choice(len(y_train), TRAIN_SAMPLE_SIZE, replace=False)
        X_kin_train = X_kin_train[indices]
        X_vis_train = X_vis_train[indices]
        y_train = y_train[indices]

    train_loader = get_dataloader(
        X_kin_train, X_vis_train, y_train, batch_size=Config.BATCH_SIZE, shuffle=True
    )

    # Load Validation Data (Full set for accurate metrics)
    X_kin_val, X_vis_val, y_val, _ = process_data("validation", load_cached_data=True)

    val_loader = get_dataloader(
        X_kin_val, X_vis_val, y_val, batch_size=Config.BATCH_SIZE * 2, shuffle=False
    )

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    model = SRVNet(
        input_dim_kin=Config.INPUT_DIM_KINEMATIC,
        input_dim_vis=Config.INPUT_DIM_VISUAL,
        kinematic_hidden_dims=Config.KINEMATIC_HIDDEN_DIMS,
        visual_hidden_dims=Config.VISUAL_HIDDEN_DIMS,
        dropout_rate=Config.DROPOUT_RATE,
        lambda_visual=Config.LAMBDA_VISUAL,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = FocalLoss(gamma=Config.FOCAL_LOSS_GAMMA)

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_val_mcc = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mcc, val_thresh = evaluate(model, val_loader, criterion, device)

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            torch.save(model.state_dict(), best_model_path)

    # =========================================================================
    # 5. Final Metrics
    # =========================================================================
    # Required Output Format
    print(f"Final Validation Metric: {best_val_mcc}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for x_kin, x_vis, _ in val_loader:
            x_kin = x_kin.to(device)
            x_vis = x_vis.to(device)
            logits = model(x_kin, x_vis)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    y_prob = np.concatenate(all_probs).flatten()
    y_true = y_val.flatten()

    # Calculate absolute error
    errors = np.abs(y_true - y_prob)

    # Correlate error with Kinematic Features at t0 (center of window)
    # Features are flattened: [t-5, ..., t0, ..., t+5]
    kin_feats = Config.KINEMATIC_FEATURES_SINGLE_STEP
    center_idx = Config.WINDOW_SIZE
    start_col = center_idx * len(kin_feats)
    end_col = start_col + len(kin_feats)

    X_kin_t0 = X_kin_val[:, start_col:end_col]

    correlations = []
    for i, name in enumerate(kin_feats):
        feat_vals = X_kin_t0[:, i]
        # Avoid division by zero in correlation
        if np.std(feat_vals) > 1e-9:
            corr = np.corrcoef(feat_vals, errors)[0, 1]
            correlations.append((name, corr))
        else:
            correlations.append((name, 0.0))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nFailure Analysis - Top Correlations with Error (t0 features):")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    THRESHOLD_SCORE = 0.6634847318478787

    if best_val_mcc > THRESHOLD_SCORE:
        generate_predictions(load_cached_data=True)
    else:
        print(
            f"Validation MCC ({best_val_mcc}) did not meet threshold ({THRESHOLD_SCORE}). Skipping submission."
        )


if __name__ == "__main__":
    main()
