import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    WORKING_DIR,
    INPUT_DIR,
)
from library.data_loader import prepare_data
from library.model import DSRPEModel
from library.trainer import train_one_epoch, validate, set_seed

# Override EPOCHS for fast baseline execution
FAST_RUN_EPOCHS = 5
SUBMISSION_THRESHOLD = 0.9975746465492954


def perform_failure_analysis(model, val_loader, device, vocab_sizes, cont_dim):
    """
    Analyzes model failure modes by correlating error magnitude with input features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_cat = []
    all_cont = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for cat_inputs, cont_inputs, targets in val_loader:
            # Store inputs for correlation analysis
            all_cat.append(cat_inputs.cpu().numpy())
            all_cont.append(cont_inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Inference
            cat_inputs = cat_inputs.to(device)
            cont_inputs = cont_inputs.to(device)

            outputs_list = model(cat_inputs, cont_inputs)

            # Ensemble mean
            stream_probs = [torch.sigmoid(out) for out in outputs_list]
            avg_probs = torch.stack(stream_probs).mean(dim=0)
            all_preds.append(avg_probs.cpu().numpy())

    # Concatenate all batches
    cat_matrix = np.concatenate(all_cat, axis=0)
    cont_matrix = np.concatenate(all_cont, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0).flatten()

    # Calculate Error
    errors = np.abs(y_true - y_pred)

    # Analyze Correlations
    # We correlate the error with each feature column.
    # For categorical features, we use the integer indices as a proxy (or could decode, but indices suffice for ID)

    feature_correlations = {}

    # Categorical Features
    for i in range(cat_matrix.shape[1]):
        feat_name = f"cat_feat_{i}"
        corr = np.corrcoef(cat_matrix[:, i], errors)[0, 1]
        feature_correlations[feat_name] = corr

    # Continuous Features
    for i in range(cont_matrix.shape[1]):
        feat_name = f"cont_feat_{i}"
        corr = np.corrcoef(cont_matrix[:, i], errors)[0, 1]
        feature_correlations[feat_name] = corr

    # Sort by absolute correlation
    sorted_feats = sorted(
        feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True
    )

    print("Top 10 Features correlated with Error Magnitude:")
    for name, corr in sorted_feats[:10]:
        print(f"{name}: {corr:.6f}")

    return y_true, y_pred


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    print("Loading data...")
    train_ds, val_ds, test_ds, vocab_sizes = prepare_data(load_cached_data=True)

    # Check dimensions
    _, sample_cont, _ = train_ds[0]
    cont_dim = sample_cont.shape[0]

    # Create Loaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 3. Model Initialization
    print(f"Initializing model (Vocab: {vocab_sizes}, Cont Dim: {cont_dim})...")
    model = DSRPEModel(vocab_sizes=vocab_sizes, cont_dim=cont_dim)
    model.to(DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=FAST_RUN_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 5. Training Loop
    print(f"Starting training for {FAST_RUN_EPOCHS} epochs...")
    best_auc = 0.0

    for epoch in range(FAST_RUN_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{FAST_RUN_EPOCHS} | Train Loss: {train_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc

    # 6. Final Validation & Failure Analysis
    print("\nTraining complete. Running full validation assessment...")
    y_true, y_pred = perform_failure_analysis(
        model, val_loader, DEVICE, vocab_sizes, cont_dim
    )

    final_auc = roc_auc_score(y_true, y_pred)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Submission
    if final_auc > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model.eval()
        all_test_preds = []

        with torch.no_grad():
            for cat_inputs, cont_inputs in test_loader:
                cat_inputs = cat_inputs.to(DEVICE)
                cont_inputs = cont_inputs.to(DEVICE)

                outputs_list = model(cat_inputs, cont_inputs)

                # Ensemble mean
                stream_probs = [torch.sigmoid(out) for out in outputs_list]
                avg_probs = torch.stack(stream_probs).mean(dim=0)
                all_test_preds.append(avg_probs.cpu().numpy())

        predictions = np.concatenate(all_test_preds).flatten()

        # Load sample submission
        sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
        submission_df = pd.read_csv(sample_sub_path)

        # Align lengths if necessary
        if len(predictions) != len(submission_df):
            print(
                f"Adjusting submission length from {len(submission_df)} to {len(predictions)}"
            )
            submission_df = submission_df.iloc[: len(predictions)]

        submission_df["target"] = predictions

        output_dir = "./submission"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "submission.csv")

        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    else:
        print(
            f"\nValidation metric ({final_auc}) did not meet threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
