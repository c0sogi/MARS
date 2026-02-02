import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import SEED, BATCH_SIZE, WORKING_DIR, SUBMISSION_DIR, METADATA_DIR
from library.utils import seed_everything, load_checkpoint
from library.data_processing import prepare_data
from library.model import HCPFE_Model, generate_submission
from library.training import Trainer


def perform_failure_analysis(model, val_loader, meta, device):
    """
    Analyzes model errors on the validation set.
    Calculates correlation between absolute error and continuous features.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_preds = []
    all_targets = []
    all_cont_inputs = []

    # Collect predictions, targets, and inputs
    with torch.no_grad():
        for cat_x, cont_x, targets in val_loader:
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            targets = targets.to(device)

            outputs = model(cat_x, cont_x)

            # Average probabilities from all streams
            probs = [torch.sigmoid(out) for out in outputs]
            avg_prob = torch.mean(torch.stack(probs, dim=0), dim=0)

            all_preds.append(avg_prob.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_cont_inputs.append(cont_x.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    cont_inputs = np.concatenate(all_cont_inputs, axis=0)

    # Calculate Absolute Error
    errors = np.abs(targets - preds)

    # Create DataFrame for correlation analysis
    cont_cols = meta["cont_cols"]
    df_analysis = pd.DataFrame(cont_inputs, columns=cont_cols)
    df_analysis["error"] = errors

    # Compute correlations
    correlations = (
        df_analysis.corrwith(df_analysis["error"]).abs().sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error:")
    print(correlations.head(6))  # head(6) because 'error' itself will be 1.0

    return correlations


def main():
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Fast Baseline Configuration
    # Increased to 30 epochs to allow full convergence (Lesson 00080)
    FAST_EPOCHS = 30

    # 2. Data Preparation
    print("Preparing data...")
    train_loader, val_loader, test_loader, meta = prepare_data(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing HC-PFE Model...")
    model = HCPFE_Model(meta).to(device)

    # 4. Training
    print(f"Starting training for {FAST_EPOCHS} epochs...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=FAST_EPOCHS,
        save_path=os.path.join(WORKING_DIR, "best_model.pth"),
    )

    best_model_path = trainer.fit()

    # 5. Final Validation Assessment
    print("Loading best model for final validation...")
    best_model = HCPFE_Model(meta).to(device)
    best_model = load_checkpoint(best_model, best_model_path, device)

    # Compute metric on full validation set
    val_auc = trainer.validate()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    perform_failure_analysis(best_model, val_loader, meta, device)

    # 7. Submission Generation
    # Threshold check
    THRESHOLD = 0.9975746465492954

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load test IDs for submission file
        test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
        test_ids = test_df["id"].values

        generate_submission(
            model_path=best_model_path,
            test_loader=test_loader,
            test_ids=test_ids,
            meta=meta,
            device=device,
        )
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
