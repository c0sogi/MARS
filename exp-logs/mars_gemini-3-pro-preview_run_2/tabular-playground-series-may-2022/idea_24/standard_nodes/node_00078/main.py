import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.dataset import get_dataloaders, set_seed
from library.model import GatedStemHybridNet
from library.trainer import Trainer


def main():
    # 1. Setup
    # --------------------------------------------------------------------------
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Using full dataset as per Idea description (Lesson 58)
    # 40 epochs on 640k rows is feasible on A100 within 2 hours.
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=1024,
        load_cached_data=True,
        max_samples=None,  # Use full dataset
        data_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        cache_dir=os.path.join(WORKING_DIR, "idea_24"),
    )

    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing model...")
    model = GatedStemHybridNet(
        continuous_dim=30,
        vocab_size=30,  # Safe upper bound for 'A'-'Z'
        seq_len=10,
        embed_dim=32,
        backbone_dropout=0.35,
    )

    # 4. Training
    # --------------------------------------------------------------------------
    # Configuration from Idea:
    # Optimizer: AdamW, weight_decay=1e-2
    # Scheduler: StepLR, step_size=10, gamma=0.1
    # Epochs: 40
    print("Starting training...")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-2,
        step_size=10,
        gamma=0.1,
    )

    trainer.fit(train_loader, val_loader, epochs=40, checkpoint_dir=WORKING_DIR)

    # 5. Validation & Metrics
    # --------------------------------------------------------------------------
    print("Performing final validation...")

    # Reload best model for validation inference
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model checkpoint not found.")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    val_targets = []
    val_preds = []
    val_inputs_cont = []

    # Collect validation data for metric and failure analysis
    # We iterate manually to gather inputs for failure analysis
    with torch.no_grad():
        for x_cont, x_seq, targets in val_loader:
            x_cont = x_cont.to(device)
            x_seq = x_seq.to(device)

            logits = model(x_cont, x_seq)
            probs = torch.sigmoid(logits)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(targets.numpy())
            val_inputs_cont.append(x_cont.cpu().numpy())

    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_inputs_cont = np.concatenate(val_inputs_cont, axis=0)

    final_auc = roc_auc_score(val_targets, val_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame for correlation analysis
    # Columns f_00 to f_30 (excluding f_27)
    feat_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    # Ensure dimensions match (val_inputs_cont might be slightly smaller if drop_last was used,
    # but val_loader usually doesn't drop last. Check shapes.)
    if val_inputs_cont.shape[0] == errors.shape[0]:
        df_analysis = pd.DataFrame(val_inputs_cont, columns=feat_cols)
        df_analysis["error"] = errors

        # Calculate correlation of features with error
        correlations = (
            df_analysis.corr()["error"].drop("error").abs().sort_values(ascending=False)
        )

        print("Top 5 features correlated with prediction error:")
        print(correlations.head(5))
    else:
        print("Shape mismatch preventing detailed failure analysis.")

    # 7. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9970005855169476

    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = trainer.predict(test_loader, checkpoint_path=best_model_path)

        submission_df = pd.DataFrame({"id": test_ids, "target": test_preds})

        save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
