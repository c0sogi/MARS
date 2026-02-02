import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.data import get_dataloaders
from library.train import Trainer
from library.inference import generate_submission
from library.model import DANRegressor


def run():
    # 1. Set Reproducibility
    seed_everything(Config.SEED)

    # 2. Configure for Fast Baseline
    # The prompt requires limiting training steps/epochs for a quick baseline.
    # DAN is fast, but we'll reduce epochs to ensure it finishes very quickly.
    Config.NUM_EPOCHS = 10
    print(f"Configuration set for fast baseline: NUM_EPOCHS={Config.NUM_EPOCHS}")

    # 3. Prepare Data
    # Load cached data if available, otherwise process from scratch
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        load_cached_data=True
    )

    # 4. Train Model
    print("Initializing Trainer...")
    trainer = Trainer(Config)

    print("Starting Training...")
    trainer.fit(train_loader, val_loader)

    # 5. Final Validation and Metric Reporting
    print("Performing Final Validation...")

    # Load the best saved model
    device = torch.device(Config.DEVICE)
    model = DANRegressor(Config).to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Model save path not found, using current model state.")
        model = trainer.model

    model.eval()

    all_preds = []
    all_labels = []
    all_lengths = []  # For failure analysis

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            scores = batch["scores"].to(device)

            # Forward pass
            outputs = model(input_ids).squeeze(-1)

            # Collect data
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(scores.cpu().numpy())

            # Calculate sequence length (non-padding tokens) for failure analysis
            # Padding index is 0
            lengths = (input_ids != 0).sum(dim=1).cpu().numpy()
            all_lengths.extend(lengths)

    # Process predictions
    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)

    # Clip and round for QWK calculation
    preds_clipped = np.clip(preds_np, 1, 6)
    preds_rounded = np.round(preds_clipped).astype(int)
    labels_int = labels_np.astype(int)

    # Compute Metric
    qwk = compute_qwk(labels_int, preds_rounded)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {qwk}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(labels_np - preds_np)
    lengths_np = np.array(all_lengths)

    # Correlation between Error and Input Length
    if len(errors) > 1:
        corr, _ = pearsonr(errors, lengths_np)
        print(
            f"Correlation between Absolute Error and Essay Length (Token Count): {corr:.4f}"
        )

        # Additional insight: Mean error by score class
        df_analysis = pd.DataFrame(
            {
                "true_score": labels_int,
                "prediction": preds_np,
                "rounded_pred": preds_rounded,
                "error": errors,
            }
        )
        print("\nMean Absolute Error by True Score:")
        print(df_analysis.groupby("true_score")["error"].mean())
    else:
        print("Not enough data for failure analysis.")

    # 7. Generate Submission
    print("\nGenerating Submission...")
    # generate_submission handles loading the model and test data internally
    # We pass load_cached_data=True to reuse the processing done in get_dataloaders
    generate_submission(load_cached_data=True)

    print("Runfile execution completed.")


if __name__ == "__main__":
    run()
