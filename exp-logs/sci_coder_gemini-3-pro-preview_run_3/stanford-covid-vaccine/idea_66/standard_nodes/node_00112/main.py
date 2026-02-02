import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure library can be imported from current directory
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.loss import MCRMSELoss
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, inference, generate_submission


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Limit epochs to ensure execution within time limits while allowing convergence
    Config.MAX_EPOCHS = 15

    # ==========================================
    # 2. Setup
    # ==========================================
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 3. Data Loading
    # ==========================================
    # load_cached_data=True utilizes pre-processed .npz files in ./working/cache
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    model = RNAModel().to(device)

    # ==========================================
    # 5. Training Setup
    # ==========================================
    criterion = MCRMSELoss()
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCHS)

    # ==========================================
    # 6. Training Loop
    # ==========================================
    best_score = float("inf")

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)
        scheduler.step()

        # Save best model based on competition metric
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # ==========================================
    # 7. Final Validation Metric
    # ==========================================
    print(f"Final Validation Metric: {best_score}")

    # ==========================================
    # 8. Failure Analysis
    # ==========================================
    # Load best model for analysis
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Collect predictions and targets for the entire validation set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"]
            ids = batch["id"]

            preds = model(features, pair_indices, pair_masks)
            all_preds.append(preds.cpu())
            all_targets.append(targets)
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute MCRMSE per sample
    # 1. Slice to scored sequence length (68)
    seq_scored = Config.SEQ_SCORED
    # 2. Filter for scoring columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scoring_indices = [0, 1, 3]

    preds_sliced = all_preds[:, :seq_scored, scoring_indices]
    targets_sliced = all_targets[:, :seq_scored, scoring_indices]

    # 3. Calculate RMSE per sample (averaged over columns)
    mse_per_sample = torch.mean((preds_sliced - targets_sliced) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load Validation Metadata to correlate errors with features
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Create error dataframe and merge
    error_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})
    analysis_df = pd.merge(val_df, error_df, on="id")

    # Feature Engineering for Analysis
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )
    analysis_df["unpaired_pct"] = analysis_df["structure"].apply(
        lambda x: x.count(".") / len(x)
    )

    # Calculate Correlations
    print("Failure Analysis (Correlation with Error):")
    cols_to_analyze = ["signal_to_noise", "SN_filter", "gc_content", "unpaired_pct"]
    cols_to_analyze = [c for c in cols_to_analyze if c in analysis_df.columns]

    correlations = analysis_df[cols_to_analyze].corrwith(analysis_df["error"])
    print(correlations)

    # ==========================================
    # 9. Conditional Submission
    # ==========================================
    threshold = 0.5884495377540588

    if best_score < threshold:
        print("Generating submission...")
        test_preds, test_ids = inference(model, test_loader, device)

        # Ensure output directory exists
        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")

        generate_submission(test_preds, test_ids, sub_path)
    else:
        print(f"Validation score {best_score} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
