import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config, set_seed
from library.trainer import Trainer
from library.data_loader import get_dataloaders
from library.model import SiameseDAN, evaluate


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set.
    Calculates correlation between error magnitude and input features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_preds = []
    all_labels = []

    # Collect predictions and labels
    with torch.no_grad():
        for batch in val_loader:
            anchor = batch["anchor"].to(device)
            target = batch["target"].to(device)
            context = batch["context"].to(device)
            labels = batch["score"].to(device)

            preds = model(anchor, target, context)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Error Magnitude
    errors = np.abs(all_preds - all_labels)

    # Load Validation Dataframe to get raw text features
    # (Since DataLoader shuffles=False for validation, order is preserved)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)

    if len(df_val) != len(errors):
        print(
            f"Warning: Validation dataframe length ({len(df_val)}) does not match prediction length ({len(errors)}). Skipping detailed feature analysis."
        )
        return

    # Feature 1: Jaccard Similarity
    def get_jaccard(row):
        s1 = set(str(row["anchor"]).lower().split())
        s2 = set(str(row["target"]).lower().split())
        union = len(s1.union(s2))
        return len(s1.intersection(s2)) / union if union > 0 else 0.0

    jaccard_sims = df_val.apply(get_jaccard, axis=1)
    corr_jaccard, _ = pearsonr(errors, jaccard_sims)
    print(f"Correlation (Error vs Jaccard Sim): {corr_jaccard}")

    # Feature 2: Length Difference (Character level)
    len_diffs = np.abs(
        df_val["anchor"].astype(str).str.len() - df_val["target"].astype(str).str.len()
    )
    corr_len, _ = pearsonr(errors, len_diffs)
    print(f"Correlation (Error vs Char Length Diff): {corr_len}")

    # Feature 3: Common Words Count
    def get_common_count(row):
        s1 = set(str(row["anchor"]).lower().split())
        s2 = set(str(row["target"]).lower().split())
        return len(s1.intersection(s2))

    common_counts = df_val.apply(get_common_count, axis=1)
    corr_common, _ = pearsonr(errors, common_counts)
    print(f"Correlation (Error vs Common Words): {corr_common}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Configure for Fast Baseline
    # We reduce epochs and ensure batch size is efficient
    Config.BATCH_SIZE = 128
    FAST_EPOCHS = 10

    print(
        f"Initializing Fast Baseline Run (Epochs={FAST_EPOCHS}, Batch={Config.BATCH_SIZE})..."
    )

    # 2. Initialize Trainer
    trainer = Trainer()

    # 3. Train Model
    # Trainer handles the training loop
    best_pearson = trainer.train(epochs=FAST_EPOCHS, load_cached_data=True)

    # 4. Post-Training Validation & Analysis
    # We reload the best model to ensure we are evaluating the optimal state
    print("\n--- Post-Training Evaluation ---")

    # Re-load dataloaders (should be fast due to caching)
    _, val_loader, test_loader, vocab_size, num_contexts, _ = get_dataloaders(
        load_cached_data=True
    )

    # Initialize model structure
    model = SiameseDAN(
        vocab_size=vocab_size,
        num_contexts=num_contexts,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        context_dim=Config.CONTEXT_EMBEDDING_DIM,
        dropout_rate=Config.DROPOUT,
    ).to(device)

    # Load weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Best model loaded successfully.")
    else:
        print(
            "Warning: Best model not found. Using initialized weights (results will be poor)."
        )

    # Calculate Final Validation Metric
    criterion = torch.nn.MSELoss()
    _, val_pearson = evaluate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_pearson}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # Conditional Submission Generation
    if val_pearson > 0.4015:
        print(
            f"Validation metric ({val_pearson:.6f}) > 0.4015. Generating submission..."
        )
        trainer.predict(model, test_loader)
    else:
        print(
            f"Validation metric ({val_pearson:.6f}) <= 0.4015. Skipping submission generation."
        )

    print("Run complete.")


if __name__ == "__main__":
    main()
