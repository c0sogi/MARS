import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from scipy.stats import pearsonr

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.trainer import Trainer
from library.utils import apk


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # Configuration Override
    # -------------------------------------------------------------------------
    # Extending training to 12 epochs to allow ArcFace convergence.
    # Increasing warmup to 2 epochs to stabilize backbone before margin penalty.
    Config.TOTAL_EPOCHS = 12
    Config.WARMUP_EPOCHS = 2

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    trainer = Trainer()
    trainer.fit()

    # -------------------------------------------------------------------------
    # 3. Validation & Metrics
    # -------------------------------------------------------------------------
    # Load the best model weights saved during training
    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        trainer.model.load_state_dict(state_dict)

    trainer.model.eval()
    val_loader = trainer.get_dataloader("val")

    all_preds = []
    all_targets = []

    # Perform inference to get detailed predictions for analysis
    with torch.no_grad():
        # Normalize class weights to serve as prototypes
        prototypes = torch.nn.functional.normalize(
            trainer.model.head.weight, p=2, dim=1
        )

        for batch in val_loader:
            images = batch["image"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)

            # Extract image embeddings
            embeddings = trainer.model(images, labels=None)

            # Compute Cosine Similarity
            logits = torch.matmul(embeddings, prototypes.t())

            # Get Top 5 predictions
            _, top_k_indices = torch.topk(logits, k=5, dim=1)

            all_preds.extend(top_k_indices.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    # Compute MAP@5
    scores = []
    for t, p in zip(all_targets, all_preds):
        scores.append(apk(t, p, k=5))

    final_metric = np.mean(scores)
    # Print metric with full precision
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing failure analysis...")

    val_df = pd.read_csv(Config.VAL_META)

    # Ensure dataframe aligns with predictions
    if len(val_df) == len(scores):
        val_df["apk"] = scores
        val_df["error_magnitude"] = 1.0 - val_df["apk"]

        # Correlation 1: Chain ID (Input Feature)
        if "chain" in val_df.columns:
            valid_chain = val_df.dropna(subset=["chain", "error_magnitude"])
            if len(valid_chain) > 1:
                corr_chain, _ = pearsonr(
                    valid_chain["chain"], valid_chain["error_magnitude"]
                )
                print(f"Correlation between Error Magnitude and Chain ID: {corr_chain}")

        # Correlation 2: Class Frequency (Derived Feature)
        # We check if the model struggles with rare classes (long-tail problem)
        train_df = pd.read_csv(Config.TRAIN_META)
        class_counts = train_df["hotel_id"].value_counts().to_dict()

        val_df["class_freq"] = val_df["hotel_id"].map(class_counts).fillna(0)

        valid_freq = val_df.dropna(subset=["class_freq", "error_magnitude"])
        if len(valid_freq) > 1:
            corr_freq, _ = pearsonr(
                valid_freq["class_freq"], valid_freq["error_magnitude"]
            )
            print(
                f"Correlation between Error Magnitude and Class Frequency: {corr_freq}"
            )
    else:
        print(
            "Warning: Validation dataframe length does not match prediction count. Skipping detailed analysis."
        )

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.5747:
        trainer.predict_and_submit()
    else:
        print(
            f"Validation metric {final_metric} is not higher than 0.5747. Submission skipped."
        )


if __name__ == "__main__":
    main()
