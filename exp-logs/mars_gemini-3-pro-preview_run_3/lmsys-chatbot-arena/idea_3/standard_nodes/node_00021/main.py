import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.train import Trainer
from library.utils import seed_everything


def analyze_failures_and_score(model, val_loader, device):
    """
    Runs validation to compute the final metric and performs failure analysis.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_losses = []
    all_lens_p = []
    all_lens_a = []
    all_lens_b = []

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in val_loader:
            input_ids_p = batch["input_ids_p"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
                logits = model(
                    input_ids_p=input_ids_p,
                    input_ids_a=input_ids_a,
                    input_ids_b=input_ids_b,
                    scalar_features=scalar_features,
                )
                loss = criterion(logits, labels)

            all_losses.append(loss.cpu().numpy())

            # Extract scalar features for correlation analysis
            # scalar_features: [log(len_p), log(len_a), log(len_b)]
            sf = scalar_features.cpu().numpy()
            all_lens_p.append(sf[:, 0])
            all_lens_a.append(sf[:, 1])
            all_lens_b.append(sf[:, 2])

    # Concatenate all batches
    all_losses = np.concatenate(all_losses)
    all_lens_p = np.concatenate(all_lens_p)
    all_lens_a = np.concatenate(all_lens_a)
    all_lens_b = np.concatenate(all_lens_b)

    # Calculate Mean Log Loss (The Metric)
    final_metric = np.mean(all_losses)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "loss": all_losses,
            "log_len_prompt": all_lens_p,
            "log_len_resp_a": all_lens_a,
            "log_len_resp_b": all_lens_b,
        }
    )

    # Compute correlations
    correlations = df_analysis.corr()["loss"].drop("loss")
    print("Correlation between Error Magnitude (Log Loss) and Input Features:")
    print(correlations)

    return final_metric


def main():
    # 1. Initialize Trainer
    # This loads data and model
    trainer = Trainer()

    # 2. Train on full dataset (Cite Lesson 00015)
    trainer.fit()

    # 5. Validation and Failure Analysis
    print("Running validation inference...")
    val_metric = analyze_failures_and_score(
        trainer.model, trainer.val_loader, trainer.device
    )

    # Print the final metric in the required format
    print(f"Final Validation Metric: {val_metric}")

    # 6. Submission Logic
    # Threshold defined in task
    THRESHOLD = 1.0392143626595562

    if val_metric < THRESHOLD:
        print(
            f"Metric {val_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"Metric {val_metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
