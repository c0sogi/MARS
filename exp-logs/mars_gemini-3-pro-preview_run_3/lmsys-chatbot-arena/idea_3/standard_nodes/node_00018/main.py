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
    Runs validation to compute the final metric and performs failure analysis
    by correlating error magnitude with input features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_losses = []
    all_lens_p = []
    all_lens_a = []
    all_lens_b = []

    # Use CrossEntropyLoss with reduction='none' to get per-sample loss
    # This is equivalent to Log Loss for multi-class when targets are probabilities
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in val_loader:
            ids_prompt = batch["ids_prompt"].to(device)
            ids_a = batch["ids_a"].to(device)
            ids_b = batch["ids_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with Mixed Precision
            with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
                logits = model(
                    ids_prompt=ids_prompt,
                    ids_a=ids_a,
                    ids_b=ids_b,
                    scalar_features=scalar_features,
                )

                # Calculate loss per sample
                # labels are probabilities (float), supported by CrossEntropyLoss
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
    # 1. Configuration Overrides
    # Cite solution_lesson_node_00015: "Prioritize data coverage over architectural complexity"
    # We remove the sample limit to train on the full dataset.
    Config.USE_FP16 = True
    # We use the epochs defined in Config (5)

    # 2. Initialize Trainer
    # This loads data and model
    trainer = Trainer()

    # 3. Train
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
