import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, logging as hf_logging
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

# Import from provided libraries
from library.config import Config, seed_everything
from library.modeling import SiameseHybridModel
from library.engine import train_model, predict
from library.data import get_dataloaders


def main():
    # 1. Setup and Configuration
    # Suppress transformers warnings/info
    hf_logging.set_verbosity_error()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Override Config for fast baseline execution
    # 2 Epochs on ~41k samples with batch size 16 is efficient enough for the time limit
    # while providing enough data for the model to converge.
    Config.EPOCHS = 2

    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 2. Training
    print("Starting Training...")
    # train_model handles the training loop, validation monitoring, and saves the best model to Config.MODEL_PATH
    train_model(tokenizer)

    # 3. Validation and Failure Analysis
    print("Starting Post-Training Validation and Failure Analysis...")

    # Load the best saved model for analysis to ensure consistency
    device = Config.DEVICE
    model = SiameseHybridModel()
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded best model from {Config.MODEL_PATH}")
    else:
        print("Warning: Model checkpoint not found. Using current model weights.")

    model.to(device)
    model.eval()

    # Get Validation Data
    # We use the same dataloader function, utilizing the cache
    _, val_loader = get_dataloaders(tokenizer, load_cached_data=True)

    # Use reduction='none' to get loss per sample for failure analysis
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    all_losses = []
    all_scalars = []
    all_preds = []
    all_labels = []

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalars = batch["scalars"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    scalars,
                )
                loss = criterion(logits, labels)

            # Store data for analysis
            all_losses.append(loss.cpu().numpy())
            all_scalars.append(scalars.cpu().numpy())

            # Store predictions for metric calculation
            probs = torch.softmax(logits.float(), dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate results
    all_losses = np.concatenate(all_losses)
    all_scalars = np.concatenate(all_scalars, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute Final Metric (Log Loss)
    # y_true is (N, 3) probabilities, y_pred is (N, 3) probabilities
    final_metric = log_loss(all_labels, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error (Loss) and Scalar Features
    # Scalars are: [log(prompt_len), log(resp_a_len), log(resp_b_len)]
    feature_names = [
        "Prompt Length (Log)",
        "Response A Length (Log)",
        "Response B Length (Log)",
    ]

    print("Failure Analysis - Correlation with Error Magnitude:")
    for i, name in enumerate(feature_names):
        # Calculate Pearson correlation
        corr, _ = pearsonr(all_scalars[:, i], all_losses)
        print(f"{name}: {corr:.4f}")

    # 4. Submission
    # Threshold defined in the task
    THRESHOLD = 1.0392143626595562

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        predict(tokenizer)
    else:
        print(
            f"Metric {final_metric} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
