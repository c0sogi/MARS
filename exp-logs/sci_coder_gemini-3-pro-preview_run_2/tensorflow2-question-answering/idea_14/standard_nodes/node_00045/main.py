import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Import library components
from library.config import Config
from library.engine import train_model
from library.inference import run_inference_pipeline
from library.data import get_vocab, get_dataloaders
from library.model import CQCRNN
from library.utils import seed_everything


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Initializing configuration for fast baseline run...")

    # Override Config for speed and memory safety during training
    Config.DEBUG_SAMPLE_SIZE = 5000  # Limit training data size for speed
    Config.NUM_EPOCHS = 1  # Single epoch for baseline
    Config.BATCH_SIZE = 64  # Reasonable batch size
    Config.LEARNING_RATE = 0.001

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    print("\n--- Starting Training Phase ---")
    # train_model handles data loading, model init, and the training loop
    # We force load_cached_data=False to ensure the DEBUG_SAMPLE_SIZE is applied
    # to the feature generation process.
    train_model(
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        load_cached_data=False,
    )

    # -------------------------------------------------------------------------
    # 3. Validation Assessment
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation Phase ---")

    # Load resources for validation
    vocab = get_vocab(load_cached_data=True)
    # We use load_cached_data=True here because train_model just created the features
    _, val_loader = get_dataloaders(vocab, load_cached_data=True)

    # Load best model
    model = CQCRNN(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            "Warning: Model checkpoint not found. Using random weights for validation."
        )

    model.eval()

    all_preds = []
    all_labels = []

    # For Failure Analysis
    losses = []
    q_lens = []
    c_lens = []

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            q_input = batch["q_input"].to(device)
            c_input = batch["c_input"].to(device)
            labels = batch["label_long"].to(device)

            # Forward
            outputs = model(q_input, c_input)
            logits = outputs["long_logits"].squeeze(-1)

            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy()
            targets = labels.long().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(targets)

            # Failure Analysis Data
            # Calculate BCE loss per element (reduction='none')
            batch_loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            losses.extend(batch_loss.cpu().numpy())

            # Calculate lengths (non-padding tokens)
            # Assuming 0 is PAD_TOKEN
            q_l = (q_input != 0).sum(dim=1).cpu().numpy()
            c_l = (c_input != 0).sum(dim=1).cpu().numpy()

            q_lens.extend(q_l)
            c_lens.extend(c_l)

    # Compute Metric
    # Using Micro F1 Score for the Long Answer binary classification task
    val_f1 = f1_score(all_labels, all_preds, average="micro")
    print(f"Final Validation Metric: {val_f1}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Performing Failure Analysis ---")

    if len(losses) > 1:
        # Create a DataFrame for analysis
        df_analysis = pd.DataFrame(
            {"loss": losses, "question_length": q_lens, "candidate_length": c_lens}
        )

        # Correlation: Question Length vs Loss
        corr_q = np.corrcoef(df_analysis["question_length"], df_analysis["loss"])[0, 1]
        print(f"Correlation (Question Length vs Error): {corr_q:.4f}")

        # Correlation: Candidate Length vs Loss
        corr_c = np.corrcoef(df_analysis["candidate_length"], df_analysis["loss"])[0, 1]
        print(f"Correlation (Candidate Length vs Error): {corr_c:.4f}")

        # High loss statistics
        mean_loss = np.mean(losses)
        print(f"Mean Validation Loss: {mean_loss:.4f}")
    else:
        print("Not enough validation data for failure analysis.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")

    # IMPORTANT: Reset sample size to None to process the FULL test set
    Config.DEBUG_SAMPLE_SIZE = None

    # Run inference pipeline
    # We must set load_cached_data=False to force re-computation of features
    # because the previous run might have cached a truncated version (if we were debugging test flow)
    # or simply to ensure the full test set is processed now that DEBUG_SAMPLE_SIZE is None.
    run_inference_pipeline(
        model_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_PATH,
        threshold=Config.LONG_ANSWER_THRESHOLD,
        load_cached_data=False,
    )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
