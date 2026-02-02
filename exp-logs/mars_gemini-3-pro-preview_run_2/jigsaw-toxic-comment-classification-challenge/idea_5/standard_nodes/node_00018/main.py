import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_score, find_optimal_weights
from library.data_factory import get_dataloaders
from library.model_factory import ToxicityModel
from library.optimization import get_optimizer, get_scheduler
from library.awp import AWP
from library.loops import train_fn, valid_fn, inference_fn
from library.linear_model import train_linear_pipeline


def run_transformer(model_name, train_loader, val_loader, test_loader, device):
    """
    Runs the training, validation, and inference pipeline for a Transformer model.
    """
    print(f"\n=== Running Transformer Pipeline: {model_name} ===")

    # Initialize Model
    model = ToxicityModel(model_name=model_name, pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    # Calculate num_training_steps
    num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACCUM_STEPS
    num_training_steps = Config.EPOCHS * num_update_steps_per_epoch

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, num_training_steps)

    # Criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # AWP (Adversarial Weight Perturbation)
    # We explicitly pass start_epoch because the default arg in __init__ is bound at import time
    awp = None
    if Config.USE_AWP:
        awp = AWP(model, optimizer, start_epoch=Config.AWP_START_EPOCH)

    # Training Loop
    best_score = 0
    val_preds_final = None

    for epoch in range(Config.EPOCHS):
        print(f"--- Epoch {epoch + 1}/{Config.EPOCHS} ---")

        # Train
        avg_loss = train_fn(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            device,
            awp=awp,
            config=Config,
        )
        print(f"Training Loss: {avg_loss:.4f}")

        # Validate
        val_loss, val_score, val_preds = valid_fn(
            val_loader, model, criterion, device, config=Config
        )

        # Keep track of best predictions (though with 1 epoch, this is just the last one)
        if val_score > best_score:
            best_score = val_score
            val_preds_final = val_preds

        # If we only run 1 epoch, ensure we have predictions
        if val_preds_final is None:
            val_preds_final = val_preds

    # Inference on Test Set
    test_preds = inference_fn(test_loader, model, device, config=Config)

    # Cleanup to save memory for the next model
    del model, optimizer, scheduler, awp
    torch.cuda.empty_cache()

    return val_preds_final, test_preds


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Override Config for Fast Baseline execution
    # We reduce epochs to 1 to ensure the script finishes within the 2-hour limit.
    # We enable AWP from epoch 0 to maximize the single-epoch performance.
    Config.EPOCHS = 1
    Config.AWP_START_EPOCH = 0

    print("Configuration:")
    print(f"Device: {device}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"AWP Start Epoch: {Config.AWP_START_EPOCH}")
    print(f"Batch Size: {Config.TRAIN_BATCH_SIZE} (Accum: {Config.GRAD_ACCUM_STEPS})")

    # Load Validation Labels for final scoring
    val_df = pd.read_csv(Config.VAL_PATH)
    y_val = val_df[Config.LABEL_COLS].values

    # 2. Train Linear Model (Branch C)
    print("\n>>> Starting Branch C: Linear Model")
    val_preds_linear, test_preds_linear = train_linear_pipeline(load_cached_data=True)

    # 3. Train DeBERTa-v3-Large (Branch A)
    print("\n>>> Starting Branch A: DeBERTa-v3-Large")
    tokenizer_a = AutoTokenizer.from_pretrained(Config.MODEL_A_NAME)
    train_loader_a, val_loader_a, test_loader_a = get_dataloaders(
        tokenizer_a,
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
    )

    val_preds_a, test_preds_a = run_transformer(
        Config.MODEL_A_NAME, train_loader_a, val_loader_a, test_loader_a, device
    )

    # 4. Train RoBERTa-Large (Branch B)
    print("\n>>> Starting Branch B: RoBERTa-Large")
    tokenizer_b = AutoTokenizer.from_pretrained(Config.MODEL_B_NAME)
    train_loader_b, val_loader_b, test_loader_b = get_dataloaders(
        tokenizer_b,
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
    )

    val_preds_b, test_preds_b = run_transformer(
        Config.MODEL_B_NAME, train_loader_b, val_loader_b, test_loader_b, device
    )

    # 5. Ensemble & Evaluation
    print("\n>>> Ensembling Models")

    # List of predictions
    preds_list = [val_preds_linear, val_preds_a, val_preds_b]
    test_preds_list = [test_preds_linear, test_preds_a, test_preds_b]

    # Find optimal weights
    print("Optimizing ensemble weights...")
    weights = find_optimal_weights(preds_list, y_val)
    print(
        f"Optimal Weights: Linear={weights[0]:.4f}, DeBERTa={weights[1]:.4f}, RoBERTa={weights[2]:.4f}"
    )

    # Blend Validation Predictions
    final_val_preds = np.zeros_like(val_preds_linear)
    for i, pred in enumerate(preds_list):
        final_val_preds += weights[i] * pred

    # Compute Metric
    final_score = get_score(y_val, final_val_preds)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    print("\n>>> Failure Analysis")
    # Calculate error per sample (Mean Absolute Error across labels)
    # Shape: (N_val, 6) -> (N_val,)
    errors = np.mean(np.abs(y_val - final_val_preds), axis=1)

    # Get text lengths from validation dataframe
    val_df["char_length"] = val_df["comment_text"].fillna("").apply(len)
    lengths = val_df["char_length"].values

    # Correlation
    corr, _ = pearsonr(errors, lengths)
    print(f"Correlation between Error Magnitude and Text Length: {corr:.6f}")

    # 7. Submission
    threshold = 0.9927306969806252
    if final_score > threshold:
        print(
            f"\nMetric ({final_score}) > Threshold ({threshold}). Generating Submission..."
        )

        # Blend Test Predictions
        final_test_preds = np.zeros_like(test_preds_linear)
        for i, pred in enumerate(test_preds_list):
            final_test_preds += weights[i] * pred

        # Create Submission DataFrame
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        submission = pd.DataFrame(
            {
                "id": sample_sub["id"],
                "toxic": final_test_preds[:, 0],
                "severe_toxic": final_test_preds[:, 1],
                "obscene": final_test_preds[:, 2],
                "threat": final_test_preds[:, 3],
                "insult": final_test_preds[:, 4],
                "identity_hate": final_test_preds[:, 5],
            }
        )

        # Save to ./submission/submission.csv
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_score}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
