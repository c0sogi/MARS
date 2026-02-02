import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from transformers import AutoTokenizer

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_loader import get_tfidf_features, get_dataloaders, get_test_dataloader
from library.models import LinearModelWrapper, CustomTransformer
from library.trainer import run_transformer_training
from library.ensemble import Ensemble


def run_inference(model, loader, device):
    """
    Runs inference on a DataLoader using the provided model.
    Returns:
        np.ndarray: Predicted probabilities (N, num_classes)
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)

            logits = model(ids, mask, token_type_ids)
            probs = torch.sigmoid(logits)
            preds.append(probs.detach().cpu().numpy())

    return np.concatenate(preds)


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    seed_everything(Config.seed)

    # Fast Baseline Overrides
    Config.epochs = (
        2  # Train for 2 epochs to improve convergence (Cite solution_lesson_node_00009)
    )
    Config.debug = False  # Use full data to ensure high metric

    print("Starting Runfile Execution...")
    print(f"Device: {Config.device}")
    print(f"Epochs: {Config.epochs}")

    # ==========================================
    # 2. Linear Model Branch
    # ==========================================
    print("\n=== Branch C: Linear Model ===")
    X_train, X_val, X_test, y_train, y_val = get_tfidf_features(load_cached_data=True)

    linear_model = LinearModelWrapper()
    linear_model.fit(X_train, y_train)

    print("Generating Linear Model predictions...")
    val_preds_linear = linear_model.predict_proba(X_val)
    test_preds_linear = linear_model.predict_proba(X_test)

    # Cleanup memory
    del X_train, X_val, X_test, linear_model
    gc.collect()

    # ==========================================
    # 3. Transformer Branches
    # ==========================================
    transformer_preds_val = []
    transformer_preds_test = []

    models_to_run = [
        ("DeBERTa-v3", Config.model_1_name, "best_model_deberta.bin"),
        ("RoBERTa", Config.model_2_name, "best_model_roberta.bin"),
    ]

    for friendly_name, model_name, save_file in models_to_run:
        print(f"\n=== Branch: {friendly_name} ({model_name}) ===")

        # A. Train
        # run_transformer_training handles the training loop and saves the best model
        best_auc = run_transformer_training(model_name, save_file)

        # B. Inference
        print(f"Loading best {friendly_name} model for inference...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Re-create loaders (validation and test)
        # Note: get_dataloaders returns (train, val). We only need val here.
        _, val_loader = get_dataloaders(tokenizer)
        test_loader, test_ids = get_test_dataloader(tokenizer)

        # Initialize model and load weights
        model = CustomTransformer(model_name, config=Config)
        checkpoint_path = os.path.join(Config.working_dir, save_file)
        model.load_state_dict(torch.load(checkpoint_path, map_location=Config.device))
        model.to(Config.device)

        # Generate predictions
        print(f"Predicting validation set ({friendly_name})...")
        val_preds = run_inference(model, val_loader, Config.device)

        print(f"Predicting test set ({friendly_name})...")
        test_preds = run_inference(model, test_loader, Config.device)

        transformer_preds_val.append(val_preds)
        transformer_preds_test.append(test_preds)

        # Cleanup
        del model, tokenizer, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    val_preds_deberta = transformer_preds_val[0]
    test_preds_deberta = transformer_preds_test[0]
    val_preds_roberta = transformer_preds_val[1]
    test_preds_roberta = transformer_preds_test[1]

    # ==========================================
    # 4. Ensemble & Validation
    # ==========================================
    print("\n=== Ensemble Optimization ===")

    val_preds_list = [val_preds_deberta, val_preds_roberta, val_preds_linear]
    test_preds_list = [test_preds_deberta, test_preds_roberta, test_preds_linear]

    ensemble = Ensemble()
    # Optimize weights to maximize AUC on validation set
    ensemble.optimize_weights(
        val_preds_list, y_val, initial_weights=Config.initial_weights
    )

    # Calculate final validation predictions
    final_val_preds = np.zeros_like(val_preds_list[0])
    for i, w in enumerate(ensemble.weights):
        final_val_preds += w * val_preds_list[i]

    final_val_auc = compute_auc(y_val, final_val_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_val_auc}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    # Shape: (N_val, 6) -> mean across classes -> (N_val,)
    sample_errors = np.mean(np.abs(y_val - final_val_preds), axis=1)

    # Load validation metadata to get text features
    val_df = pd.read_csv(Config.val_path)
    if Config.debug:
        val_df = val_df.sample(
            n=min(500, len(val_df)), random_state=Config.seed
        ).reset_index(drop=True)

    # Feature: Character Length
    val_df["char_length"] = val_df["comment_text"].fillna("").apply(len)

    # Correlation
    corr, _ = pearsonr(sample_errors, val_df["char_length"])
    print(
        f"Correlation between Error Magnitude and Input Feature (Char Length): {corr:.6f}"
    )

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.9926490739470294

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_val_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Blend test predictions
        final_test_preds = ensemble.blend_predictions(test_preds_list)

        # Create submission DataFrame
        submission = pd.DataFrame(
            {
                "id": test_ids,
                "toxic": final_test_preds[:, 0],
                "severe_toxic": final_test_preds[:, 1],
                "obscene": final_test_preds[:, 2],
                "threat": final_test_preds[:, 3],
                "insult": final_test_preds[:, 4],
                "identity_hate": final_test_preds[:, 5],
            }
        )

        # Save
        submission.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"\nValidation metric ({final_val_auc}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
