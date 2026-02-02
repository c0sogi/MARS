import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

# Import from the provided library files
from library.configuration import Config
from library.utilities import set_seed
from library.dataset import load_data, get_dataloader
from library.architecture import TransformerModel
from library.engine import run_training, predict


def main():
    # 1. Setup and Initialization
    set_seed(42)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load processed dataframes (decoding is handled by load_data)
    print("Loading datasets...")
    df_train = load_data("train")
    df_val = load_data("val")
    df_test = load_data("test")

    print(f"Train size: {len(df_train)}")
    print(f"Val size: {len(df_val)}")
    print(f"Test size: {len(df_test)}")

    # 3. Ensemble Training
    # We iterate over the model configurations (RoBERTa, DeBERTa) and seeds defined in Config.
    # Total models: 2 architectures * 3 seeds = 6 models.

    trained_models_info = []

    for config in Config.MODEL_CONFIGS:
        for seed in Config.SEEDS:
            model_name_clean = config["model_name"].replace("/", "_")
            save_name = f"{model_name_clean}_seed_{seed}.bin"
            save_path = os.path.join(Config.MODEL_DIR, save_name)

            print(f"\n--- Training {config['model_name']} (Seed {seed}) ---")

            # Execute training using the engine's run_training function
            # This handles the training loop, validation monitoring, and saving the best model.
            run_training(
                df_train=df_train,
                df_val=df_val,
                model_config=config,
                seed=seed,
                save_name=save_name,
            )

            trained_models_info.append({"path": save_path, "config": config})

    # 4. Validation Inference & Aggregation
    print("\n--- Running Validation Inference ---")
    val_preds_accum = np.zeros(len(df_val))

    # Iterate through all trained models to generate predictions
    for info in trained_models_info:
        config = info["config"]
        path = info["path"]

        # Re-initialize tokenizer and loader for the specific model architecture
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
        val_loader = get_dataloader(
            df_val,
            tokenizer,
            batch_size=Config.VALID_BATCH_SIZE,
            is_test=False,  # We have labels, but predict() ignores them
            shuffle=False,
        )

        # Load Model
        model = TransformerModel(
            model_name=config["model_name"],
            dropout=config["dropout"],
            freeze_layers=config["freeze_layers"],
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        # Generate predictions
        preds = predict(val_loader, model, device)
        val_preds_accum += np.array(preds)

        # Cleanup to save memory
        del model, tokenizer, val_loader
        torch.cuda.empty_cache()

    # Compute Ensemble Average
    avg_val_preds = val_preds_accum / len(trained_models_info)

    # 5. Validation Metric
    val_targets = df_val["Insult"].values
    final_auc = roc_auc_score(val_targets, avg_val_preds)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(val_targets - avg_val_preds)

    # Feature: Text Length (Comment is already decoded by load_data)
    # We check if longer/shorter comments are harder to classify
    text_lengths = df_val["Comment"].str.len().values

    # Calculate correlation (using numpy to avoid scipy dependency issues)
    # Handle potential NaN or zero variance if dataset is weird, though unlikely here
    if np.std(errors) > 0 and np.std(text_lengths) > 0:
        corr = np.corrcoef(errors, text_lengths)[0, 1]
    else:
        corr = 0.0

    print(f"Correlation between Error Magnitude and Text Length: {corr:.4f}")

    # 7. Submission Generation
    # Only generate submission if metric exceeds threshold
    threshold = 0.9639490968801314

    if final_auc > threshold:
        print(f"\nMetric exceeds threshold ({threshold}). Generating submission...")

        test_preds_accum = np.zeros(len(df_test))

        for info in trained_models_info:
            config = info["config"]
            path = info["path"]

            # Setup for Test Inference
            tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
            test_loader = get_dataloader(
                df_test,
                tokenizer,
                batch_size=Config.VALID_BATCH_SIZE,
                is_test=True,
                shuffle=False,
            )

            # Load Model
            model = TransformerModel(
                model_name=config["model_name"],
                dropout=config["dropout"],
                freeze_layers=config["freeze_layers"],
            )
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()

            # Predict
            preds = predict(test_loader, model, device)
            test_preds_accum += np.array(preds)

            # Cleanup
            del model, tokenizer, test_loader
            torch.cuda.empty_cache()

        # Average Test Predictions
        avg_test_preds = test_preds_accum / len(trained_models_info)

        # Create Submission DataFrame
        # Format: Insult, Date, Comment (based on sample_submission_null.csv)
        submission = df_test.copy()
        submission["Insult"] = avg_test_preds

        # Ensure column order matches requirements
        submission = submission[["Insult", "Date", "Comment"]]

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
