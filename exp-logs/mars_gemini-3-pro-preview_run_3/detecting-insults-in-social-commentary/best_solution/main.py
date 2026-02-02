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


def train_ensemble(df_train, df_val, suffix=""):
    """
    Trains the ensemble of models defined in Config.
    Returns a list of dictionaries containing model paths and configs.
    """
    trained_models_info = []

    for config in Config.MODEL_CONFIGS:
        for seed in Config.SEEDS:
            model_name_clean = config["model_name"].replace("/", "_")
            save_name = f"{model_name_clean}_seed_{seed}{suffix}.bin"

            print(f"\n--- Training {config['model_name']} (Seed {seed}) [{suffix}] ---")

            run_training(
                df_train=df_train,
                df_val=df_val,
                model_config=config,
                seed=seed,
                save_name=save_name,
            )

            save_path = os.path.join(Config.MODEL_DIR, save_name)
            trained_models_info.append({"path": save_path, "config": config})

    return trained_models_info


def inference_ensemble(trained_models_info, df_target, is_test=False):
    """
    Runs inference using the provided list of models on the target dataframe.
    Returns averaged predictions.
    """
    device = Config.DEVICE
    preds_accum = np.zeros(len(df_target))

    for info in trained_models_info:
        config = info["config"]
        path = info["path"]

        # Setup Tokenizer and Loader
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
        loader = get_dataloader(
            df_target,
            tokenizer,
            batch_size=Config.VALID_BATCH_SIZE,
            is_test=is_test,
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
        preds = predict(loader, model, device)
        preds_accum += np.array(preds)

        # Cleanup
        del model, tokenizer, loader
        torch.cuda.empty_cache()

    return preds_accum / len(trained_models_info)


def main():
    # 1. Setup and Initialization
    set_seed(42)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    df_train = load_data("train")
    df_val = load_data("val")
    df_test = load_data("test")

    print(f"Train size: {len(df_train)}")
    print(f"Val size: {len(df_val)}")
    print(f"Test size: {len(df_test)}")

    # 3. Stage 1: Initial Ensemble Training
    print("\n=== Stage 1: Initial Ensemble Training ===")
    models_stage1 = train_ensemble(df_train, df_val, suffix="_stage1")

    # 4. Pseudo-Labeling
    # Cite solution_lesson_node_00027: Pseudo-labeling (implemented with consistent batch size)
    print("\n=== Generating Pseudo-Labels ===")
    test_preds_stage1 = inference_ensemble(models_stage1, df_test, is_test=True)

    # Select high-confidence samples
    high_conf_indices = []
    pseudo_labels = []

    for i, pred in enumerate(test_preds_stage1):
        if pred < Config.PSEUDO_LABEL_CONF_LOW:
            high_conf_indices.append(i)
            pseudo_labels.append(0)  # Neutral
        elif pred > Config.PSEUDO_LABEL_CONF_HIGH:
            high_conf_indices.append(i)
            pseudo_labels.append(1)  # Insulting

    print(
        f"Selected {len(high_conf_indices)} pseudo-labeled samples from {len(df_test)} test samples."
    )

    if len(high_conf_indices) > 0:
        df_pseudo = df_test.iloc[high_conf_indices].copy()
        df_pseudo["Insult"] = pseudo_labels

        # Augment Training Data
        df_train_aug = pd.concat([df_train, df_pseudo], axis=0, ignore_index=True)
        print(f"Augmented Train size: {len(df_train_aug)}")

        # 5. Stage 2: Retraining on Augmented Data
        print("\n=== Stage 2: Retraining Ensemble on Augmented Data ===")
        # We use the augmented training set, but keep the original validation set for fair comparison
        models_final = train_ensemble(df_train_aug, df_val, suffix="_stage2")
    else:
        print("No high-confidence samples found. Skipping Stage 2.")
        models_final = models_stage1

    # 6. Final Validation Inference
    print("\n=== Final Validation Inference ===")
    avg_val_preds = inference_ensemble(models_final, df_val, is_test=False)

    # 7. Validation Metric
    val_targets = df_val["Insult"].values
    final_auc = roc_auc_score(val_targets, avg_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 8. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(val_targets - avg_val_preds)
    text_lengths = df_val["Comment"].str.len().values
    if np.std(errors) > 0 and np.std(text_lengths) > 0:
        corr = np.corrcoef(errors, text_lengths)[0, 1]
    else:
        corr = 0.0
    print(f"Correlation between Error Magnitude and Text Length: {corr:.4f}")

    # 9. Submission Generation
    threshold = 0.9648522167487685

    if final_auc > threshold:
        print(f"\nMetric exceeds threshold ({threshold}). Generating submission...")

        # Generate predictions on Test Set using Final Models
        avg_test_preds = inference_ensemble(models_final, df_test, is_test=True)

        submission = pd.read_csv(Config.TEST_PATH)
        submission["Insult"] = avg_test_preds
        submission = submission[["Insult", "Date", "Comment"]]
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
