import os
import gc
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import InsultDataset, get_tokenizer, load_dataset_dataframe
from library.model import InsultModel
from library.trainer import run_fold
from library.inference import predict_fn


def main():
    # 1. Configuration & Setup
    seed_everything(Config.seed)

    # Override Config for fast baseline execution
    # Reducing epochs to 2 ensures the run completes quickly on the A100
    Config.epochs = 2

    print("Loading Data...")
    # Load train and val metadata using the caching mechanism
    train_df_part = load_dataset_dataframe(
        Config.train_path, "train_cleaned", load_cached_data=True
    )
    val_df_part = load_dataset_dataframe(
        Config.val_path, "val_cleaned", load_cached_data=True
    )

    # Combine for Cross-Validation
    full_train_df = pd.concat([train_df_part, val_df_part]).reset_index(drop=True)

    # 2. Training & OOF Generation
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Array to store aggregated Out-Of-Fold predictions
    # Initialize with zeros. We will add predictions from each backbone and normalize later.
    oof_preds = np.zeros(len(full_train_df))

    # Iterate over Heterogeneous Ensemble Backbones
    for model_name in Config.model_backbones:
        print(f"\n{'='*40}\nProcessing Backbone: {model_name}\n{'='*40}")

        # Iterate over Folds
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_train_df, full_train_df["Insult"])
        ):
            print(f"\n--- Fold {fold} ---")

            # Train the model for this fold
            # This function saves the best model to Config.working_dir
            run_fold(fold, full_train_df, train_idx, val_idx, model_name)

            # --- Generate OOF Predictions for this Fold ---
            print(f"Generating OOF predictions for Fold {fold}...")

            # Prepare Validation Data
            val_fold_df = full_train_df.iloc[val_idx].reset_index(drop=True)
            tokenizer = get_tokenizer(model_name)
            val_dataset = InsultDataset(
                val_fold_df, tokenizer, Config.max_len, is_test=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.batch_size * 2,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

            # Load the Model
            model = InsultModel(model_name, Config)
            model_name_safe = model_name.replace("/", "_")
            weight_path = os.path.join(
                Config.working_dir, f"{model_name_safe}_fold_{fold}.pth"
            )

            state_dict = torch.load(weight_path, map_location=Config.device)
            model.load_state_dict(state_dict)
            model.to(Config.device)
            model.eval()

            fold_preds = []

            # Inference Loop (No Gradients)
            with torch.no_grad():
                for data in val_loader:
                    input_ids = data["input_ids"].to(Config.device)
                    attention_mask = data["attention_mask"].to(Config.device)

                    with autocast(enabled=Config.use_fp16):
                        outputs = model(input_ids, attention_mask)
                        probs = torch.sigmoid(outputs).view(-1)

                    fold_preds.append(probs.cpu().numpy())

            # Concatenate predictions for this fold
            fold_preds = np.concatenate(fold_preds)

            # Accumulate predictions
            # We average across backbones, so we add (pred / num_backbones)
            oof_preds[val_idx] += fold_preds / len(Config.model_backbones)

            # Cleanup
            del model, val_loader, val_dataset
            gc.collect()
            torch.cuda.empty_cache()

    # 3. Validation Assessment
    final_auc = roc_auc_score(full_train_df["Insult"], oof_preds)
    print(f"\nFinal Validation Metric: {final_auc}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    full_train_df["pred"] = oof_preds
    full_train_df["error"] = np.abs(full_train_df["Insult"] - full_train_df["pred"])

    # Feature: Text Length
    full_train_df["text_len"] = full_train_df["Comment"].apply(lambda x: len(str(x)))

    # Calculate Correlation
    correlation = full_train_df[["error", "text_len"]].corr().iloc[0, 1]
    print(f"Correlation between Error and Text Length: {correlation}")

    # 5. Submission Generation
    threshold = 0.9632101806239738

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) > Threshold ({threshold}). Generating submission..."
        )
        predict_fn()
    else:
        print(
            f"\nValidation metric ({final_auc}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
