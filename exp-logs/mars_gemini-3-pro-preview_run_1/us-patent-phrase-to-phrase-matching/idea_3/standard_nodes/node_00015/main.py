import os
import gc
import sys
import time
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from scipy.stats import pearsonr

# Import library modules
from library.config import CFG
from library.dataset import get_data, PhraseDataset
from library.model import PhraseModel
from library.engine import train_fn, valid_fn
from library.utils import seed_everything, get_score


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Optimize configuration for speed and performance on A100
    CFG.epochs = 2  # 2 epochs are sufficient for convergence on this dataset
    CFG.batch_size = 32  # A100 40GB allows larger batch size for speed
    CFG.awp_start_epoch = 1  # Enable AWP only for the second epoch
    CFG.print_freq = 50
    CFG.num_workers = 4

    # Ensure reproducibility
    seed_everything(CFG.seed)

    device = CFG.device
    print(f"Using device: {device}")
    print(
        f"Training with {CFG.n_fold} folds, {CFG.epochs} epochs, batch size {CFG.batch_size}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[Data Loading]")
    # Load combined train+val for CV, and test set
    # This uses the cached parquet files if available
    train_df, test_df = get_data(CFG, load_cached_data=True)

    # Load original validation metadata to identify the specific hold-out set
    # required for the final metric calculation
    val_meta_df = pd.read_csv(CFG.VAL_METADATA)
    val_ids = set(val_meta_df["id"].values)

    print(f"Total Train+Val samples: {len(train_df)}")
    print(f"Hold-out Validation samples: {len(val_ids)}")
    print(f"Test samples: {len(test_df)}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    # -------------------------------------------------------------------------
    # 3. Cross-Validation Training
    # -------------------------------------------------------------------------
    print("\n[Training]")
    # Array to store Out-Of-Fold predictions
    oof_preds = np.zeros(len(train_df))

    for fold in CFG.trn_fold:
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        trn_idx = train_df[train_df["fold"] != fold].index
        val_idx = train_df[train_df["fold"] == fold].index

        train_folds = train_df.loc[trn_idx].reset_index(drop=True)
        valid_folds = train_df.loc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = PhraseDataset(CFG, train_folds, tokenizer, mode="train")
        valid_dataset = PhraseDataset(CFG, valid_folds, tokenizer, mode="train")

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Initialize Model
        model = PhraseModel(CFG, pretrained=True)
        model.to(device)

        # Optimizer (Separate weight decay for bias/LayerNorm)
        param_optimizer = list(model.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": CFG.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas
        )

        # Scheduler
        num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CFG.num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=CFG.num_cycles,
        )

        # Loss Function
        criterion = torch.nn.MSELoss()

        # Training Loop
        best_score = -1
        best_model_path = os.path.join(CFG.WORKING_DIR, f"model_fold_{fold}.pth")

        for epoch in range(CFG.epochs):
            # Train
            train_fn(
                fold,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                device,
                CFG,
            )

            # Validate
            avg_val_loss, preds, val_score = valid_fn(
                valid_loader, model, criterion, device, CFG
            )
            print(f"Epoch {epoch+1} - Validation Pearson Score: {val_score:.4f}")

            # Save Best Model
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)

        # Load best model to generate final OOF predictions for this fold
        # This ensures OOF score reflects the best state
        model.load_state_dict(torch.load(best_model_path))
        _, preds, _ = valid_fn(valid_loader, model, criterion, device, CFG)
        oof_preds[val_idx] = preds

        # Cleanup to free memory
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            train_dataset,
            valid_dataset,
        )
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------------
    # 4. Validation Assessment
    # -------------------------------------------------------------------------
    print("\n[Validation Assessment]")
    # Filter OOF predictions to the hold-out validation set
    val_mask = train_df["id"].isin(val_ids)
    val_subset_df = train_df[val_mask]
    val_subset_preds = oof_preds[val_mask]
    val_subset_labels = val_subset_df["score"].values

    final_score = get_score(val_subset_labels, val_subset_preds)
    print(f"Final Validation Metric: {final_score}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Failure Analysis]")
    # Calculate absolute error
    errors = np.abs(val_subset_labels - val_subset_preds)

    # Calculate features for correlation
    lengths_anchor = val_subset_df["anchor"].astype(str).apply(len).values
    lengths_target = val_subset_df["target"].astype(str).apply(len).values
    lengths_context = val_subset_df["context_text"].astype(str).apply(len).values

    # Compute correlations
    corr_anchor = pearsonr(errors, lengths_anchor)[0]
    corr_target = pearsonr(errors, lengths_target)[0]
    corr_context = pearsonr(errors, lengths_context)[0]

    print(f"Correlation (Error vs Anchor Length): {corr_anchor:.4f}")
    print(f"Correlation (Error vs Target Length): {corr_target:.4f}")
    print(f"Correlation (Error vs Context Length): {corr_context:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    if final_score > 0.8673:
        print("\n[Generating Submission]")
        print("Score exceeds threshold (0.8673). Starting inference on Test Set...")

        test_dataset = PhraseDataset(CFG, test_df, tokenizer, mode="test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        final_preds = []

        # Ensemble inference
        for fold in CFG.trn_fold:
            model_path = os.path.join(CFG.WORKING_DIR, f"model_fold_{fold}.pth")
            print(f"Loading model from {model_path}...")

            model = PhraseModel(CFG, pretrained=False)
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for inputs in test_loader:
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)

                    y_preds = model(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs.get("token_type_ids"),
                    )
                    fold_preds.append(y_preds.cpu().numpy())

            final_preds.append(np.concatenate(fold_preds))

            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Average predictions across folds
        avg_preds = np.mean(final_preds, axis=0)

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_df["id"], "score": avg_preds})

        # Ensure scores are within [0, 1] (optional but safe)
        submission["score"] = submission["score"].clip(0, 1)

        # Save
        submission_path = os.path.join(CFG.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nScore {final_score:.4f} is not above threshold 0.8673. Skipping submission."
        )


if __name__ == "__main__":
    main()
