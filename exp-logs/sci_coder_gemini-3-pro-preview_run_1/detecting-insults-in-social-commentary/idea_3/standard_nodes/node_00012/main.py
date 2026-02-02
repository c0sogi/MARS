import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_processing import load_data
from library.model import HybridDebertaModel
from library.train import train_fn, eval_fn, predict_fn


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Load Data
    # load_cached_data=True allows using precomputed features from ./working if available
    train_dataset, val_dataset, test_dataset = load_data(load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = HybridDebertaModel(
        model_name=Config.MODEL_NAME,
        num_structural_features=Config.SVD_COMPONENTS,
        hidden_size=Config.HIDDEN_SIZE,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(device)

    # 4. Optimizer and Scheduler
    optimizer_parameters = [
        {
            "params": model.backbone.parameters(),
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": model.fusion_head.parameters(),
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = AdamW(optimizer_parameters)
    criterion = nn.BCEWithLogitsLoss()

    num_train_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_runfile.bin")

    # Limit epochs if needed for "fast baseline", but Config.NUM_EPOCHS=5 is already reasonable for 3k samples.
    # We will stick to Config.NUM_EPOCHS.

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_auc = eval_fn(model, val_loader, device, criterion)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_auc, best_model_path)

    # 6. Final Validation & Metric
    # Load best model for evaluation
    checkpoint = load_checkpoint(best_model_path, model, device=device)
    model.eval()

    # Re-run evaluation on validation set to get predictions for failure analysis
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            structural_features = batch["structural_features"].to(device)
            targets = batch["label"].to(device)

            logits = model(input_ids, attention_mask, structural_features)
            preds = torch.sigmoid(logits).view(-1).cpu().numpy()

            val_preds.extend(preds)
            val_targets.extend(targets.cpu().numpy())

    final_val_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_df = pd.read_csv(Config.VAL_DATA_PATH)

    # Calculate absolute error
    errors = np.abs(np.array(val_targets) - np.array(val_preds))

    # Feature 1: Comment Length
    val_df["comment_len"] = val_df["Comment"].fillna("").apply(len)

    # Feature 2: Structural Feature Magnitude (L2 norm of the SVD vector)
    # We need to access the structural features corresponding to the validation set.
    # Since the dataset order is preserved, we can retrieve them from the dataset object.
    # val_dataset.structural_features is a numpy array or list of arrays.
    struct_feats = val_dataset.structural_features
    if isinstance(struct_feats, list):
        struct_feats = np.array(struct_feats)
    struct_magnitudes = np.linalg.norm(struct_feats, axis=1)

    # Compute correlations
    corr_len = np.corrcoef(errors, val_df["comment_len"])[0, 1]
    corr_struct = np.corrcoef(errors, struct_magnitudes)[0, 1]

    print(f"Correlation between Error and Comment Length: {corr_len}")
    print(f"Correlation between Error and Structural Feature Magnitude: {corr_struct}")

    # 8. Conditional Submission
    THRESHOLD = 0.9539080459770114

    if final_val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_val_auc}) > threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = predict_fn(model, test_loader, device)

        # Load test metadata to ensure correct format
        test_df = pd.read_csv(Config.TEST_DATA_PATH)

        # Fill predictions
        test_df["Insult"] = test_preds

        # Ensure columns exist (Date and Comment should be there from metadata)
        if "Date" not in test_df.columns:
            test_df["Date"] = ""
        if "Comment" not in test_df.columns:
            test_df["Comment"] = ""

        # Select required columns
        submission_cols = ["Insult", "Date", "Comment"]
        submission_df = test_df[submission_cols]

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation metric ({final_val_auc}) <= threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
