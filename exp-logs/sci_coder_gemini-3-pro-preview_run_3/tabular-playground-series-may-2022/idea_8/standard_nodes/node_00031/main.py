import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from library import config
from library import utils
from library import data
from library import model
from library import engine


def main():
    # 1. Setup Environment
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Preparation
    print("Initializing DataProcessor...")
    processor = data.DataProcessor()

    # Load data using the library's processor.
    # We use debug=False to train on the full dataset to ensure we meet the high AUC threshold.
    # load_cached_data=True allows skipping processing if artifacts exist.
    train_loader, val_loader, test_loader, vocab_sizes = processor.get_dataloaders(
        load_cached_data=True, debug=False
    )

    # Infer input dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    cont_dim = sample_batch["cont"].shape[1]
    print(
        f"Data loaded. Continuous features: {cont_dim}, Categorical features: {len(vocab_sizes)}"
    )

    # 3. Model Initialization
    print("Initializing ManufacturingMLP (Early Fusion)...")
    net = model.ManufacturingMLP(
        vocab_sizes=vocab_sizes,
        cont_dim=cont_dim,
        embed_dim=config.EMBED_DIM,
        backbone_layers=config.BACKBONE_LAYERS,
        dropout=config.DROPOUT,
        output_dim=config.OUTPUT_DIM,
    )

    # 4. Training
    # The engine handles the training loop, validation, early stopping, and saving the best model.
    # It also generates a submission file at the end using the test set.
    print("Starting Training Engine...")
    engine.train_engine(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=config.EPOCHS,
        max_lr=config.MAX_LR,
        weight_decay=config.WEIGHT_DECAY,
        patience=config.PATIENCE,
    )

    # 5. Validation Assessment & Failure Analysis
    print("\nPerforming Final Validation Assessment...")

    # Load the best model saved during training
    if os.path.exists(config.MODEL_SAVE_PATH):
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    net.to(device)
    net.eval()

    val_targets = []
    val_preds = []
    val_inputs_cont = []

    # Run inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            cat_x = batch["cat"].to(device)
            cont_x = batch["cont"].to(device)
            targets = batch["target"].to(device)

            logits = net(cat_x, cont_x)
            probs = torch.sigmoid(logits)

            val_targets.append(targets.cpu().numpy())
            val_preds.append(probs.cpu().numpy())
            val_inputs_cont.append(cont_x.cpu().numpy())

    # Flatten results
    val_targets = np.concatenate(val_targets).flatten()
    val_preds = np.concatenate(val_preds).flatten()
    val_inputs_cont = np.concatenate(val_inputs_cont, axis=0)

    # Compute and print the required metric
    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation of Error with Features
    print("\n=== Failure Analysis ===")
    errors = np.abs(val_targets - val_preds)

    # Retrieve feature names to make analysis readable
    # We use the cached validation dataframe to get the column names
    val_df_path = processor.val_cache
    if os.path.exists(val_df_path):
        val_df_ref = pd.read_parquet(val_df_path)

        # Reconstruct the list of continuous columns used in the dataset
        cat_keys = list(vocab_sizes.keys())
        exclude = set(
            cat_keys
            + [config.ID_COL, config.TARGET_COL, config.STRING_COL, "source_path"]
        )
        # The DataProcessor sorts continuous columns, so we must sort here too
        cont_names = sorted([c for c in val_df_ref.columns if c not in exclude])

        correlations = []
        for i, name in enumerate(cont_names):
            if i < val_inputs_cont.shape[1]:
                feat_values = val_inputs_cont[:, i]
                # Calculate correlation if variance exists
                if np.std(feat_values) > 1e-9 and np.std(errors) > 1e-9:
                    corr = np.corrcoef(errors, feat_values)[0, 1]
                else:
                    corr = 0.0
                correlations.append((name, corr))

        # Sort by magnitude of correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 5 Features correlated with Error Magnitude:")
        for name, corr in correlations[:5]:
            print(f"{name}: {corr:.4f}")
    else:
        print(
            "Warning: Cached validation data not found. Skipping detailed feature correlation."
        )

    # 6. Submission Check
    # The engine generated the submission file automatically.
    # We verify if the metric meets the strict requirement.
    threshold = 0.9971550270448856

    if final_auc > threshold:
        print(
            f"\nSuccess: Validation AUC ({final_auc}) exceeds threshold ({threshold})."
        )
        print(f"Submission saved to {config.SUBMISSION_FILE}")
    else:
        print(
            f"\nCondition Failed: Validation AUC ({final_auc}) does not exceed threshold ({threshold})."
        )
        if os.path.exists(config.SUBMISSION_FILE):
            print("Removing submission file...")
            os.remove(config.SUBMISSION_FILE)
            print("Submission file removed.")


if __name__ == "__main__":
    main()
