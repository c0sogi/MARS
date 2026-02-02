import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import GranularSiameseModel
from library.engine import train_one_epoch, evaluate


def main():
    # 1. Setup and Configuration
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # Override Config for fast baseline execution
    # Reducing epochs to 3 ensures the run completes well within the 2-hour limit
    # while still providing enough training signal for this dataset size.
    Config.EPOCHS = 3
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8

    print(
        f"Configuration: Device={Config.DEVICE}, Epochs={Config.EPOCHS}, Batch Size={Config.TRAIN_BATCH_SIZE}"
    )

    # 2. Data Loading
    print("Loading data...")
    # load_cached_data=True utilizes pre-processed .npz files if available
    train_loader, val_loader, test_loader, cat_dims = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = GranularSiameseModel(cat_dims=cat_dims)
    model.to(Config.DEVICE)

    # 4. Optimizer Setup
    # Using differential learning rates as defined in Config/Model strategy
    optimizer_grouped_parameters = [
        {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
        {"params": model.cat_embeddings.parameters(), "lr": Config.LR_HEAD},
        {"params": model.head.parameters(), "lr": Config.LR_HEAD},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training Loop
    best_val_score = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_runfile.pth")

    print("Starting training loop...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, Config.DEVICE, max_grad_norm=1.0
        )

        # Validate
        val_loss, val_score = evaluate(model, val_loader, Config.DEVICE)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Spearman: {val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Validation & Failure Analysis
    print("\n--- Final Evaluation & Failure Analysis ---")

    # Load best model weights
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Re-compute metric on full validation set to be precise
    _, final_metric = evaluate(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlate error with text length
    # Collect predictions and targets manually to compute row-wise error
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = [b.to(Config.DEVICE) for b in batch]
            # Unpack inputs matching GranularSiameseModel.forward arguments
            # Indices: 1=q_input_ids, ..., 9=cat_feats, 10=targets
            preds = model(
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[5],
                batch[6],
                batch[7],
                batch[8],
                batch[9],
            )
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch[10].cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Compute Mean Absolute Error per sample (averaged over 30 targets)
    sample_errors = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Load validation metadata to get features
    val_df = pd.read_csv(Config.VAL_PATH)

    # Calculate lengths
    val_df["q_len"] = val_df["question_body"].fillna("").astype(str).apply(len)
    val_df["a_len"] = val_df["answer"].fillna("").astype(str).apply(len)

    # Compute correlations
    corr_q_len, _ = spearmanr(sample_errors, val_df["q_len"])
    corr_a_len, _ = spearmanr(sample_errors, val_df["a_len"])

    print(f"Correlation between Error and Question Length: {corr_q_len:.4f}")
    print(f"Correlation between Error and Answer Length: {corr_a_len:.4f}")

    # 7. Submission Generation
    THRESHOLD = 0.41003785424660755

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds_list = []
        test_qa_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch = [b.to(Config.DEVICE) for b in batch]
                # Index 0 is qa_id
                qa_ids = batch[0]
                # Forward pass
                preds = model(
                    batch[1],
                    batch[2],
                    batch[3],
                    batch[4],
                    batch[5],
                    batch[6],
                    batch[7],
                    batch[8],
                    batch[9],
                )

                test_preds_list.append(preds.cpu().numpy())
                test_qa_ids_list.extend(qa_ids.cpu().numpy())

        test_preds = np.vstack(test_preds_list)

        # Create DataFrame
        sub_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
        sub_df.insert(0, "qa_id", test_qa_ids_list)

        # Save
        save_path = os.path.join(submission_dir, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
