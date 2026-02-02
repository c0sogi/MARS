import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.preprocess import preprocess_data
from library.dataset import CachedNotebookDataset, custom_collate_fn
from library.model import DualContextAnchorNetwork
from library.engine import train_model, validate
from library.inference import predict_and_rank


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def run_failure_analysis(model, dataloader, device):
    """
    Analyzes model performance against notebook features.
    Computes correlation between Error (1 - Tau) and structural features.
    """
    model.eval()

    taus = []
    num_code_list = []
    num_md_list = []

    print("\nRunning Failure Analysis on Validation Set...")

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            code_emb = batch["code_embeddings"].to(device)
            code_lens = batch["code_lens"].to(device)
            code_mask = batch["code_padding_mask"].to(device)
            md_emb = batch["markdown_embeddings"].to(device)
            md_lens = batch["md_lens"].to(device)
            md_mask = batch["md_padding_mask"].to(device)

            # Forward pass
            logits = model(code_emb, code_lens, code_mask, md_emb, md_lens, md_mask)
            probs = torch.softmax(logits, dim=-1)

            # Compute Expected Index
            max_c = probs.size(-1)
            indices = torch.arange(max_c, device=device).float()
            pred_scores = torch.sum(probs * indices, dim=-1)

            pred_scores_np = pred_scores.cpu().numpy()
            md_lens_np = md_lens.cpu().numpy()
            code_lens_np = code_lens.cpu().numpy()

            # Calculate Tau per notebook
            for i in range(len(pred_scores_np)):
                length = md_lens_np[i]
                n_code = code_lens_np[i]

                num_md_list.append(length)
                num_code_list.append(n_code)

                if length < 2:
                    taus.append(1.0)
                else:
                    scores = pred_scores_np[i, :length]
                    ground_truth = np.arange(length)
                    t, _ = kendalltau(ground_truth, scores)
                    if np.isnan(t):
                        t = 0.0
                    taus.append(t)

    # Compute correlations
    errors = 1.0 - np.array(taus)
    num_code = np.array(num_code_list)
    num_md = np.array(num_md_list)

    # Pearson Correlation
    if len(errors) > 1:
        corr_code, _ = pearsonr(errors, num_code)
        corr_md, _ = pearsonr(errors, num_md)

        print(f"Correlation between Error and Num Code Cells: {corr_code:.4f}")
        print(f"Correlation between Error and Num Markdown Cells: {corr_md:.4f}")

        if corr_code > 0.1:
            print(
                "Observation: Model performance degrades on notebooks with many code cells."
            )
        elif corr_code < -0.1:
            print(
                "Observation: Model performs better on notebooks with many code cells."
            )

        if corr_md > 0.1:
            print(
                "Observation: Model performance degrades on notebooks with many markdown cells."
            )
    else:
        print("Insufficient data for correlation analysis.")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Override Config for fast baseline execution
    Config.NUM_EPOCHS = 3
    print(
        f"Configuration: Device={Config.DEVICE}, Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )

    # 2. Preprocess Data
    # This will load from cache if available, or compute if not.
    print("Step 1: Preprocessing Data...")
    preprocess_data(load_cached_data=True)

    # 3. Prepare Datasets
    print("Step 2: Loading Datasets...")
    train_dataset = CachedNotebookDataset(split="train")
    val_dataset = CachedNotebookDataset(split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    # 4. Initialize Model and Optimizer
    print("Step 3: Initializing Model...")
    model = DualContextAnchorNetwork().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Train
    print("Step 4: Training...")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        Config.DEVICE,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 6. Validate
    print("Step 5: Final Validation...")
    # Load best model
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)

    val_loss, val_tau = validate(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_tau}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, Config.DEVICE)

    # 8. Inference & Submission
    THRESHOLD = 0.8315021559000814

    if val_tau > THRESHOLD:
        print(
            f"\nMetric ({val_tau}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_rank()
    else:
        print(
            f"\nMetric ({val_tau}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
