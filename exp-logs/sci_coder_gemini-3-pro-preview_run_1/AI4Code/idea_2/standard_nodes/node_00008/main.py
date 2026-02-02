import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.feature_extraction import extract_features
from library.dataset import HAPSDataset, haps_collate_fn
from library.model import HAPSModel
from library.loss import HAPSLoss
from library.inference_utils import (
    compute_global_sort,
    refine_order,
    generate_predictions,
)
from library.train import set_seed, compute_kendall_tau


def main():
    # 1. Configuration for Full Training
    # We override default config settings to optimize performance
    Config.num_epochs = 10  # Increased to reach convergence (~0.66 KT) Cite {solution_lesson_node_00006}
    Config.batch_size = (
        64  # Optimal batch size for adapter capacity Cite {solution_lesson_node_00004}
    )

    # Set seeds for reproducibility
    set_seed(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Feature Extraction
    # This will use cached parquet files if they exist, or generate them otherwise
    extract_features(load_cached_data=True)

    # 3. Data Loading
    # Load full datasets
    full_train_dataset = HAPSDataset(Config.train_features_path, mode="train")
    val_dataset = HAPSDataset(Config.val_features_path, mode="val")

    # Use the full training dataset to maximize performance Cite {solution_lesson_node_00003}
    train_dataset = full_train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=haps_collate_fn,
        pin_memory=True,
    )

    # 4. Model Initialization
    model = HAPSModel().to(device)
    criterion = HAPSLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

    # 5. Training Loop
    best_val_score = -float("inf")

    # We use a small subset of validation data for epoch-level monitoring to save time
    val_monitor_size = min(len(val_dataset), 1000)
    val_monitor_indices = np.random.choice(
        len(val_dataset), val_monitor_size, replace=False
    )

    for epoch in range(Config.num_epochs):
        model.train()

        for batch in train_loader:
            # Move data to device
            code_emb = batch["code_embeddings"].to(device)
            code_mask = batch["code_mask"].to(device)
            md_emb = batch["md_embeddings"].to(device)
            md_mask = batch["md_mask"].to(device)

            anchor_labels = batch.get("anchor_labels")
            if anchor_labels is not None:
                anchor_labels = anchor_labels.to(device)

            pairwise_indices = batch.get("pairwise_indices")
            if pairwise_indices is not None:
                pairwise_indices = pairwise_indices.to(device)

            pairwise_labels = batch.get("pairwise_labels")
            if pairwise_labels is not None:
                pairwise_labels = pairwise_labels.to(device)

            batch_data = {
                "anchor_labels": anchor_labels,
                "pairwise_labels": pairwise_labels,
            }

            # Forward & Backward
            optimizer.zero_grad()
            outputs = model(code_emb, code_mask, md_emb, md_mask, pairwise_indices)
            loss_dict = criterion(outputs, batch_data)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

        # Quick Validation Monitoring
        model.eval()
        total_swaps = 0
        total_denom = 0

        with torch.no_grad():
            for idx in val_monitor_indices:
                sample = val_dataset[idx]
                nb_id = sample["notebook_id"]

                # Prepare inputs
                code_emb = sample["code_embeddings"].unsqueeze(0).to(device)
                md_emb = sample["md_embeddings"].unsqueeze(0).to(device)
                n_code = code_emb.size(1)
                n_md = md_emb.size(1)

                code_mask = torch.ones((1, n_code), dtype=torch.bool, device=device)
                md_mask = torch.ones((1, n_md), dtype=torch.bool, device=device)

                # Get ground truth and IDs
                nb_df = val_dataset.grouped.get_group(nb_id)
                gt_order = nb_df.sort_values("rank")["cell_id"].tolist()
                code_ids = nb_df[nb_df["cell_type"] == "code"]["cell_id"].tolist()
                md_ids = nb_df[nb_df["cell_type"] == "markdown"]["cell_id"].tolist()

                # Inference
                outputs = model(code_emb, code_mask, md_emb, md_mask)
                all_cells = compute_global_sort(
                    outputs["anchor_logits"], code_ids, md_ids
                )

                if n_md > 1:
                    pred_order = refine_order(
                        model, all_cells, md_emb, device, passes=1
                    )
                else:
                    pred_order = [c["cell_id"] for c in all_cells]

                s, n = compute_kendall_tau(gt_order, pred_order)
                total_swaps += s
                total_denom += n * (n - 1)

        val_score = 1 - 4 * (total_swaps / total_denom) if total_denom > 0 else 0.0

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)

    # 6. Final Validation & Failure Analysis
    # Load the best model
    if os.path.exists(Config.model_save_path):
        model.load_state_dict(torch.load(Config.model_save_path))

    model.eval()
    total_swaps = 0
    total_denom = 0
    analysis_data = []

    # Iterate through the FULL validation set
    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            nb_id = sample["notebook_id"]

            code_emb = sample["code_embeddings"].unsqueeze(0).to(device)
            md_emb = sample["md_embeddings"].unsqueeze(0).to(device)
            n_code = code_emb.size(1)
            n_md = md_emb.size(1)

            code_mask = torch.ones((1, n_code), dtype=torch.bool, device=device)
            md_mask = torch.ones((1, n_md), dtype=torch.bool, device=device)

            nb_df = val_dataset.grouped.get_group(nb_id)
            gt_order = nb_df.sort_values("rank")["cell_id"].tolist()
            code_ids = nb_df[nb_df["cell_type"] == "code"]["cell_id"].tolist()
            md_ids = nb_df[nb_df["cell_type"] == "markdown"]["cell_id"].tolist()

            outputs = model(code_emb, code_mask, md_emb, md_mask)
            all_cells = compute_global_sort(outputs["anchor_logits"], code_ids, md_ids)

            if n_md > 1:
                pred_order = refine_order(model, all_cells, md_emb, device, passes=2)
            else:
                pred_order = [c["cell_id"] for c in all_cells]

            s, n = compute_kendall_tau(gt_order, pred_order)

            # Accumulate for global metric
            total_swaps += s
            total_denom += n * (n - 1)

            # Accumulate for failure analysis
            nb_denom = n * (n - 1)
            nb_kt = 1 - 4 * (s / nb_denom) if nb_denom > 0 else 1.0
            analysis_data.append({"n_code": n_code, "n_md": n_md, "error": 1.0 - nb_kt})

    final_metric = 1 - 4 * (total_swaps / total_denom) if total_denom > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    df_analysis = pd.DataFrame(analysis_data)
    if not df_analysis.empty:
        corr_code = df_analysis["n_code"].corr(df_analysis["error"])
        corr_md = df_analysis["n_md"].corr(df_analysis["error"])
        print(f"Error vs n_code Correlation: {corr_code}")
        print(f"Error vs n_md Correlation: {corr_md}")

    # 7. Submission
    # Conditional generation based on metric threshold
    threshold = 0.6598830915782636
    if final_metric > threshold:
        test_dataset = HAPSDataset(Config.test_features_path, mode="test")
        submission_df = generate_predictions(model, test_dataset, device)
        submission_df.to_csv(Config.submission_path, index=False)


if __name__ == "__main__":
    main()
