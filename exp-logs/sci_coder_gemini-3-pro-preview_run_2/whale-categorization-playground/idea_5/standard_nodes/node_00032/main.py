import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import WhaleModel
from library.loss import ArcFaceLoss
from library.engine import train_fn, eval_fn, extract_features, generate_submission
from library.rerank import re_ranking


def main():
    # ---------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------
    seed_everything(Config.seed)
    device = Config.device
    print(f"Device: {device}")

    # Limit epochs for fast baseline execution
    # Config.epochs is 25, we reduce to 10 to ensure runtime constraints
    NUM_EPOCHS = Config.epochs

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    # Load cached data to speed up initialization
    (
        train_loader,
        val_loader,
        gallery_loader,
        test_loader,
        num_classes,
        label_encoder,
    ) = get_loaders(load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Model & Loss Initialization
    # ---------------------------------------------------------
    print("Initializing Model and Loss...")
    model = WhaleModel(
        embedding_size=Config.embedding_size, pretrained=Config.pretrained
    )
    model.to(device)

    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=Config.embedding_size,
        s=Config.arc_s,
        m=Config.arc_m,
    )
    criterion.to(device)

    # ---------------------------------------------------------
    # 4. Optimizer & Scheduler
    # ---------------------------------------------------------
    # Optimize both the backbone/head weights and the ArcFace centers
    optimizer = optim.AdamW(
        [{"params": model.parameters()}, {"params": criterion.parameters()}],
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    # Scheduler to reduce LR when MAP@5 plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.scheduler_factor,
        patience=Config.scheduler_patience,
        min_lr=Config.min_lr,
    )

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_score = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Train Step
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, device, scheduler=None
        )

        # Validation Step
        # Note: eval_fn computes MAP@5 using the gallery
        val_score = eval_fn(val_loader, gallery_loader, model, device, label_encoder)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss:.6f} | Val MAP@5: {val_score:.10f}"
        )

        # Scheduler Step
        scheduler.step(val_score)

        # Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"New best model saved with MAP@5: {best_score:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # ---------------------------------------------------------
    # 6. Final Evaluation
    # ---------------------------------------------------------
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    # Re-compute metric to ensure exact value is captured
    final_metric = eval_fn(val_loader, gallery_loader, model, device, label_encoder)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # ---------------------------------------------------------
    # 7. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Extract features for analysis
    query_feats, query_targets_enc = extract_features(val_loader, model, device)
    gallery_feats, gallery_targets_enc = extract_features(gallery_loader, model, device)

    # Decode labels
    query_labels = []
    for t in query_targets_enc:
        if t == -1:
            query_labels.append("new_whale")
        else:
            query_labels.append(label_encoder.inverse_transform([t])[0])
    gallery_labels = label_encoder.inverse_transform(gallery_targets_enc)

    # Compute Confidence (Max Cosine Similarity)
    q_norm = F.normalize(query_feats, p=2, dim=1)
    g_norm = F.normalize(gallery_feats, p=2, dim=1)
    sim_matrix = torch.matmul(q_norm, g_norm.T)
    max_sim_vals, _ = torch.max(sim_matrix, dim=1)
    max_sim_vals = max_sim_vals.numpy()

    # Compute Per-Sample AP and Error
    dist_matrix = re_ranking(query_feats, gallery_feats)

    errors = []
    for i in range(len(query_labels)):
        dists = dist_matrix[i]
        sorted_indices = np.argsort(dists)
        top_candidates = [gallery_labels[idx] for idx in sorted_indices[:5]]

        # Apply Open-Set Logic for consistency with inference
        if max_sim_vals[i] < Config.new_whale_threshold:
            preds = ["new_whale"] + top_candidates[:4]
        else:
            preds = top_candidates

        # Calculate AP for this query
        true_label = query_labels[i]
        if true_label in preds:
            rank = preds.index(true_label) + 1
            ap = 1.0 / rank
        else:
            ap = 0.0

        # Error Magnitude = 1 - AP
        errors.append(1.0 - ap)

    errors = np.array(errors)

    # Calculate Correlation
    if len(errors) > 1:
        # Using numpy for correlation
        corr_matrix = np.corrcoef(errors, max_sim_vals)
        correlation = corr_matrix[0, 1]
        print(
            f"Correlation between Error Magnitude (1-AP) and Confidence (Max Sim): {correlation:.6f}"
        )
    else:
        print("Insufficient samples for failure analysis correlation.")

    # ---------------------------------------------------------
    # 8. Conditional Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.756541
    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric:.6f}) > {THRESHOLD}. Generating submission..."
        )
        generate_submission(test_loader, gallery_loader, model, device, label_encoder)
    else:
        print(
            f"\nValidation metric ({final_metric:.6f}) <= {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
