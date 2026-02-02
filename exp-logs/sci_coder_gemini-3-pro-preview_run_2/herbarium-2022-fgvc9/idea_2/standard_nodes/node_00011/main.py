import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure local library is importable
sys.path.append(os.getcwd())

from library.config import Config
from library.data_setup import get_dataloaders, TaxonomyProcessor
from library.model import HierarchicalEfficientNet
from library.trainer import Trainer
from library.utils import set_seed, calculate_macro_f1
from library.inference import predict_test_set


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override epochs to ensure completion within the "Fast Baseline" time limit.
    # 5 epochs on A100 with batch size 256 takes approximately 40-50 minutes.
    Config.NUM_EPOCHS = 5

    set_seed(Config.SEED)
    print(f"Running baseline with {Config.NUM_EPOCHS} epochs on {Config.DEVICE}...")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    train_loader, val_loader, counts = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Initializing Hierarchical Multi-Task Model...")
    model = HierarchicalEfficientNet(
        num_families=counts["num_families"],
        num_genera=counts["num_genera"],
        num_species=counts["num_species"],
        pretrained=True,
    )

    print("Starting training loop...")
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # ==========================================
    # 4. Final Validation & Metric Calculation
    # ==========================================
    print("Computing final validation metrics...")
    # Load best model for validation logic if needed, but trainer.model is the latest state.
    # Ideally we evaluate the *best* model saved.

    checkpoint_path = Config.MODEL_SAVE_PATH
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        state_dict = checkpoint["model_state_dict"]
        # Handle module prefix if present
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with F1: {checkpoint['score']}"
        )

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)
            # Target 0 is species
            species_targets = targets[0].to(Config.DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                # We use the species head for the primary metric
                preds = torch.argmax(outputs["species"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(species_targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    final_f1 = calculate_macro_f1(all_targets, all_preds)
    print(f"Final Validation Metric: {final_f1}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n==== Failure Analysis ====")
    # 1. Error Vector (0 = Correct, 1 = Incorrect)
    errors = (all_preds != all_targets).astype(int)

    # 2. Correlation with Class Frequency
    # Load training data to get class counts
    train_df = pd.read_csv(Config.TRAIN_CSV)
    class_counts = train_df["label"].value_counts().to_dict()
    target_freqs = np.array([class_counts.get(t, 0) for t in all_targets])

    if len(np.unique(errors)) > 1:
        corr_freq = np.corrcoef(errors, target_freqs)[0, 1]
        print(f"Correlation (Error vs Class Frequency): {corr_freq:.4f}")
    else:
        print("Correlation (Error vs Class Frequency): N/A (No variance in errors)")

    # 3. Correlation with Genus Diversity (Taxonomic Complexity)
    processor = TaxonomyProcessor()
    mapping_df, _ = processor.process_taxonomy(load_cached_data=True)
    # Map species_id -> genus_id
    species_to_genus = dict(zip(mapping_df["category_id"], mapping_df["genus_id"]))
    # Count species per genus
    genus_sizes = mapping_df["genus_id"].value_counts().to_dict()

    target_genus_sizes = []
    for t in all_targets:
        g_id = species_to_genus.get(t, -1)
        size = genus_sizes.get(g_id, 0)
        target_genus_sizes.append(size)
    target_genus_sizes = np.array(target_genus_sizes)

    if len(np.unique(errors)) > 1:
        corr_genus = np.corrcoef(errors, target_genus_sizes)[0, 1]
        print(f"Correlation (Error vs Genus Size): {corr_genus:.4f}")
    else:
        print("Correlation (Error vs Genus Size): N/A")

    # ==========================================
    # 6. Submission
    # ==========================================
    THRESHOLD = 0.6021914648406147

    if final_f1 > THRESHOLD:
        print(
            f"\nMetric {final_f1} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        predict_test_set()
    else:
        print(
            f"\nMetric {final_f1} does not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
