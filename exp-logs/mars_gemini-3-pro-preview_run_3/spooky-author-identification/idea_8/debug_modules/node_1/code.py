import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import library modules
from library.config import PathConfig, ModelConfig, TrainConfig, FeatureConfig
from library.utils import set_seed, calculate_log_loss
from library.data_processing import (
    load_data,
    get_tfidf_features,
    StylometricDataset,
    MLMDataset,
    get_auxiliary_targets,
)
from library.dapt import run_mlm_pretraining, get_mlm_corpus
from library.models import StylometricTransformer, StatisticalPredictor
from library.training_utils import run_fold_training
from library.ensemble import LengthAdaptiveBlender


def main():
    print("--- Starting Demonstration Script ---")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment for fast demonstration...")
    set_seed(42)

    # Override paths to use a specific demo directory within working
    PathConfig.WORKING_DIR = "./working/demo_run"
    PathConfig.create_dirs()

    # Override TrainConfig for speed
    TrainConfig.EPOCHS = 1
    TrainConfig.DAPT_EPOCHS = 1
    TrainConfig.BATCH_SIZE = 4
    TrainConfig.DAPT_BATCH_SIZE = 4
    TrainConfig.DEBUG = True
    TrainConfig.USE_AWP = False  # Disable AWP for speed

    # Override ModelConfig to use a tiny model
    # 'prajjwal1/bert-tiny' is very small (2 layers, 128 hidden)
    ModelConfig.BACKBONES = ["prajjwal1/bert-tiny"]
    # Disable last 4 layers pooling as bert-tiny only has 2 layers
    ModelConfig.USE_LAST_4_LAYERS = False

    print("Configuration patched: Using 'prajjwal1/bert-tiny' and reduced epochs.")

    # 2. Data Loading and Slicing
    print("\n[2] Loading and preprocessing data...")
    train_df, val_df, test_df = load_data()

    # Slice data to a tiny subset for demonstration
    subset_size = 50
    train_subset = train_df.iloc[:subset_size].copy()
    val_subset = val_df.iloc[:subset_size].copy()
    test_subset = test_df.iloc[:subset_size].copy()

    print(f"Data loaded. Using subset of size {subset_size} for demo.")

    # 3. TF-IDF Feature Generation
    print("\n[3] Generating TF-IDF features...")
    # Force re-computation by setting load_cached_data=False (or ensure cache dir is clean)
    # We use the patched WORKING_DIR so it should be clean.
    X_train_tfidf, X_val_tfidf, X_test_tfidf = get_tfidf_features(
        train_subset["text"],
        val_subset["text"],
        test_subset["text"],
        load_cached_data=False,
    )

    # Validation
    assert X_train_tfidf.shape[0] == subset_size
    assert X_val_tfidf.shape[0] == subset_size
    assert X_test_tfidf.shape[0] == subset_size
    print(f"TF-IDF Shapes verified: {X_train_tfidf.shape}")

    # 4. Statistical Model Demonstration
    print("\n[4] Training Statistical Predictor...")
    stat_model = StatisticalPredictor(seed=42)

    # Map string labels to integers for sklearn
    label_map = {l: i for i, l in enumerate(ModelConfig.LABELS)}
    y_train_indices = train_subset["author"].map(label_map).values

    stat_model.fit(X_train_tfidf, y_train_indices)

    # Predict
    stat_preds_val = stat_model.predict_proba(X_val_tfidf)
    stat_preds_test = stat_model.predict_proba(X_test_tfidf)

    # Validation
    assert stat_preds_val.shape == (subset_size, 3)
    assert np.allclose(stat_preds_val.sum(axis=1), 1.0)
    print("Statistical model training and prediction successful.")

    # 5. DAPT (MLM) Demonstration
    print("\n[5] Running DAPT (MLM) in debug mode...")
    # This will use the patched BACKBONES list
    run_mlm_pretraining(debug=True, load_cached_data=False)

    # Verify DAPT output exists
    safe_name = ModelConfig.BACKBONES[0].replace("/", "-")
    dapt_model_path = os.path.join(PathConfig.MLM_MODELS_DIR, f"mlm_{safe_name}")
    assert os.path.exists(
        os.path.join(dapt_model_path, "model.safetensors")
    ) or os.path.exists(os.path.join(dapt_model_path, "pytorch_model.bin"))
    print("DAPT execution completed and model saved.")

    # 6. Neural Model Training Demonstration
    print("\n[6] Training Neural Model (StylometricTransformer)...")

    # Prepare Datasets
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.BACKBONES[0])

    train_dataset = StylometricDataset(
        texts=train_subset["text"],
        labels=train_subset["author"],
        tokenizer=tokenizer,
        max_length=128,  # Reduced max length for speed
    )
    val_dataset = StylometricDataset(
        texts=val_subset["text"],
        labels=val_subset["author"],
        tokenizer=tokenizer,
        max_length=128,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=False
    )

    # Initialize Model
    model = StylometricTransformer(
        backbone_name=ModelConfig.BACKBONES[0],
        num_labels=ModelConfig.NUM_LABELS,
        mtl_head_dim=ModelConfig.MTL_HEAD_DIM,
    )

    # Run Training for 1 fold (which is just 1 epoch here due to config override)
    best_val_loss, neural_preds_val = run_fold_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=0,
        backbone_name=ModelConfig.BACKBONES[0],
        epochs=TrainConfig.EPOCHS,
    )

    # Generate Test Predictions with the trained model
    print("Generating Neural Test Predictions...")
    test_dataset = StylometricDataset(
        texts=test_subset["text"],
        labels=None,  # No labels for test
        tokenizer=tokenizer,
        max_length=128,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=False
    )

    model.eval()
    neural_preds_test = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                k: v.to(TrainConfig.DEVICE)
                for k, v in batch.items()
                if k != "aux_targets"
            }
            outputs = model(**inputs)
            probs = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
            neural_preds_test.append(probs)
    neural_preds_test = np.concatenate(neural_preds_test, axis=0)

    # Validation
    assert neural_preds_val.shape == (subset_size, 3)
    assert neural_preds_test.shape == (subset_size, 3)
    print(f"Neural model training complete. Best Val Loss: {best_val_loss:.4f}")

    # 7. Ensemble Demonstration
    print("\n[7] Running Length-Adaptive Blender...")

    # Prepare dictionaries for the blender
    # Keys should match model names
    oof_preds = {"statistical": stat_preds_val, "neural": neural_preds_val}
    test_preds_dict = {"statistical": stat_preds_test, "neural": neural_preds_test}

    y_val_indices = val_subset["author"].map(label_map).values

    blender = LengthAdaptiveBlender(n_bins=2, seed=42)  # 2 bins for small data

    # Fit Blender
    blender.fit(oof_preds, y_val_indices, val_subset["text"])

    # Predict
    final_preds = blender.predict(test_preds_dict, test_subset["text"])

    # Validation
    assert final_preds.shape == (subset_size, 3)
    # Check if rows sum to ~1
    row_sums = final_preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)

    print("Ensemble blending complete.")

    # 8. Submission Generation
    print("\n[8] Generating Submission File...")
    submission_path = os.path.join(PathConfig.WORKING_DIR, "demo_submission.csv")
    blender.generate_submission(
        ids=test_subset["id"], probabilities=final_preds, output_path=submission_path
    )

    assert os.path.exists(submission_path)
    print(f"Demo script completed successfully. Output at {submission_path}")


if __name__ == "__main__":
    main()
