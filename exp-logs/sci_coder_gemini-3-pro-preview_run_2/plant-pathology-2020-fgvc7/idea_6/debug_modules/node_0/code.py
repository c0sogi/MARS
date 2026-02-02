import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_loaders, get_test_loader, get_class_weights
from library.model import AppleDiseaseModel
from library.engine import fit
from library.inference import predict_ensemble, generate_submission

if __name__ == "__main__":
    print("Starting Apple Disease Detection Pipeline Demo...")

    # 1. Configuration
    # We enable debug mode to use a small subset (50 images) and reduce epochs for speed.
    cfg = Config(debug=True, epochs=1, batch_size=4)

    # Override model name to a lightweight model for rapid demonstration purposes
    # efficientnetv2_l is very large; resnet18 is sufficient to test the pipeline logic.
    cfg.model_name = "resnet18"

    # Ensure working directory for this run exists
    os.makedirs(cfg.idea_dir, exist_ok=True)

    print(
        f"Configuration: Debug={cfg.debug}, Device={cfg.device}, Model={cfg.model_name}"
    )

    # 2. Reproducibility
    seed_everything(cfg.seed)

    # 3. Data Loading (Fold 0)
    print("\n--- Step 1: Data Loading ---")
    # load_cached_data=False forces processing to demonstrate the raw data handling
    train_loader, val_loader = get_loaders(fold=0, cfg=cfg, load_cached_data=False)

    # Verification: Check batch shapes
    # We expect: Inputs (Batch, 3, 480, 480), Targets (Batch, 2)
    sample_imgs, sample_targets = next(iter(train_loader))
    print(
        f"Train Batch Shape: Images {sample_imgs.shape}, Targets {sample_targets.shape}"
    )

    assert sample_imgs.shape == (
        cfg.batch_size,
        3,
        cfg.img_size,
        cfg.img_size,
    ), "Incorrect image batch shape"
    assert sample_targets.shape == (
        cfg.batch_size,
        2,
    ), "Incorrect target batch shape (Expected 2 for Rust/Scab)"
    print("Data Loading Verified.")

    # 4. Model Initialization
    print("\n--- Step 2: Model Initialization ---")
    # pretrained=False to avoid downloading weights during this timed demo
    model = AppleDiseaseModel(
        model_name=cfg.model_name, num_classes=cfg.num_classes, pretrained=False
    )
    device = get_device()
    model.to(device)

    # Verification: Check forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, cfg.img_size, cfg.img_size).to(device)
        dummy_output = model(dummy_input)
        assert dummy_output.shape == (
            2,
            2,
        ), f"Model output shape mismatch. Expected (2, 2), got {dummy_output.shape}"
    print("Model Initialization Verified.")

    # 5. Training Loop (Engine)
    print("\n--- Step 3: Training (Simulation) ---")
    # Calculate class weights for the loss function
    pos_weights = get_class_weights(fold=0, cfg=cfg, load_cached_data=False).to(device)
    print(f"Class Weights: {pos_weights.cpu().numpy()}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Run training for 1 epoch (as configured)
    # Passing None for scheduler as we don't need it for a 1-epoch demo
    trained_model, best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        criterion=criterion,
        cfg=cfg,
        fold=0,
    )

    print(f"Training completed. Best Val AUC: {best_auc}")
    assert isinstance(best_auc, float), "AUC should be a float"
    # Note: AUC might be 0.5 or similar if the debug subset is too small/unbalanced,
    # but the function must return valid types.

    # 6. Inference
    print("\n--- Step 4: Inference ---")
    test_loader = get_test_loader(cfg, load_cached_data=False)

    # Use the trained model for inference (Ensemble of 1)
    ids, probs = predict_ensemble(
        models=[trained_model], dataloader=test_loader, device=device
    )

    print(f"Inference completed. Generated predictions for {len(ids)} images.")
    assert len(ids) == len(probs), "Mismatch between IDs and predictions count"
    assert probs.shape[1] == 2, "Predictions should be binary logits (Rust, Scab)"

    # 7. Submission Generation
    print("\n--- Step 5: Submission Generation ---")
    generate_submission(ids, probs, cfg.submission_path)

    # Verification: Check output file
    assert os.path.exists(cfg.submission_path), "Submission file was not created"

    submission_df = pd.read_csv(cfg.submission_path)
    expected_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]

    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {submission_df.columns}"
    assert len(submission_df) == len(ids), "Submission row count mismatch"

    # Verify probabilities sum to ~1 (floating point tolerance)
    # logic: healthy + multiple + rust + scab should be 1.0
    row_sums = submission_df[["healthy", "multiple_diseases", "rust", "scab"]].sum(
        axis=1
    )
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("Submission Verified.")
    print("\nPipeline Demo Completed Successfully.")
