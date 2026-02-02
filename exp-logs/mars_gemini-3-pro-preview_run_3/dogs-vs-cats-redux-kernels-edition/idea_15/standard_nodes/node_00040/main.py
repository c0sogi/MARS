import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from library import config, utils, data_loader, model_factory, engine, inference


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = config.DEVICE

    # Prepare storage for ensemble predictions
    # Validation: List of arrays (one per model)
    # Test: List of dicts (one per model)
    ensemble_val_probs = []
    ensemble_test_preds = []

    # Load validation ground truth for metric calculation
    val_df = pd.read_csv(config.VAL_CSV)
    y_true = val_df["label"].values

    # 2. Iterate through models in the ensemble
    # Order: resnet, convnext, maxvit
    model_keys = ["resnet", "convnext", "maxvit"]

    for key in model_keys:
        specs = config.MODEL_SPECS[key]

        # A. Data Loading
        train_loader, val_loader, test_loader = data_loader.get_dataloaders(
            key, load_cached_data=True
        )

        # B. Model Initialization
        model = model_factory.create_model(key, pretrained=True)
        model = model.to(device)

        # C. Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=specs["learning_rate"],
            weight_decay=specs["weight_decay"],
        )

        # Cosine Annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=specs["epochs"], eta_min=specs["scheduler_min_lr"]
        )

        # D. Training
        checkpoint_name = f"{key}_best.pth"
        model = engine.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=specs["epochs"],
            scheduler=scheduler,
            checkpoint_name=checkpoint_name,
        )

        # E. Validation Inference (with TTA for consistency)
        model.eval()
        val_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)

                # Original
                out = model(images)
                prob = torch.sigmoid(out)

                # Flip (TTA)
                images_flip = torch.flip(images, dims=[3])
                out_flip = model(images_flip)
                prob_flip = torch.sigmoid(out_flip)

                # Average
                avg_prob = (prob + prob_flip) / 2.0
                val_probs.extend(avg_prob.cpu().numpy().flatten())

        ensemble_val_probs.append(np.array(val_probs))

        # F. Test Inference
        test_preds = inference.predict_with_tta(model, test_loader, device)
        ensemble_test_preds.append(test_preds)

        # Cleanup to save memory
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    # 3. Ensemble Aggregation (Validation)
    # Average probabilities across models
    # Stack arrays: shape (num_models, num_samples)
    stacked_val_probs = np.vstack(ensemble_val_probs)
    avg_val_probs = np.mean(stacked_val_probs, axis=0)

    # Calculate Metric
    final_metric = utils.calculate_log_loss(y_true, avg_val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Add predictions and error to dataframe
    val_df["pred_prob"] = avg_val_probs
    val_df["error"] = (val_df["label"] - val_df["pred_prob"]).abs()

    # Extract features for correlation
    # We need to read image files to get width/height/size
    widths = []
    heights = []
    file_sizes = []

    for idx, row in val_df.iterrows():
        full_path = os.path.join(config.INPUT_DIR, row["filepath"])
        try:
            # File size
            file_sizes.append(os.path.getsize(full_path))

            # Dimensions (Lazy load)
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    val_df["width"] = widths
    val_df["height"] = heights
    val_df["file_size"] = file_sizes

    # Calculate correlations
    features = ["width", "height", "file_size"]
    for feat in features:
        if val_df[feat].std() > 0:  # Avoid constant columns
            corr = val_df["error"].corr(val_df[feat])
            print(f"Correlation Error vs {feat}: {corr}")
        else:
            print(f"Correlation Error vs {feat}: NaN (Constant feature)")

    # 5. Submission
    threshold = 0.009241249605204765
    if final_metric < threshold:
        final_test_preds = inference.average_ensemble_predictions(ensemble_test_preds)
        inference.generate_submission(final_test_preds)


if __name__ == "__main__":
    main()
