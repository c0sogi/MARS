import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.data_loader import load_raw_data, AuthorDataset
from library.neural_models import CustomTransformer
from library.statistical_models import StatisticalModel
from library.pretraining import run_dapt_pipeline
from library.distillation_engine import DistillationEngine
from library.ensemble_optimizer import EnsembleOptimizer


def main():
    # 1. Configuration Overrides for Speed & Efficiency
    # We reduce epochs to ensure the run completes quickly as a baseline.
    # We increase batch size to utilize the A100 GPU.
    Config.FT_EPOCHS = 3
    Config.MLM_EPOCHS = 3
    Config.TRAIN_BATCH_SIZE = 32
    Config.setup()
    seed_everything()

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading data...")
    train_df, val_df, test_df = load_raw_data()

    # Prepare Validation Labels for optimization and scoring
    y_val = val_df["author"].map(Config.LABEL2ID).values

    # 3. Phase 0: Domain Adaptive Pretraining (DAPT)
    print("\n=== Phase 0: Domain Adaptive Pretraining ===")
    # Returns a dict mapping original model names to paths of fine-tuned MLM models
    adapted_model_paths = run_dapt_pipeline(load_cached_data=True)

    # 4. Phase 1: Teacher Training & Ensemble Optimization
    print("\n=== Phase 1: Teacher Training & Ensemble Optimization ===")

    # Dictionaries to store predictions
    val_preds_dict = {}
    test_preds_dict = {}

    # --- 4a. Statistical Model (Static Anchor) ---
    print("Training Statistical Model...")
    stat_model = StatisticalModel()
    # Fits LR and NB, optimizes their internal blend on Val
    stat_model.fit(load_cached_data=True)

    val_preds_dict["statistical"] = stat_model.predict_proba(
        "val", load_cached_data=True
    )
    test_preds_dict["statistical"] = stat_model.predict_proba(
        "test", load_cached_data=True
    )

    # --- 4b. Neural Models (Supervised Teachers) ---
    for model_name in Config.MODEL_BACKBONES:
        print(f"Training Teacher: {model_name}")

        # Use DAPT weights if available, otherwise fallback to base model
        model_path = adapted_model_paths.get(model_name, model_name)

        # Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create Datasets
        train_ds = AuthorDataset(train_df, tokenizer)
        val_ds = AuthorDataset(val_df, tokenizer)
        test_ds = AuthorDataset(test_df, tokenizer, is_test=True)

        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Initialize Model and Engine
        model = CustomTransformer(model_path).to(device)
        engine = DistillationEngine(model, device, tokenizer)

        # Train Supervised
        engine.train_supervised(train_loader, val_loader, epochs=Config.FT_EPOCHS)

        # Generate Predictions
        _, val_probs = engine.evaluate(val_loader)
        _, test_probs = engine.evaluate(test_loader)

        val_preds_dict[model_name] = val_probs
        test_preds_dict[model_name] = test_probs

        # Cleanup to free GPU memory
        del model, engine, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    # 5. Optimize Ensemble Weights
    print("Optimizing Ensemble Weights...")
    optimizer = EnsembleOptimizer()
    best_weights = optimizer.optimize_weights(val_preds_dict, y_val)

    # 6. Phase 2: Generate Soft Targets
    print("\n=== Phase 2: Generating Soft Targets ===")
    # Blend Phase 1 Test predictions to create soft targets for distillation
    soft_targets = optimizer.blend_predictions(test_preds_dict, best_weights)

    # 7. Phase 3: Student Distillation
    print("\n=== Phase 3: Student Distillation ===")

    distilled_val_preds = {}
    distilled_test_preds = {}

    # The Statistical model is NOT retrained. It acts as a Static Anchor.
    # We carry over its Phase 1 predictions.
    distilled_val_preds["statistical"] = val_preds_dict["statistical"]
    distilled_test_preds["statistical"] = test_preds_dict["statistical"]

    # Prepare raw text lists for the DistillationDataset
    train_texts = train_df["text"].tolist()
    train_labels = train_df["author"].map(Config.LABEL2ID).values
    test_texts = test_df["text"].tolist()

    for model_name in Config.MODEL_BACKBONES:
        print(f"Distilling Student: {model_name}")

        # Re-initialize from DAPT weights (Fresh Student)
        model_path = adapted_model_paths.get(model_name, model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Validation Loader (for monitoring)
        val_ds = AuthorDataset(val_df, tokenizer)
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Test Loader (for final inference)
        test_ds = AuthorDataset(test_df, tokenizer, is_test=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=Config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        model = CustomTransformer(model_path).to(device)
        engine = DistillationEngine(model, device, tokenizer)

        # Train Distilled (Train Hard Labels + Test Soft Targets)
        engine.train_distilled(
            train_texts,
            train_labels,
            test_texts,
            soft_targets,
            val_loader,
            epochs=Config.FT_EPOCHS,
        )

        # Generate Final Predictions
        _, val_probs = engine.evaluate(val_loader)
        _, test_probs = engine.evaluate(test_loader)

        distilled_val_preds[model_name] = val_probs
        distilled_test_preds[model_name] = test_probs

        del model, engine, val_loader, test_loader
        torch.cuda.empty_cache()

    # 8. Final Evaluation
    print("\n=== Final Evaluation ===")

    # Blend Distilled Students + Static Anchor using Phase 1 weights
    final_val_probs = optimizer.blend_predictions(distilled_val_preds, best_weights)
    final_metric = compute_log_loss(y_val, final_val_probs)

    print(f"Final Validation Metric: {final_metric}")

    # 9. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Cross Entropy per sample to identify hard samples
    rows = np.arange(len(y_val))
    prob_true = final_val_probs[rows, y_val]
    prob_true = np.clip(prob_true, 1e-15, 1.0)
    sample_losses = -np.log(prob_true)

    # Correlate error with text length
    val_lengths = val_df["text"].str.len().values
    correlation = np.corrcoef(sample_losses, val_lengths)[0, 1]
    print(f"Correlation between Error and Text Length: {correlation}")

    # 10. Submission
    # Threshold defined in task description
    threshold = 0.2435629959371868

    if final_metric < threshold:
        print("Metric below threshold. Generating submission...")
        final_test_probs = optimizer.blend_predictions(
            distilled_test_preds, best_weights
        )

        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                "EAP": final_test_probs[:, 0],
                "HPL": final_test_probs[:, 1],
                "MWS": final_test_probs[:, 2],
            }
        )

        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"Metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
