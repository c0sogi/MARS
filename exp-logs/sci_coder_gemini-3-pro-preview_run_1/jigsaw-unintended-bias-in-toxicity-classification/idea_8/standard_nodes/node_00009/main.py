import os
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, get_device
from library.data import get_dataloaders, load_and_process_data
from library.model import ToxicityModel
from library.loss import CompositeLoss, AWP
from library.metrics import JigsawEvaluator
from library.engine import train_fn, eval_fn, inference_fn


def main():
    # 1. Configuration
    # We enable debug to allow subsetting, but set a large enough subset for meaningful training
    config = Config(debug=True)

    # Adjust config for fast baseline execution while maintaining performance potential
    config.train_subset_size = 100000  # 100k samples for speed (approx 30 mins on A100)
    config.epochs = 1  # 1 epoch to fit in time limit
    config.awp_start_epoch = 0  # Start AWP immediately since we only have 1 epoch
    config.train_batch_size = 16  # Safe batch size for DeBERTa-Large on 40GB GPU

    # Setup environment
    seed_everything(config.seed)
    device = get_device()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    logger = get_logger(os.path.join(config.output_dir, "run.log"))
    logger.info("Starting Runfile Execution...")
    logger.info(
        f"Config: Epochs={config.epochs}, Subset={config.train_subset_size}, BS={config.train_batch_size}"
    )

    # 2. Data Loading
    logger.info("Loading Data...")
    # load_cached_data=True allows using pre-processed parquet files if available
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # Load raw validation dataframe for metrics and failure analysis
    val_df = load_and_process_data(config, mode="val", load_cached_data=True)

    # 3. Model Initialization
    logger.info(f"Initializing Model: {config.model_name}")
    model = ToxicityModel(
        model_name=config.model_name,
        num_identity_classes=len(config.aux_identity_cols),
        num_aux_attack_classes=1 if config.aux_attack_col else 0,
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # 5. Loss & AWP
    criterion = CompositeLoss(config)
    awp = AWP(model, optimizer, adv_lr=config.awp_lr, adv_eps=config.awp_eps)

    # 6. Evaluator
    evaluator = JigsawEvaluator(config)

    # 7. Training Loop
    best_score = -float("inf")
    final_val_preds = None

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch+1}/{config.epochs}")

        # Train
        train_fn(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            awp,
            device,
            config,
            epoch,
            logger,
        )

        # Validate
        val_preds = eval_fn(model, val_loader, device)
        final_val_preds = val_preds

        # Evaluate
        score, metrics_dict = evaluator.evaluate(val_df, val_preds)
        logger.info(f"Validation Score: {score}")

        if score > best_score:
            best_score = score
            # We don't strictly need to save the checkpoint for this script unless we want to reload it,
            # but keeping it in memory is fine for the single run flow.
            # However, to be safe against OOM or crashes, we can save it.
            # torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.bin"))

    # Print Required Metric
    print(f"Final Validation Metric: {best_score}")

    # 8. Failure Analysis
    logger.info("Performing Failure Analysis...")
    if final_val_preds is not None:
        # Calculate error
        val_df["prediction"] = final_val_preds
        val_df["error"] = (val_df["target"] - val_df["prediction"]).abs()

        # Calculate correlations
        cols_to_check = config.aux_identity_cols + ["target"]
        # Ensure columns exist and fill NaNs for correlation calculation
        analysis_df = val_df[cols_to_check + ["error"]].fillna(0.0)

        print("-" * 30)
        print("Correlation between Error Magnitude and Features:")
        for col in cols_to_check:
            if col in analysis_df.columns:
                corr = analysis_df[col].corr(analysis_df["error"])
                print(f"  {col}: {corr:.4f}")
        print("-" * 30)

    # 9. Conditional Submission
    submission_threshold = 0.9268315106992828

    if best_score > submission_threshold:
        logger.info(
            f"Score {best_score} > {submission_threshold}. Generating submission..."
        )

        # Run Inference
        ids, preds = inference_fn(model, test_loader, device)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": ids, "prediction": preds})

        # Save
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.info(
            f"Score {best_score} <= {submission_threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
