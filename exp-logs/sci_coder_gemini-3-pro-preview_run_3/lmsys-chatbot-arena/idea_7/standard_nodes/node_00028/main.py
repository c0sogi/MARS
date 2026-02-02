import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import set_seed, get_logger
from library.data import get_dataloaders, get_test_loader
from library.model import SiameseDebertaMultiLayer
from library.engine import run_training, eval_fn, inference_fn


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude (Log Loss) and scalar features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_scalars = []
    all_preds = []
    all_targets = []

    # Collect predictions, targets, and scalar features
    with torch.no_grad():
        for batch in val_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            response_mask_a = batch["response_mask_a"].to(device)

            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            response_mask_b = batch["response_mask_b"].to(device)

            scalars = batch["scalars"].to(device)
            targets = batch["target"].to(device)

            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                outputs = model(
                    input_ids_a,
                    attention_mask_a,
                    response_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    response_mask_b,
                    scalars,
                )

            preds = torch.softmax(outputs.float(), dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())
            all_scalars.append(scalars.cpu().numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    scalars = np.concatenate(all_scalars)

    # Calculate individual Log Loss (Error Magnitude)
    # Clip predictions to avoid log(0)
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    # Compute cross entropy per sample: -sum(target * log(pred))
    sample_losses = -np.sum(targets * np.log(predictions), axis=1)

    # Scalars: [log_prompt_len, log_resp_a_len, log_resp_b_len]
    # Calculate correlations
    feature_names = ["Log Prompt Len", "Log Resp A Len", "Log Resp B Len"]
    print("\nCorrelation between Error Magnitude (Log Loss) and Input Features:")

    for i, name in enumerate(feature_names):
        feature_values = scalars[:, i]
        # Check for constant values to avoid warning
        if np.std(feature_values) == 0 or np.std(sample_losses) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_values, sample_losses)[0, 1]
        print(f"{name}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(Config.seed)
    logger = get_logger()

    # Adjust Config for fast baseline execution
    Config.epochs = 2  # Limit epochs to ensure completion within 2 hours
    print(
        f"Configuration: Device={Config.device}, Epochs={Config.epochs}, Batch Size={Config.train_batch_size}"
    )

    # 2. Data Loading
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    print("Loading DataLoaders...")
    train_loader, val_loader = get_dataloaders(tokenizer, load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = SiameseDebertaMultiLayer()
    model.to(Config.device)

    # 4. Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training
    print("Starting Training...")
    model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.device,
        epochs=Config.epochs,
        patience=2,  # Early stopping patience
        save_path=os.path.join(Config.working_dir, "best_model.pth"),
    )

    # 6. Validation Assessment
    print("Performing Final Validation...")
    val_metrics = eval_fn(model, val_loader, Config.device)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {val_metrics['log_loss']}")

    # 7. Failure Analysis
    failure_analysis(model, val_loader, Config.device)

    # 8. Submission Logic
    threshold = 1.0061561136439758
    if val_metrics["log_loss"] < threshold:
        print(
            f"\nValidation Log Loss ({val_metrics['log_loss']}) is better than threshold ({threshold}). Generating submission..."
        )

        test_loader = get_test_loader(tokenizer)
        inference_fn(model, test_loader, Config.device)

    else:
        print(
            f"\nValidation Log Loss ({val_metrics['log_loss']}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
