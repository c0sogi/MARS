import pandas as pd
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import CrossEncoderModel, train_model, generate_submission
from library.utils import compute_pearson_correlation


def main():
    # 1. Setup Environment
    seed_everything(Config.seed)
    device = Config.device

    # 2. Prepare Data
    # Initialize tokenizer for the backbone model
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load dataloaders with caching enabled
    # We use the full dataset (debug=False) as it is small enough (~30k samples)
    # to train quickly (minutes) on the provided hardware, ensuring best performance.
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer, load_cached_data=True, debug=False
    )

    # 3. Initialize Model
    model = CrossEncoderModel(
        model_name=Config.model_name,
        num_labels=Config.num_labels,
        dropout=Config.dropout,
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Run Training
    # train_model handles the training loop, validation, early stopping, and saving best model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.epochs,
        patience=Config.patience,
        save_path=Config.model_save_path,
    )

    # 6. Final Validation & Metric Calculation
    # We run a pass on validation set to get predictions for metric and failure analysis
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            logits = outputs["logits"].detach().cpu().numpy().flatten()
            label_ids = labels.detach().cpu().numpy().flatten()

            val_preds.extend(logits)
            val_targets.extend(label_ids)

    final_metric = compute_pearson_correlation(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    # Load validation metadata to access features
    df_val = pd.read_csv(Config.val_path)

    # Ensure lengths match (in case of dropped samples, though loaders should be consistent)
    min_len = min(len(df_val), len(val_preds))
    df_val = df_val.iloc[:min_len].copy()
    val_preds = val_preds[:min_len]
    val_targets = val_targets[:min_len]

    df_val["prediction"] = val_preds
    df_val["abs_error"] = (df_val["score"] - df_val["prediction"]).abs()

    # Compute simple text features
    df_val["anchor_len"] = df_val["anchor"].astype(str).apply(len)
    df_val["target_len"] = df_val["target"].astype(str).apply(len)

    # Calculate correlations with error
    print("Correlation between Model Error (Absolute) and Features:")
    for feature in ["anchor_len", "target_len", "score"]:
        if feature in df_val.columns:
            corr = df_val["abs_error"].corr(df_val[feature])
            print(f"  {feature}: {corr}")

    # 8. Conditional Submission
    threshold = 0.8288092510484422
    if final_metric > threshold:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_submission(
            model=model,
            test_loader=test_loader,
            device=device,
            submission_path=Config.submission_path,
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
