import sys
import os
import torch
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.utils import set_seed, get_device, load_checkpoint
from library.data_loader import get_data, TaggerDataset
from library.trainer import train_tagger, train_seq2seq
from library.models_tagger import BiLSTM_CRF
from library.models_seq2seq import CharTransformer
from library.inference import NormalizationPipeline


def run_validation():
    """
    Runs the full inference pipeline on the validation set to compute the metric
    and collect statistics for failure analysis.
    """
    device = get_device()
    print("\n" + "=" * 40)
    print("Running Validation on Full Pipeline")
    print("=" * 40)

    # 1. Load Data & Vocabs
    # Since training has run, these should be cached and consistent with the trained models
    (
        vocab_tokens,
        vocab_chars,
        vocab_classes,
        _,  # train_grouped
        val_grouped,  # Validation data
        _,  # test_grouped
        _,  # seq2seq_train_df
    ) = get_data(load_cached=True)

    # 2. Load Knowledge Base
    kb_path = Config.KNOWLEDGE_BASE_PATH
    kb = {}
    if os.path.exists(kb_path):
        print(f"Loading Knowledge Base from {kb_path}...")
        kb_df = pd.read_parquet(kb_path)
        kb = {
            (row["before"], row["class"]): row["after"] for _, row in kb_df.iterrows()
        }

    # 3. Load Models
    print("Loading models for validation...")
    tagger = BiLSTM_CRF(
        vocab_size=len(vocab_tokens),
        char_vocab_size=len(vocab_chars),
        num_classes=len(vocab_classes),
    ).to(device)

    try:
        load_checkpoint(Config.TAGGER_MODEL_PATH, tagger, device=device)
    except FileNotFoundError:
        print("Error: Tagger model checkpoint not found.")
        return 0.0, None, None

    seq2seq = CharTransformer(
        num_chars=len(vocab_chars), num_classes=len(vocab_classes)
    ).to(device)

    try:
        load_checkpoint(Config.SEQ2SEQ_MODEL_PATH, seq2seq, device=device)
    except FileNotFoundError:
        print("Error: Seq2Seq model checkpoint not found.")
        return 0.0, None, None

    tagger.eval()
    seq2seq.eval()

    # 4. Prepare Validation Loader
    val_ds = TaggerDataset(
        val_grouped, vocab_tokens, vocab_chars, vocab_classes, is_test=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 5. Inference Loop
    correct_count = 0
    total_count = 0

    # Stats for analysis
    errors = []
    stats_data = []  # List of (length, is_error)

    sos_idx = vocab_chars.stoi[Config.SOS_TOKEN]
    eos_idx = vocab_chars.stoi[Config.EOS_TOKEN]
    pad_idx = vocab_chars.stoi[Config.PAD_TOKEN]
    unk_idx = vocab_chars.stoi[Config.UNK_TOKEN]

    df_idx = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            token_ids = batch["token_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            mask = batch["mask"].to(device)

            # --- Step 1: Tagger Prediction ---
            pred_tags = tagger.decode(token_ids, char_ids, mask)

            # Retrieve raw data chunk corresponding to this batch
            current_batch_size = token_ids.size(0)
            df_batch = val_grouped.iloc[df_idx : df_idx + current_batch_size]
            df_idx += current_batch_size

            oov_queue = []
            batch_predictions = []

            # --- Step 2: Logic & Knowledge Base ---
            for b_i in range(current_batch_size):
                row = df_batch.iloc[b_i]
                raw_tokens = row["before"]

                seq_len = len(raw_tokens)
                sent_preds = []

                for t_i in range(seq_len):
                    token_str = raw_tokens[t_i]
                    class_idx = pred_tags[b_i, t_i].item()
                    class_str = vocab_classes.lookup_token(class_idx)

                    norm_text = token_str

                    kb_key = (token_str, class_str)

                    if kb_key in kb:
                        norm_text = kb[kb_key]
                    elif class_str == "PLAIN" or class_str == "PUNCT":
                        norm_text = token_str
                    else:
                        # OOV -> Queue for Seq2Seq
                        oov_queue.append(
                            {
                                "batch_idx": b_i,
                                "token_idx": t_i,
                                "token_str": token_str,
                                "class_idx": class_idx,
                            }
                        )
                        norm_text = "<PENDING>"

                    sent_preds.append(norm_text)
                batch_predictions.append(sent_preds)

            # --- Step 3: Seq2Seq for OOVs ---
            if oov_queue:
                src_id_list = []
                class_id_list = []

                for item in oov_queue:
                    token_str = item["token_str"]
                    c_ids = [vocab_chars.stoi.get(c, unk_idx) for c in token_str]
                    # Pad/Truncate
                    if len(c_ids) > Config.MAX_CHAR_LEN:
                        c_ids = c_ids[: Config.MAX_CHAR_LEN]
                    else:
                        c_ids += [pad_idx] * (Config.MAX_CHAR_LEN - len(c_ids))
                    src_id_list.append(c_ids)
                    class_id_list.append(item["class_idx"])

                src_tensor = torch.tensor(src_id_list, dtype=torch.long).to(device)
                class_tensor = torch.tensor(class_id_list, dtype=torch.long).to(device)

                generated_ids = seq2seq.predict(
                    src_tensor, class_tensor, sos_idx, eos_idx
                )

                for i, row_ids in enumerate(generated_ids):
                    chars = []
                    for idx in row_ids:
                        idx = idx.item()
                        if idx == eos_idx:
                            break
                        if idx == pad_idx:
                            continue
                        chars.append(vocab_chars.lookup_token(idx))
                    pred_str = "".join(chars)

                    # Update placeholder
                    q_item = oov_queue[i]
                    batch_predictions[q_item["batch_idx"]][
                        q_item["token_idx"]
                    ] = pred_str

            # --- Step 4: Compare and Collect Stats ---
            for b_i in range(current_batch_size):
                row = df_batch.iloc[b_i]
                true_afters = row["after"]
                raw_tokens = row["before"]
                raw_classes = row["class"]
                preds = batch_predictions[b_i]

                for t_i in range(len(raw_tokens)):
                    p = preds[t_i]
                    a = true_afters[t_i]
                    is_error = 0

                    if p != a:
                        is_error = 1
                        errors.append(
                            {
                                "token": raw_tokens[t_i],
                                "class": raw_classes[t_i],
                                "predicted": p,
                                "actual": a,
                                "len": len(raw_tokens[t_i]),
                            }
                        )

                    stats_data.append((len(raw_tokens[t_i]), is_error))

                    if is_error == 0:
                        correct_count += 1
                    total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Print Metric exactly as requested
    print(f"Final Validation Metric: {accuracy}")

    return accuracy, pd.DataFrame(errors), stats_data


def perform_failure_analysis(error_df, stats_data):
    print("\n" + "=" * 40)
    print("Failure Analysis")
    print("=" * 40)

    if error_df is None or len(error_df) == 0:
        print("No errors found to analyze.")
        return

    print(f"Total Errors: {len(error_df)}")

    # 1. Error Distribution by Class
    print("\n--- Error Distribution by Class (Top 5) ---")
    print(error_df["class"].value_counts().head(5).to_string())

    # 2. Correlation with Length
    print("\n--- Correlation: Error Magnitude vs Input Feature (Length) ---")
    if len(stats_data) > 0:
        lengths = np.array([x[0] for x in stats_data])
        is_errs = np.array([x[1] for x in stats_data])

        if np.std(is_errs) > 0 and np.std(lengths) > 0:
            corr = np.corrcoef(lengths, is_errs)[0, 1]
            print(
                f"Correlation (Point-Biserial) between Token Length and Error: {corr:.4f}"
            )
            if corr > 0.1:
                print(
                    "Observation: Longer tokens are positively correlated with higher error rates."
                )
            elif corr < -0.1:
                print(
                    "Observation: Shorter tokens are positively correlated with higher error rates."
                )
            else:
                print(
                    "Observation: Weak or no linear correlation between length and error."
                )
        else:
            print("Correlation undefined (zero variance in errors or lengths).")
    else:
        print("Insufficient data for correlation analysis.")


def main():
    # 1. Configuration Overrides for Fast Baseline
    print("Configuring Fast Baseline Run...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 100000  # Use 100k sentences (approx 1.3M tokens)
    Config.NUM_EPOCHS = 3  # Reduced epochs for speed

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    # 2. Train Models
    # train_tagger will trigger data processing. load_cached=True is safe as it builds if missing.
    # We rely on Config.DEBUG=True to ensure we process the subset.
    print("\n>>> Stage 1: Training Tagger")
    train_tagger(load_cached=True)

    # Increase epochs slightly for Seq2Seq as it trains on a smaller subset (only changed tokens)
    Config.NUM_EPOCHS = 5
    print("\n>>> Stage 2: Training Seq2Seq Fallback")
    train_seq2seq(load_cached=True)

    # 3. Validation
    accuracy, error_df, stats_data = run_validation()

    # 4. Failure Analysis
    perform_failure_analysis(error_df, stats_data)

    # 5. Submission
    # Threshold defined in task
    THRESHOLD = 0.9949142925818993

    if accuracy > THRESHOLD:
        print(
            f"\nValidation Metric ({accuracy}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        pipeline = NormalizationPipeline(load_cached=True)
        pipeline.predict()
    else:
        print(
            f"\nValidation Metric ({accuracy}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
