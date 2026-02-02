import re
import os
import pandas as pd
import numpy as np


def clean_html_text(text: str) -> str:
    """
    Cleans the input text by stripping HTML tags and removing non-alphanumeric characters.

    Args:
        text (str): The input string containing HTML and text.

    Returns:
        str: The cleaned, lowercased text with only alphanumeric characters and spaces.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags (replace with space to prevent word merging)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove non-alphanumeric characters (keep digits and english letters)
    # We keep spaces (\s) to preserve word boundaries.
    # Everything else (punctuation, symbols like +, #, etc.) is removed
    # as per the specific requirement to "remove non-alphanumeric characters".
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Normalize whitespace (replace multiple spaces with single space) and strip
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    return text.lower()


def save_submission(ids, tags, filename: str):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or np.array): List of question IDs.
        tags (list or np.array): List of predicted tag strings (space-delimited).
        filename (str): The path to save the submission file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({"Id": ids, "Tags": tags})

    # Save to CSV
    # index=False avoids writing the row numbers
    # quoting=1 (csv.QUOTE_ALL) isn't strictly necessary unless delimiters appear in data,
    # but pandas defaults are usually sufficient for standard CSV parsers.
    df.to_csv(filename, index=False)
