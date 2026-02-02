import re
from library.config import set_seed


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard similarity score between two strings.

    Args:
        str1 (str): The ground truth string.
        str2 (str): The predicted string.

    Returns:
        float: The Jaccard score (intersection over union).
    """
    # Ensure inputs are strings
    if not isinstance(str1, str):
        str1 = str(str1) if str1 is not None else ""
    if not isinstance(str2, str):
        str2 = str(str2) if str2 is not None else ""

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    denominator = len(a) + len(b) - len(c)

    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def split_sentences(text):
    """
    Splits text into sentences using regex-based logic for Hindi and Tamil.
    Handles delimiters like '.', '?', '!', '|', and the Devanagari Danda ('।').

    Args:
        text (str): The input text block.

    Returns:
        list: A list of sentence strings.
    """
    if not isinstance(text, str):
        return []

    # Regex pattern to match sentence delimiters.
    # We include:
    # . (period)
    # ? (question mark)
    # ! (exclamation mark)
    # | (pipe, often used as danda in noisy data)
    # \u0964 (Devanagari Danda)
    # The pipe | is treated as a literal character inside the [] set.
    pattern = r"([.?|!|\u0964]+)"

    # Split the text, keeping the delimiters (due to capturing group)
    parts = re.split(pattern, text)

    sentences = []

    # re.split with capturing group returns [text, delim, text, delim, ...]
    # We iterate in steps of 2 to combine text with its following delimiter.
    for i in range(0, len(parts) - 1, 2):
        content = parts[i].strip()
        delimiter = parts[i + 1]

        # Append combined sentence if there is content or just a delimiter
        if content or delimiter:
            sentences.append(content + delimiter)

    # Handle any remaining text that didn't end with a delimiter
    if len(parts) % 2 != 0:
        last_part = parts[-1].strip()
        if last_part:
            sentences.append(last_part)

    return sentences
