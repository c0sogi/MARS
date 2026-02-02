from library.config import seed_everything


def jaccard(str1, str2):
    """
    Computes the word-level Jaccard similarity score between two strings.

    Args:
        str1: The first string (e.g., predicted text).
        str2: The second string (e.g., ground truth text).

    Returns:
        float: The Jaccard similarity score between 0.0 and 1.0.
    """
    # Convert to set of lowercased words
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())

    # Calculate intersection and union
    c = a.intersection(b)
    union_len = len(a) + len(b) - len(c)

    # Handle edge case where both strings are empty or contain no words
    if union_len == 0:
        return 0.0

    return float(len(c)) / union_len
