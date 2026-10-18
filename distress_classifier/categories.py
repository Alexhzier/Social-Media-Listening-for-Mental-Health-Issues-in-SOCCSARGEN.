"""
Category labels for the multi-class distress-pattern classifier.

These categories indicate types of mental health-related LINGUISTIC PATTERNS only.
They do NOT represent clinical diagnoses. For research and thesis use only.
"""

# Order defines the model's output indices (0 = Depression, 1 = Stress, ...).
# Only 5 categories for focused classification.
CATEGORIES = [
    "Depression",
    "Stress",
    "Anxiety",
    "Addiction",
    "Suicidal Behavior",
]

# Normalized keys for use in data.py (text, key) -> mapped to index via this list.
CATEGORY_KEYS = [
    "depression",
    "stress",
    "anxiety",
    "addiction",
    "suicidal_behavior",
]


def category_to_index(category_key: str) -> int:
    """Convert category key (e.g. 'depression') to model index (0-4)."""
    key = category_key.lower().strip().replace(" ", "_").replace("-", "_")
    try:
        return CATEGORY_KEYS.index(key)
    except ValueError:
        raise ValueError(f"Unknown category: {category_key}. Use one of: {CATEGORY_KEYS}")


def index_to_category(index: int) -> str:
    """Convert model index to display name (e.g. 1 -> 'Depression')."""
    return CATEGORIES[index]

