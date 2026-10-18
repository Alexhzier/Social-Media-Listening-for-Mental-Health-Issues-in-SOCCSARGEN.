"""
Training data for the multi-class distress-pattern classifier.

Designed for Cebuano, Tagalog, and English. Each sample is (text, category_key).
Category keys: depression, stress, anxiety, addiction, suicidal_behavior.

This file is intentionally blank so training data can be loaded from Google Sheets.
"""

from categories import CATEGORY_KEYS

# (text, category_key). Category key must be one of CATEGORY_KEYS.
# Leave empty when training with --google_sheet links.
SAMPLES = []

