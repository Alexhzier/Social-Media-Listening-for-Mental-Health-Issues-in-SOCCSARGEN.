"""
Load training data from a Google Sheet (CSV export) or keep using data.py.

Supports sheets with:
  - "Mental healt indication" / "Mental health indication" / "Mental Health Indicators" (or "category" / "label") -> category
  - "Text Content" / "Texts" (or "text") -> text used for training
  - "Location" is read but not used for training (optional).

Category values: depression, stress, anxiety, addiction, suicidal_behavior.

Share the sheet with "Anyone with the link can view" so the CSV export works without login.
"""

import csv
import io
from typing import List, Tuple
from urllib.request import urlopen, Request

from categories import category_to_index


def _normalize_category(raw: str) -> str:
    """Map sheet value to category key (e.g. 'Depression' -> 'depression')."""
    key = raw.lower().strip().replace(" ", "_").replace("-", "_")
    return key


def _find_column(fieldnames: List[str], *candidates: str) -> str:
    """Return first header that contains (case-insensitive) any of the candidate substrings."""
    names = [n.strip() for n in fieldnames]
    for name in names:
        lower = name.lower()
        for c in candidates:
            if c in lower:
                return name
    return None


def load_from_google_sheet(url: str, max_rows: int = None) -> Tuple[List[str], List[int]]:
    """
    Load (texts, labels) from a Google Sheet CSV export URL.

    Accepts sheets with columns like:
      - "Mental healt indication" or "Mental health indication" or "category" / "label" -> category
      - "Text Content" or "text" -> text
    Rows with empty category or empty/short text are skipped.

    Args:
        url: Full CSV export URL, or use build_google_sheet_url(sheet_id, gid=0).

    Returns:
        (list of text strings, list of integer labels 0-4).
    """
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ThesisDataLoader/1.0)"})
    with urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames:
        raise ValueError("Empty or invalid CSV from Google Sheet")
    fields = list(reader.fieldnames)

    label_col = _find_column(fields, "indication", "indicator", "category", "label")
    text_col = _find_column(fields, "text content", "text", "content", "sentence")
    if not text_col:
        text_col = _find_column(fields, "content")
    if label_col is None:
        raise ValueError(
            f"Sheet must have a column for category (e.g. 'Mental healt indication', 'category'). Found: {fields}"
        )
    if text_col is None:
        raise ValueError(
            f"Sheet must have a column for text (e.g. 'Text Content', 'text'). Found: {fields}"
        )

    skip_text_values = {"duplicate", "gensan", "glan", ""}
    min_text_len = 3
    texts = []
    labels = []
    row_count = 0
    for row in reader:
        if max_rows is not None and row_count >= max_rows:
            break
        t = (row.get(text_col) or "").strip()
        c = (row.get(label_col) or "").strip()
        if not c or not t or len(t) < min_text_len:
            continue
        if t.lower().strip() in skip_text_values:
            continue
        key = _normalize_category(c)
        try:
            idx = category_to_index(key)
        except ValueError:
            continue
        texts.append(t)
        labels.append(idx)
        row_count += 1

    if not texts:
        raise ValueError("No valid rows (non-empty category + text) in the Google Sheet.")
    return texts, labels


def build_google_sheet_url(sheet_id: str, gid: int = 0) -> str:
    """
    Build the CSV export URL for a Google Sheet.

    Args:
        sheet_id: The ID from the sheet URL. From
            https://docs.google.com/spreadsheets/d/ABC123xyz/edit
            the sheet_id is ABC123xyz.
        gid: Sheet (tab) index. First tab = 0.

    Returns:
        URL that returns CSV when the sheet is shared with "Anyone with the link can view".
    """
    return f"https://docs.google.com/spreadsheets/d/{sheet_id.strip()}/export?format=csv&gid={gid}"

