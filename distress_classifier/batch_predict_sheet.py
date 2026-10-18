"""
Batch inference from Google Sheets for the 5-class distress-pattern model.

Use this when you have text-only rows and want model predictions at scale.
The script supports:
  - One or more Google Sheet links/IDs
  - One or more tab GIDs per sheet
  - CSV export with predicted class + confidence + per-class probabilities
"""

import argparse
import csv
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

import numpy as np

from predict import DEFAULT_UNCERTAIN_THRESHOLD, load_model_and_tokenizer
from preprocess import clean_text, texts_to_sequences

TRASH_LABEL = "Trash"


TEXT_COLUMN_CANDIDATES = (
    "text content",
    "text",
    "post",
    "content",
    "sentence",
    "message",
    "body",
)

DATE_COLUMN_CANDIDATES = (
    "date",
    "timestamp",
    "time",
    "posted",
    "created",
)

LOCATION_COLUMN_CANDIDATES = (
    "location",
    "loc",
    "place",
    "city",
    "municipality",
    "province",
    "barangay",
    "school",
    "campus",
    "region",
)

PLATFORM_COLUMN_CANDIDATES = (
    "platform",
    "source",
    "app",
    "site",
    "channel",
    "social media",
)

DEFAULT_DUAL_CATEGORY_THRESHOLD = 0.40
DEFAULT_DUAL_MAX_GAP = 0.10
DEFAULT_DUAL_MIN_MARGIN_FROM_THIRD = 0.08
DEFAULT_SUBSET_THRESHOLD = 0.50
DEFAULT_MAIN_THRESHOLD = 0.70
DEFAULT_TRASH_TOP_MAX = 0.35
DEFAULT_TRASH_MAX_GAP = 0.12
DEFAULT_TRASH_THIRD_MIN = 0.15


def _sheet_id_from_url(url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else ""


def _gid_from_url(url: str) -> int:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "gid" in query and query["gid"]:
        try:
            return int(query["gid"][0])
        except ValueError:
            return 0
    if parsed.fragment.startswith("gid="):
        try:
            return int(parsed.fragment.split("=", 1)[1])
        except ValueError:
            return 0
    return 0


def build_google_sheet_csv_url(sheet_arg: str, gid: int = 0) -> str:
    """Accept full Google Sheet URL or bare sheet ID and return CSV export URL."""
    arg = sheet_arg.strip()
    if arg.startswith("http"):
        if "export?format=csv" in arg:
            # Force desired gid if provided by caller.
            parsed = urlparse(arg)
            q = parse_qs(parsed.query)
            q["format"] = ["csv"]
            q["gid"] = [str(gid)]
            sheet_id = _sheet_id_from_url(arg)
            if not sheet_id:
                return arg
            return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        sheet_id = _sheet_id_from_url(arg)
        if not sheet_id:
            raise ValueError(f"Could not parse sheet ID from URL: {arg}")
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return f"https://docs.google.com/spreadsheets/d/{arg}/export?format=csv&gid={gid}"


def _find_text_column(fieldnames: List[str], explicit: str = None) -> str:
    names = [n.strip() for n in fieldnames if n and n.strip()]
    if explicit:
        explicit_lower = explicit.strip().lower()
        for name in names:
            if name.lower() == explicit_lower:
                return name
        raise ValueError(f"Text column {explicit!r} not found. Available columns: {names}")

    for name in names:
        lowered = name.lower()
        for candidate in TEXT_COLUMN_CANDIDATES:
            if candidate in lowered:
                return name

    if len(names) == 1:
        return names[0]

    raise ValueError(
        "Could not auto-detect text column. "
        f"Pass --text_column explicitly. Available columns: {names}"
    )


def _find_optional_column(
    fieldnames: List[str],
    explicit: str = None,
    candidates: Tuple[str, ...] = (),
) -> str:
    names = [n.strip() for n in fieldnames if n and n.strip()]
    if explicit:
        explicit_lower = explicit.strip().lower()
        for name in names:
            if name.lower() == explicit_lower:
                return name
        raise ValueError(f"Column {explicit!r} not found. Available columns: {names}")

    for name in names:
        lowered = name.lower()
        for candidate in candidates:
            if candidate in lowered:
                return name
    return ""


def _looks_like_no_header_csv(first_row: List[str]) -> bool:
    """
    Heuristic: first row looks like data instead of header.
    Example seen in user sheets: [<actual post text>, DATE, LOCATION]
    """
    if not first_row:
        return False
    first = (first_row[0] or "").strip().lower()
    if not first:
        return False

    # If first cell is long / sentence-like and not a standard header label.
    standard_header_tokens = ("text", "content", "post", "message", "sentence", "body")
    is_standard = any(token in first for token in standard_header_tokens)
    sentence_like = len(first) >= 24 or len(first.split()) >= 4

    tail = [(c or "").strip().lower() for c in first_row[1:]]
    has_meta_tail = any(c in ("date", "location", "loc", "place") for c in tail)
    return (not is_standard) and sentence_like and has_meta_tail


def load_text_rows_from_sheet(
    csv_url: str,
    text_column: str = None,
    date_column: str = None,
    location_column: str = None,
    platform_column: str = None,
    max_rows: int = None,
) -> Tuple[List[Dict], str, str, str, str]:
    """
    Load text rows from a Google Sheet CSV URL.

    Returns:
      rows: [{"row_number": int, "text": str, "date": str, "location": str, "platform": str}, ...]
      actual_text_column: detected/selected text column name
      actual_date_column: detected/selected date column name or ""
      actual_location_column: detected/selected location column name or ""
      actual_platform_column: detected/selected platform column name or ""
    """
    req = Request(csv_url, headers={"User-Agent": "Mozilla/5.0 (compatible; ThesisBatchPredict/1.0)"})
    with urlopen(req, timeout=45) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    raw_rows = list(csv.reader(io.StringIO(raw)))
    if not raw_rows:
        raise ValueError("Empty CSV")

    # Handle sheets where first row is data (no header row).
    if text_column is None and _looks_like_no_header_csv(raw_rows[0]):
        rows = []
        for idx, row in enumerate(raw_rows, start=1):
            if max_rows is not None and len(rows) >= max_rows:
                break
            text = (row[0] if row else "").strip()
            rows.append(
                {
                    "row_number": idx,
                    "text": text,
                    "date": (row[1] if len(row) > 1 else "").strip(),
                    "location": (row[2] if len(row) > 2 else "").strip(),
                    "platform": (row[3] if len(row) > 3 else "").strip(),
                }
            )
        if not rows:
            raise ValueError("No non-empty text rows found in no-header CSV.")
        return rows, "__first_column__", "__second_column__", "__third_column__", "__fourth_column__"

    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames:
        raise ValueError("Empty or invalid CSV")

    text_col = _find_text_column(list(reader.fieldnames), explicit=text_column)
    date_col = _find_optional_column(
        list(reader.fieldnames),
        explicit=date_column,
        candidates=DATE_COLUMN_CANDIDATES,
    )
    location_col = _find_optional_column(
        list(reader.fieldnames),
        explicit=location_column,
        candidates=LOCATION_COLUMN_CANDIDATES,
    )
    platform_col = _find_optional_column(
        list(reader.fieldnames),
        explicit=platform_column,
        candidates=PLATFORM_COLUMN_CANDIDATES,
    )

    rows = []
    for idx, row in enumerate(reader, start=2):  # row 1 = header
        if max_rows is not None and len(rows) >= max_rows:
            break
        text = (row.get(text_col) or "").strip()
        rows.append(
            {
                "row_number": idx,
                "text": text,
                "date": (row.get(date_col) or "").strip() if date_col else "",
                "location": (row.get(location_col) or "").strip() if location_col else "",
                "platform": (row.get(platform_col) or "").strip() if platform_col else "",
            }
        )

    if not rows:
        raise ValueError("No non-empty text rows found in sheet.")
    return rows, text_col, date_col, location_col, platform_col


def build_google_sheet_gviz_csv_url(sheet_arg: str, sheet_name: str) -> str:
    """Build gviz CSV URL for a specific tab name (avoids needing gid)."""
    sheet_id = _sheet_id_from_url(sheet_arg)
    if not sheet_id and not sheet_arg.startswith("http"):
        sheet_id = sheet_arg.strip()
    if not sheet_id:
        raise ValueError(f"Could not parse sheet ID from: {sheet_arg}")
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}"


def _slugify_category(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "category"


def predict_batch(
    texts: List[str],
    model,
    tokenizer,
    maxlen: int,
    categories: List[str],
    batch_size: int,
    uncertain_threshold: float,
    dual_category_threshold: float,
    dual_max_gap: float,
    dual_min_margin_from_third: float,
    subset_threshold: float,
    main_threshold: float,
    trash_top_max: float,
    trash_max_gap: float,
    trash_third_min: float,
) -> List[Dict]:
    cleaned = [clean_text(t) for t in texts]
    X = texts_to_sequences(tokenizer, cleaned, maxlen=maxlen)
    probs = model.predict(X, batch_size=batch_size, verbose=0)

    results = []
    for row_probs in probs:
        sorted_indices = np.argsort(row_probs)[::-1]
        idx = int(sorted_indices[0])
        second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else None
        third_prob = float(row_probs[int(sorted_indices[2])]) if len(sorted_indices) > 2 else 0.0

        top_prob = float(row_probs[idx])
        top_class = categories[idx]
        second_prob = float(row_probs[second_idx]) if second_idx is not None else 0.0
        second_class = categories[second_idx] if second_idx is not None else ""

        # Dual-label rule: assign both classes only if top-2 are both strong and
        # close to each other (conflicting), and clearly above the 3rd class.
        top_second_gap = top_prob - second_prob
        second_third_gap = second_prob - third_prob
        is_dual_conflict = (
            second_idx is not None
            and top_prob >= dual_category_threshold
            and second_prob >= dual_category_threshold
            and top_second_gap <= dual_max_gap
            and second_third_gap >= dual_min_margin_from_third
        )

        # Main rule: high-confidence single class.
        is_main = (
            not is_dual_conflict
            and top_prob >= main_threshold
        )

        # Subset rule: one dominant class in the moderate range (not high enough for main).
        # Example target behavior:
        # - 0.60/0.40 -> subset of top class
        # - 0.50/(rest split) -> subset of top class
        # - 0.50/0.45 -> dual (if dual conditions are met), not subset
        is_subset = (
            not is_dual_conflict
            and not is_main
            and top_prob >= subset_threshold
            and (second_idx is None or second_prob < subset_threshold)
        )

        # Trash rule: low and split probabilities (no clear dominant category).
        is_trash = (
            not is_dual_conflict
            and not is_main
            and not is_subset
            and top_prob <= trash_top_max
            and top_second_gap <= trash_max_gap
            and third_prob >= trash_third_min
        )

        # Priority: dual > main > subset > trash.
        # Any remaining non-confident case is forced to Trash (no Uncertain bucket).
        is_uncertain = False
        if (
            not is_dual_conflict
            and not is_main
            and not is_subset
            and not is_trash
        ):
            is_trash = True
        predicted = TRASH_LABEL if is_trash else top_class
        assigned_categories_list = [top_class, second_class] if is_dual_conflict else [top_class]
        if is_trash:
            assigned_categories_list = [TRASH_LABEL]
        assigned_categories = "|".join([c for c in assigned_categories_list if c])
        assignment_mode = (
            "dual"
            if is_dual_conflict
            else (
                "main"
                if is_main
                else (
                    "subset"
                    if is_subset
                    else ("trash" if is_trash else "single")
                )
            )
        )
        subset_category = top_class if is_subset else ""
        main_category = top_class if is_main else ""
        all_probs = {categories[i]: float(row_probs[i]) for i in range(len(categories))}
        results.append(
            {
                "predicted_category": predicted,
                "top_class": top_class,
                "second_class": second_class,
                "second_probability": second_prob,
                "top_second_gap": top_second_gap,
                "second_third_gap": second_third_gap,
                "probability": top_prob,
                "is_uncertain": is_uncertain,
                "is_dual_conflict": is_dual_conflict,
                "is_main": is_main,
                "is_subset": is_subset,
                "is_trash": is_trash,
                "main_category": main_category,
                "subset_category": subset_category,
                "assignment_mode": assignment_mode,
                "assigned_categories": assigned_categories,
                "assigned_categories_list": assigned_categories_list,
                "all_probabilities": all_probs,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference from Google Sheets (text-only) using trained 5-class model."
    )
    parser.add_argument(
        "--google_sheet",
        type=str,
        nargs="+",
        required=True,
        help="One or more Google Sheet IDs or full URLs.",
    )
    parser.add_argument(
        "--sheet_gid",
        type=int,
        nargs="*",
        default=[0],
        help="One or more GIDs (tabs) to read from each sheet. Example: --sheet_gid 0 1 2 3 4",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        nargs="*",
        default=None,
        help="Optional tab names. If provided, reads tabs by name via gviz CSV (more reliable than guessing gid).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Exact text column name. If omitted, script auto-detects it.",
    )
    parser.add_argument(
        "--date_column",
        type=str,
        default=None,
        help="Optional date column name. If omitted, script auto-detects common date labels.",
    )
    parser.add_argument(
        "--location_column",
        type=str,
        default=None,
        help="Optional location column name. If omitted, script auto-detects common location labels.",
    )
    parser.add_argument(
        "--platform_column",
        type=str,
        default=None,
        help="Optional platform column name. If omitted, script auto-detects common platform labels.",
    )
    parser.add_argument("--max_rows", type=int, default=None, help="Optional max rows per sheet tab.")
    parser.add_argument("--model_dir", type=str, default="output", help="Directory with trained artifacts.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for model.predict.")
    parser.add_argument(
        "--uncertain_threshold",
        type=float,
        default=DEFAULT_UNCERTAIN_THRESHOLD,
        help="Deprecated (no effect): uncertain bucket is disabled; low-confidence rows go to Trash.",
    )
    parser.add_argument(
        "--dual_threshold",
        type=float,
        default=DEFAULT_DUAL_CATEGORY_THRESHOLD,
        help=(
            "Minimum probability required for BOTH top categories to consider dual assignment."
        ),
    )
    parser.add_argument(
        "--dual_max_gap",
        type=float,
        default=DEFAULT_DUAL_MAX_GAP,
        help=(
            "Maximum allowed difference between top and second probability "
            "(e.g., 0.10 allows 0.45 vs 0.50)."
        ),
    )
    parser.add_argument(
        "--dual_min_margin_third",
        type=float,
        default=DEFAULT_DUAL_MIN_MARGIN_FROM_THIRD,
        help=(
            "Minimum required margin between second and third probability to avoid "
            "3-way ambiguity."
        ),
    )
    parser.add_argument(
        "--subset_threshold",
        type=float,
        default=DEFAULT_SUBSET_THRESHOLD,
        help=(
            "If one category reaches this probability (and row is not dual), mark it as subset "
            "of that category."
        ),
    )
    parser.add_argument(
        "--main_threshold",
        type=float,
        default=DEFAULT_MAIN_THRESHOLD,
        help=(
            "If one category reaches this probability (and row is not dual), mark it as main "
            "high-confidence category."
        ),
    )
    parser.add_argument(
        "--trash_top_max",
        type=float,
        default=DEFAULT_TRASH_TOP_MAX,
        help="Trash rule: maximum allowed top probability for low-split rows.",
    )
    parser.add_argument(
        "--trash_max_gap",
        type=float,
        default=DEFAULT_TRASH_MAX_GAP,
        help="Trash rule: maximum top-vs-second gap for low-split rows.",
    )
    parser.add_argument(
        "--trash_third_min",
        type=float,
        default=DEFAULT_TRASH_THIRD_MIN,
        help="Trash rule: minimum third probability to confirm split distribution.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="batch_predictions.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    model, tokenizer, maxlen, categories = load_model_and_tokenizer(args.model_dir)

    # Collect rows from all requested sheets/tabs.
    all_rows = []
    for sheet_arg in args.google_sheet:
        if args.sheet_name:
            for tab_name in args.sheet_name:
                csv_url = build_google_sheet_gviz_csv_url(sheet_arg, tab_name)
                try:
                    rows, detected_text_col, detected_date_col, detected_location_col, detected_platform_col = load_text_rows_from_sheet(
                        csv_url=csv_url,
                        text_column=args.text_column,
                        date_column=args.date_column,
                        location_column=args.location_column,
                        platform_column=args.platform_column,
                        max_rows=args.max_rows,
                    )
                    for r in rows:
                        all_rows.append(
                            {
                                "sheet": sheet_arg,
                                "sheet_name": tab_name,
                                "gid": "",
                                "row_number": r["row_number"],
                                "text": r["text"],
                                "date": r["date"],
                                "location": r["location"],
                                "platform": r["platform"],
                                "text_column": detected_text_col,
                                "date_column": detected_date_col,
                                "location_column": detected_location_col,
                                "platform_column": detected_platform_col,
                            }
                        )
                    print(
                        f"Loaded {len(rows)} rows from sheet={sheet_arg[:40]}... "
                        f"tab={tab_name} (text column: {detected_text_col}, "
                        f"date column: {detected_date_col or '-'}, "
                        f"location column: {detected_location_col or '-'}, "
                        f"platform column: {detected_platform_col or '-'})"
                    )
                except Exception as exc:
                    print(f"Warning: skipping sheet={sheet_arg[:40]}... tab={tab_name}. Reason: {exc}")
        else:
            # If URL already carries gid and user did not pass explicit list, keep that gid.
            gids = args.sheet_gid if args.sheet_gid else [0]
            if sheet_arg.startswith("http") and args.sheet_gid == [0]:
                parsed_gid = _gid_from_url(sheet_arg)
                gids = [parsed_gid]

            for gid in gids:
                csv_url = build_google_sheet_csv_url(sheet_arg, gid=gid)
                try:
                    rows, detected_text_col, detected_date_col, detected_location_col, detected_platform_col = load_text_rows_from_sheet(
                        csv_url=csv_url,
                        text_column=args.text_column,
                        date_column=args.date_column,
                        location_column=args.location_column,
                        platform_column=args.platform_column,
                        max_rows=args.max_rows,
                    )
                    for r in rows:
                        all_rows.append(
                            {
                                "sheet": sheet_arg,
                                "sheet_name": "",
                                "gid": gid,
                                "row_number": r["row_number"],
                                "text": r["text"],
                                "date": r["date"],
                                "location": r["location"],
                                "platform": r["platform"],
                                "text_column": detected_text_col,
                                "date_column": detected_date_col,
                                "location_column": detected_location_col,
                                "platform_column": detected_platform_col,
                            }
                        )
                    print(
                        f"Loaded {len(rows)} rows from sheet={sheet_arg[:40]}... gid={gid} "
                        f"(text column: {detected_text_col}, "
                        f"date column: {detected_date_col or '-'}, "
                        f"location column: {detected_location_col or '-'}, "
                        f"platform column: {detected_platform_col or '-'})"
                    )
                except Exception as exc:
                    print(f"Warning: skipping sheet={sheet_arg[:40]}... gid={gid}. Reason: {exc}")

    if not all_rows:
        raise SystemExit("No rows loaded from any sheet/gid.")

    predictable_indices = [idx for idx, row in enumerate(all_rows) if (row.get("text") or "").strip()]
    texts = [all_rows[idx]["text"] for idx in predictable_indices]
    predictions = []
    if texts:
        predictions = predict_batch(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            maxlen=maxlen,
            categories=categories,
            batch_size=args.batch_size,
            uncertain_threshold=args.uncertain_threshold,
            dual_category_threshold=args.dual_threshold,
            dual_max_gap=args.dual_max_gap,
            dual_min_margin_from_third=args.dual_min_margin_third,
            subset_threshold=args.subset_threshold,
            main_threshold=args.main_threshold,
            trash_top_max=args.trash_top_max,
            trash_max_gap=args.trash_max_gap,
            trash_third_min=args.trash_third_min,
        )
    prediction_by_row_index = {row_idx: pred for row_idx, pred in zip(predictable_indices, predictions)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prob_cols = [f"prob_{_slugify_category(c)}" for c in categories]
    fieldnames = [
        "sheet",
        "sheet_name",
        "gid",
        "row_number",
        "text_column",
        "date_column",
        "location_column",
        "platform_column",
        "text",
        "date",
        "location",
        "platform",
        "predicted_category",
        "assignment_mode",
        "assigned_categories",
        "top_class",
        "second_class",
        "main_category",
        "subset_category",
        "probability",
        "second_probability",
        "top_second_gap",
        "second_third_gap",
        "is_uncertain",
        "is_dual_conflict",
        "is_main",
        "is_subset",
        "is_trash",
    ] + prob_cols

    missing_rows = len(all_rows) - len(predictions)
    if missing_rows > 0:
        print(f"Warning: {missing_rows} row(s) are missing text and will be excluded from output CSV.")

    written_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(all_rows):
            pred = prediction_by_row_index.get(idx)
            if pred is None:
                continue
            rec = {
                "sheet": row["sheet"],
                "sheet_name": row.get("sheet_name", ""),
                "gid": row["gid"],
                "row_number": row["row_number"],
                "text_column": row["text_column"],
                "date_column": row.get("date_column", ""),
                "location_column": row.get("location_column", ""),
                "platform_column": row.get("platform_column", ""),
                "text": row["text"],
                "date": row.get("date", ""),
                "location": row.get("location", ""),
                "platform": row.get("platform", ""),
                "predicted_category": pred["predicted_category"],
                "assignment_mode": pred["assignment_mode"],
                "assigned_categories": pred["assigned_categories"],
                "top_class": pred["top_class"],
                "second_class": pred["second_class"],
                "main_category": pred["main_category"],
                "subset_category": pred["subset_category"],
                "probability": f"{pred['probability']:.6f}",
                "second_probability": f"{pred['second_probability']:.6f}",
                "top_second_gap": f"{pred['top_second_gap']:.6f}",
                "second_third_gap": f"{pred['second_third_gap']:.6f}",
                "is_uncertain": str(bool(pred["is_uncertain"])).lower(),
                "is_dual_conflict": str(bool(pred["is_dual_conflict"])).lower(),
                "is_main": str(bool(pred["is_main"])).lower(),
                "is_subset": str(bool(pred["is_subset"])).lower(),
                "is_trash": str(bool(pred["is_trash"])).lower(),
            }
            for cat in categories:
                rec[f"prob_{_slugify_category(cat)}"] = f"{pred['all_probabilities'][cat]:.6f}"
            writer.writerow(rec)
            written_rows += 1

    if written_rows != len(predictions):
        raise SystemExit(
            f"Output validation failed: wrote {written_rows} rows but expected {len(predictions)} predicted rows."
        )

    # Simple summary
    summary: Dict[str, int] = {}
    assigned_summary: Dict[str, int] = {}
    mode_summary: Dict[str, int] = {}
    for pred in predictions:
        summary[pred["predicted_category"]] = summary.get(pred["predicted_category"], 0) + 1
        for cat in pred.get("assigned_categories_list", []):
            assigned_summary[cat] = assigned_summary.get(cat, 0) + 1
        mode_summary[pred["assignment_mode"]] = mode_summary.get(pred["assignment_mode"], 0) + 1

    print(f"\nSaved predictions: {out_path.resolve()}")
    print(f"Total rows loaded: {len(all_rows)}")
    print(f"Rows missing text (excluded): {missing_rows}")
    print(f"Total rows written to CSV: {written_rows}")
    print("Predicted category counts:")
    for cat in sorted(summary.keys()):
        print(f"  {cat}: {summary[cat]}")
    print("Assigned category counts (dual-label aware):")
    for cat in sorted(assigned_summary.keys()):
        print(f"  {cat}: {assigned_summary[cat]}")
    print("Assignment modes:")
    for mode in sorted(mode_summary.keys()):
        print(f"  {mode}: {mode_summary[mode]}")


if __name__ == "__main__":
    main()
