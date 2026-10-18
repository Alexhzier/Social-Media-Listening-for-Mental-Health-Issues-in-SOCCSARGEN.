# CNN-LSTM 5-Class Distress-Pattern Classifier (Cebuano, Tagalog, English)

Research-oriented text classifier that predicts which category of mental health-related linguistic pattern a text best matches.

Supported classes:
1. Depression
2. Stress
3. Anxiety
4. Addiction
5. Suicidal Behavior

This model does not diagnose mental illness. It is for academic use and screening support only.

## Setup

```bash
cd distress_classifier
python -m pip install -r requirements.txt
```

## Data Source

This project is configured to train from Google Sheets.
Local `data.py` is intentionally blank.

Expected category keys in sheets:
- `depression`
- `stress`
- `anxiety`
- `addiction`
- `suicidal_behavior`

## Train and Evaluate

```bash
python main.py --google_sheet YOUR_SHEET_ID --epochs 25 --save_dir output
```

You can pass multiple sheet IDs:

```bash
python main.py --google_sheet ID1 ID2 ID3 --epochs 25 --save_dir output
```

Optional tab index:

```bash
python main.py --google_sheet YOUR_SHEET_ID --sheet_gid 0
```

Saves to `output/`:
- `best_model.keras`
- `tokenizer.pkl`
- `tokenizer_config.json`
- `categories.json`
- `metrics.json`
- `confusion_matrix.json`
- `training_history.json`
- `loss_curve.png`
- `accuracy_curve.png`

## Inference

```bash
python predict.py "I have been feeling really down and alone lately." --model_dir output
```

## Batch Inference From Google Sheets (Text-Only)

Use this for large-scale runs on sheet rows that contain text only.

Example (single tab):

```bash
python batch_predict_sheet.py --google_sheet YOUR_SHEET_LINK --sheet_gid 0 --model_dir output --out batch_predictions.csv
```

Example (5 tabs in one workbook):

```bash
python batch_predict_sheet.py --google_sheet YOUR_SHEET_LINK --sheet_gid 0 1 2 3 4 --model_dir output --out batch_predictions.csv
```

Recommended (read tabs by name, avoids gid issues):

```bash
python batch_predict_sheet.py --google_sheet YOUR_SHEET_LINK --sheet_name "Depression" "Anxiety" "Stress" "Suicidal behavior" "Addiction" --model_dir output --out batch_predictions.csv
```

Output CSV includes:
- original text
- original `date` and `location` (auto-detected if present, or set with `--date_column` / `--location_column`)
- predicted category
- top-class probability
- `Uncertain` bucket disabled: low-confidence rows are assigned to `Trash`
- dual-category handling (close-conflict only):
  - `--dual_threshold` (minimum for top 2, default `0.40`)
  - `--dual_max_gap` (top-vs-second closeness, default `0.10`)
  - `--dual_min_margin_third` (second-vs-third separation, default `0.08`)
- subset handling for single dominant class:
  - `--subset_threshold` (default `0.50`)
- main handling for high-confidence single class:
  - `--main_threshold` (default `0.70`)
- trash handling for low/split confidence rows:
  - `--trash_top_max` (default `0.35`)
  - `--trash_max_gap` (default `0.12`)
  - `--trash_third_min` (default `0.15`)
- `assignment_mode`, `assigned_categories`, `second_class`, `main_category`, `subset_category`,
  `second_probability`, `top_second_gap`, `second_third_gap`,
  `is_dual_conflict`, `is_main`, `is_subset`, `is_trash`
- per-class probabilities

If your sheet has non-standard column names, pass them explicitly:

```bash
python batch_predict_sheet.py --google_sheet YOUR_SHEET_LINK --sheet_name "Sheet1" --text_column "post_text" --date_column "event_date" --location_column "event_location" --model_dir output --out batch_predictions.csv
```

## Ethical Use

This tool classifies linguistic patterns only. It does not diagnose mental illness and should not replace professional care.

