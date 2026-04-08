# Math Problem Analyzer (Transformer MVP)

End-to-end service for classifying mathematical problems by text with a multitask transformer model:
- Topic classification (original MATH subject/type labels)
- Difficulty classification (easy/medium/hard)

The project includes:
- Programmatic dataset download from Hugging Face
- Multitask DistilBERT training
- Artifact saving
- Inference module
- Streamlit web UI

## Dataset source

The code tries Hugging Face dataset candidates in this order:
1. `hendrycks/competition_math` (default, preferred)
2. `nlile/hendrycks-MATH-benchmark` (fallback)

The loader automatically maps real column names to required semantics:
- `problem` / `question` / `text`
- `type` / `subject` / `topic`
- `level` / `difficulty` / `grade`

## Project structure

```text
math_problem_analyzer/
  app.py
  train.py
  inference.py
  model.py
  data.py
  utils.py
  config.py
  requirements.txt
  README.md
  artifacts/
  examples/
    geometry.txt
    number_theory.txt
    algebra.txt
```

## 1. Install dependencies with uv

```bash
cd math_problem_analyzer
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternative without activating venv:

```bash
uv run --with-requirements requirements.txt python -V
```

## 2. Step-by-step pipeline (recommended)

### Step 1: Load and inspect dataset

```bash
bash scripts/01_inspect_data.sh
```

This will:
- download dataset from Hugging Face
- normalize fields (`problem`, `topic`, `level_num`, `difficulty`)
- print distributions and sample rows
- save preview to `artifacts/data_preview.csv`

### Step 2: Quick train

```bash
bash scripts/02_train_quick.sh
```

This run includes:
- `tqdm` progress bar
- TensorBoard logs to `artifacts/tb_logs`

### Step 3: Optional longer training (continue improving)

```bash
bash scripts/03_train_long.sh
```

### Step 4: Local CLI inference

```bash
bash scripts/04_infer_cli.sh "Find the remainder when 7^2025 is divided by 13"
```

### Step 5: Launch local Streamlit app

```bash
bash scripts/05_run_streamlit.sh
```

### Step 6: Watch training curves in TensorBoard

```bash
bash scripts/06_tensorboard.sh
```

## 3. Direct training command (manual)

```bash
uv run python train.py --epochs 3 --batch-size 8 --lr 2e-5 --max-length 256 --artifacts-dir artifacts
uv run python train.py --limit 1500
```

Training saves into `artifacts/`:
- `model.pt`
- `tokenizer/`
- `label_mappings.json`
- `metadata.json`
- `train_history.csv`

## 4. Run inference from CLI

```bash
uv run python inference.py --text "Find the remainder when 7^2025 is divided by 13"
```

Output includes:
- best topic
- topic probabilities
- top-95% topics by cumulative probability
- best difficulty
- difficulty probabilities

## 5. Run Streamlit app

```bash
uv run streamlit run app.py
```

UI includes:
- large input text area
- Analyze button
- result section with topic, difficulty, confidence
- top-95% topics
- probability tables
- example buttons (geometry, number theory, algebra)
- user-friendly error handling for missing weights / bad input

## Difficulty mapping

Raw level is parsed into integer and mapped as:
- easy: levels 1-2
- medium: level 3
- hard: levels 4-5

If raw level is a string (for example `Level 2`), numeric part is extracted automatically.

## Mac and MPS support

Device selection is automatic:
- if `torch.backends.mps.is_available()` -> use `mps`
- else -> use `cpu`

No CUDA-specific code is used.

## Metrics on validation set

The trainer reports:
- topic accuracy
- topic macro F1
- difficulty accuracy
- difficulty macro F1

Early stopping is applied using topic macro F1 (with validation loss tie-break).

## Notes

- This project does **classification only**.
- It does not generate hints, solutions, or reasoning steps.
