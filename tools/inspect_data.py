from __future__ import annotations

import argparse
from pathlib import Path

from data import load_math_dataframe
from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load MATH dataset from HF and print dataset preview"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap for number of rows to inspect",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="",
        help="Optional path to save preview dataframe as CSV",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="How many sample rows to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit > 0 else None

    result = load_math_dataframe(limit=limit)
    df = result.dataframe

    print("=== DATASET SOURCE ===")
    print(f"dataset_id: {result.dataset_id}")
    print(
        "source columns: "
        f"problem={result.problem_col}, "
        f"topic={result.topic_col}, "
        f"level={result.level_col}"
    )

    print("\n=== SHAPE ===")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")

    print("\n=== TOPIC DISTRIBUTION (top 15) ===")
    print(df["topic"].value_counts().head(15).to_string())

    print("\n=== DIFFICULTY DISTRIBUTION ===")
    print(df["difficulty"].value_counts().to_string())

    print("\n=== LEVEL DISTRIBUTION ===")
    print(df["level_num"].value_counts().sort_index().to_string())

    print(f"\n=== SAMPLE ({args.sample_rows} rows) ===")
    preview = df[["problem", "topic", "level_num", "difficulty"]]
    print(preview.head(args.sample_rows).to_string(index=False))

    if args.save_csv:
        save_path = Path(args.save_csv)
        ensure_dir(save_path.parent)
        df.to_csv(save_path, index=False)
        print(f"\nSaved full normalized dataframe to: {save_path}")


if __name__ == "__main__":
    main()
