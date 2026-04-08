from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from config import DATASET_CANDIDATES
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from utils import level_to_difficulty, parse_level_to_int


def soft_clean_math_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\$\$", " ", cleaned)
    cleaned = re.sub(r"\$", " ", cleaned)
    cleaned = re.sub(r"\\left|\\right", " ", cleaned)
    cleaned = re.sub(r"\\([A-Za-z]+)", r"\1", cleaned)
    cleaned = cleaned.replace("\\", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass
class DataLoadResult:
    dataframe: pd.DataFrame
    dataset_id: str
    problem_col: str
    topic_col: str
    level_col: str


class MathTextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        topic_labels: List[int],
        difficulty_labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.topic_labels = topic_labels
        self.difficulty_labels = difficulty_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = soft_clean_math_text(self.texts[index])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "topic_labels": torch.tensor(
                self.topic_labels[index],
                dtype=torch.long,
            ),
            "difficulty_labels": torch.tensor(
                self.difficulty_labels[index],
                dtype=torch.float32,
            ),
        }


def _find_column(columns: Iterable[str], candidates: List[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def _normalize_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, str, str]:
    problem_col = _find_column(df.columns, ["problem", "question", "text"])
    topic_col = _find_column(df.columns, ["type", "subject", "topic"])
    level_col = _find_column(df.columns, ["level", "difficulty", "grade"])

    if not problem_col or not topic_col or not level_col:
        raise ValueError(
            "Required columns were not found. Need problem/question/text, "
            "type/subject/topic, and level/difficulty/grade."
        )

    normalized = df[[problem_col, topic_col, level_col]].copy()
    normalized.columns = ["problem", "topic", "level"]

    normalized["problem"] = (
        normalized["problem"].astype(str).map(soft_clean_math_text)
    )
    normalized["topic"] = (
        normalized["topic"].astype(str).str.strip().str.lower()
    )
    normalized["level_num"] = normalized["level"].apply(parse_level_to_int)

    normalized = normalized.dropna(subset=["problem", "topic", "level_num"])
    normalized = normalized[normalized["problem"].str.len() > 0]
    normalized["level_num"] = normalized["level_num"].astype(int)
    normalized = normalized[
        (normalized["level_num"] >= 1) & (normalized["level_num"] <= 5)
    ]

    normalized["difficulty"] = normalized["level_num"].apply(
        level_to_difficulty
    )

    return (
        normalized[["problem", "topic", "level_num", "difficulty"]],
        problem_col,
        topic_col,
        level_col,
    )


def load_math_dataframe(limit: int | None = None) -> DataLoadResult:
    last_error: Exception | None = None

    for dataset_id in DATASET_CANDIDATES:
        try:
            dataset = load_dataset(dataset_id)
            if not isinstance(dataset, DatasetDict):
                raise ValueError(
                    f"Unsupported dataset format for {dataset_id}"
                )

            all_parts = []
            for split_name in dataset.keys():
                split_df = pd.DataFrame(dataset[split_name])
                if not split_df.empty:
                    all_parts.append(split_df)

            if not all_parts:
                raise ValueError(f"No non-empty splits found for {dataset_id}")

            combined_df = pd.concat(all_parts, ignore_index=True)
            (
                normalized,
                problem_col,
                topic_col,
                level_col,
            ) = _normalize_dataframe(combined_df)

            if limit is not None and limit > 0:
                normalized = normalized.sample(
                    n=min(limit, len(normalized)),
                    random_state=42,
                ).reset_index(drop=True)

            return DataLoadResult(
                dataframe=normalized.reset_index(drop=True),
                dataset_id=dataset_id,
                problem_col=problem_col,
                topic_col=topic_col,
                level_col=level_col,
            )
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Failed to load a compatible MATH dataset from "
        "Hugging Face candidates: "
        f"{DATASET_CANDIDATES}. Last error: {last_error}"
    )
