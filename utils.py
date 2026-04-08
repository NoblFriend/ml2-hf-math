import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_level_to_int(level_value: Any) -> int | None:
    if level_value is None:
        return None

    if isinstance(level_value, (int, float)):
        level_num = int(level_value)
        return level_num if 1 <= level_num <= 5 else None

    if isinstance(level_value, str):
        match = re.search(r"(\d+)", level_value)
        if not match:
            return None
        level_num = int(match.group(1))
        return level_num if 1 <= level_num <= 5 else None

    return None


def level_to_difficulty(level: int) -> str:
    if level in (1, 2):
        return "easy"
    if level == 3:
        return "medium"
    return "hard"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def select_top95_topics(
    topic_probs: Dict[str, float],
    threshold: float = 0.95,
) -> List[Dict[str, float]]:
    sorted_items = sorted(
        topic_probs.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    cumulative = 0.0
    selected: List[Dict[str, float]] = []

    for topic, prob in sorted_items:
        selected.append({"topic": topic, "probability": float(prob)})
        cumulative += prob
        if cumulative >= threshold:
            break

    return selected
