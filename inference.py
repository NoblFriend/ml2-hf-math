from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from config import ARTIFACTS_DIR, MIN_TEXT_LENGTH
from data import soft_clean_math_text
from model import MultiTaskDistilBertClassifier
from transformers import AutoTokenizer
from utils import get_device, load_json, select_top95_topics, softmax


@dataclass
class InferenceResult:
    best_topic: str
    best_difficulty: str
    difficulty_score: float
    difficulty_level: int
    topic_probabilities: Dict[str, float]
    difficulty_probabilities: Dict[str, float]
    top95_topics: List[Dict[str, float]]


class MathProblemInferenceService:
    def __init__(self, artifacts_dir: str | Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.device = get_device()

        (
            self.model,
            self.tokenizer,
            self.mappings,
            self.max_length,
        ) = self._load_artifacts()

    def _load_artifacts(self):
        model_path = self.artifacts_dir / "model.pt"
        tokenizer_dir = self.artifacts_dir / "tokenizer"
        mappings_path = self.artifacts_dir / "label_mappings.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_dir}")
        if not mappings_path.exists():
            raise FileNotFoundError(
                f"Label mappings not found: {mappings_path}"
            )

        checkpoint = torch.load(model_path, map_location=self.device)
        mappings = load_json(mappings_path)

        model_name = checkpoint["model_name"]
        num_topics = int(checkpoint["num_topics"])
        num_difficulties = int(checkpoint["num_difficulties"])
        max_length = int(checkpoint.get("max_length", 256))

        model = MultiTaskDistilBertClassifier(
            model_name=model_name,
            num_topics=num_topics,
            num_difficulties=num_difficulties,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        return model, tokenizer, mappings, max_length

    @staticmethod
    def _score_to_difficulty(level: int) -> str:
        if level <= 2:
            return "easy"
        if level == 3:
            return "medium"
        return "hard"

    @staticmethod
    def _difficulty_probs_from_score(score: float) -> Dict[str, float]:
        centers = np.array([1.5, 3.0, 4.5], dtype=np.float32)
        labels = ["easy", "medium", "hard"]
        distances = -np.abs(centers - score)
        probs = softmax(distances)
        return {
            label: float(prob)
            for label, prob in zip(labels, probs.tolist())
        }

    def predict(self, text: str) -> InferenceResult:
        normalized_text = soft_clean_math_text(text.strip())

        if not normalized_text:
            raise ValueError("Input is empty. Please enter a math problem.")
        if len(normalized_text) < MIN_TEXT_LENGTH:
            raise ValueError(
                "Input is too short. Please provide at least "
                f"{MIN_TEXT_LENGTH} characters."
            )

        encoded = self.tokenizer(
            normalized_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
            )

        topic_logits = outputs.topic_logits.squeeze(0).detach().cpu().numpy()
        difficulty_score = float(
            outputs.difficulty_logits.squeeze(0).detach().cpu().item()
        )
        difficulty_score = float(np.clip(difficulty_score, 1.0, 5.0))
        difficulty_level = int(np.clip(np.rint(difficulty_score), 1, 5))

        topic_probs_array = softmax(topic_logits)

        id_to_topic = {
            int(k): v
            for k, v in self.mappings["id_to_topic"].items()
        }

        topic_probs = {
            id_to_topic[idx]: float(prob)
            for idx, prob in enumerate(topic_probs_array.tolist())
        }
        diff_probs = self._difficulty_probs_from_score(difficulty_score)

        best_topic = max(topic_probs.items(), key=lambda x: x[1])[0]
        best_difficulty = self._score_to_difficulty(difficulty_level)

        top95 = select_top95_topics(topic_probs, threshold=0.95)

        return InferenceResult(
            best_topic=best_topic,
            best_difficulty=best_difficulty,
            difficulty_score=difficulty_score,
            difficulty_level=difficulty_level,
            topic_probabilities=dict(
                sorted(topic_probs.items(), key=lambda x: x[1], reverse=True)
            ),
            difficulty_probabilities=dict(
                sorted(diff_probs.items(), key=lambda x: x[1], reverse=True)
            ),
            top95_topics=top95,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for MATH problem classifier"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(ARTIFACTS_DIR),
    )
    parser.add_argument("--text", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = MathProblemInferenceService(args.artifacts_dir)
    result = service.predict(args.text)

    print("Predicted topic:", result.best_topic)
    print("Predicted difficulty:", result.best_difficulty)
    print("Predicted level score [1..5]:", f"{result.difficulty_score:.3f}")
    print("Topic probabilities:")
    for topic, prob in result.topic_probabilities.items():
        print(f"  {topic}: {prob:.4f}")

    print("Difficulty probabilities:")
    for label, prob in result.difficulty_probabilities.items():
        print(f"  {label}: {prob:.4f}")

    print("Top-95% topics:")
    for item in result.top95_topics:
        print(f"  {item['topic']}: {item['probability']:.4f}")


if __name__ == "__main__":
    main()
