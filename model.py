from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import DistilBertModel


@dataclass
class MultiTaskOutput:
    loss: Optional[torch.Tensor]
    topic_loss: Optional[torch.Tensor]
    difficulty_loss: Optional[torch.Tensor]
    topic_logits: torch.Tensor
    difficulty_logits: torch.Tensor


class MultiTaskDistilBertClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_topics: int,
        num_difficulties: int,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained(model_name)

        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.topic_head = nn.Linear(hidden_size, num_topics)
        self.difficulty_head = nn.Linear(hidden_size, 1)

        self.topic_loss_fn = nn.CrossEntropyLoss()
        self.difficulty_loss_fn = nn.L1Loss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        topic_labels: Optional[torch.Tensor] = None,
        difficulty_labels: Optional[torch.Tensor] = None,
    ) -> MultiTaskOutput:
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = encoder_output.last_hidden_state[:, 0]
        shared_repr = self.dropout(cls_embedding)

        topic_logits = self.topic_head(shared_repr)
        difficulty_logits = self.difficulty_head(shared_repr).squeeze(-1)

        topic_loss: Optional[torch.Tensor] = None
        difficulty_loss: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = None

        if topic_labels is not None:
            topic_loss = self.topic_loss_fn(topic_logits, topic_labels)

        if difficulty_labels is not None:
            difficulty_labels = difficulty_labels.view(-1).to(
                difficulty_logits.dtype
            )
            difficulty_logits = difficulty_logits.view(-1)
            difficulty_loss = self.difficulty_loss_fn(
                difficulty_logits,
                difficulty_labels,
            )

        if topic_loss is not None and difficulty_loss is not None:
            total_loss = topic_loss + 0.3 * difficulty_loss

        return MultiTaskOutput(
            loss=total_loss,
            topic_loss=topic_loss,
            difficulty_loss=difficulty_loss,
            topic_logits=topic_logits,
            difficulty_logits=difficulty_logits,
        )
