from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from config import (ARTIFACTS_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS,
                    DEFAULT_LR, DEFAULT_MODEL_NAME, DEFAULT_PATIENCE,
                    DEFAULT_RANDOM_STATE, DEFAULT_VAL_SIZE, MAX_LENGTH)
from data import MathTextDataset, load_math_dataframe
from model import MultiTaskDistilBertClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multitask classifier for MATH problems"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(ARTIFACTS_DIR),
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--tb-log-dir",
        type=str,
        default=str(ARTIFACTS_DIR / "tb_logs"),
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: cap dataset size for quick debug",
    )
    return parser.parse_args()


def _build_dataloaders(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int,
    max_length: int,
    val_size: float,
    seed: int,
) -> Tuple[
    DataLoader,
    DataLoader,
    LabelEncoder,
    pd.DataFrame,
    pd.DataFrame,
]:
    topic_encoder = LabelEncoder()
    df["topic_id"] = topic_encoder.fit_transform(df["topic"])

    df["difficulty_target"] = df["level_num"].astype(float)

    split_kwargs = {
        "test_size": val_size,
        "random_state": seed,
    }

    try:
        train_df, val_df = train_test_split(
            df,
            stratify=df["topic_id"],
            **split_kwargs,
        )
    except ValueError:
        train_df, val_df = train_test_split(df, **split_kwargs)

    train_dataset = MathTextDataset(
        texts=train_df["problem"].tolist(),
        topic_labels=train_df["topic_id"].tolist(),
        difficulty_labels=train_df["difficulty_target"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = MathTextDataset(
        texts=val_df["problem"].tolist(),
        topic_labels=val_df["topic_id"].tolist(),
        difficulty_labels=val_df["difficulty_target"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        topic_encoder,
        train_df,
        val_df,
    )


def evaluate(
    model,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    losses = []
    topic_targets = []
    topic_preds = []
    diff_targets = []
    diff_preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            topic_labels = batch["topic_labels"].to(device)
            difficulty_labels = batch["difficulty_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                topic_labels=topic_labels,
                difficulty_labels=difficulty_labels,
            )

            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().cpu().item()))

            topic_batch_preds = torch.argmax(outputs.topic_logits, dim=1)
            diff_batch_preds = outputs.difficulty_logits.view(-1)

            topic_targets.extend(topic_labels.detach().cpu().numpy().tolist())
            topic_preds.extend(
                topic_batch_preds.detach().cpu().numpy().tolist()
            )
            diff_targets.extend(
                difficulty_labels.detach().cpu().numpy().tolist()
            )
            diff_preds.extend(diff_batch_preds.detach().cpu().numpy().tolist())

    diff_targets_array = np.array(diff_targets, dtype=np.float32)
    diff_preds_array = np.array(diff_preds, dtype=np.float32)

    diff_preds_clipped = np.clip(diff_preds_array, 1.0, 5.0)
    diff_preds_rounded = np.rint(diff_preds_clipped).astype(int)
    diff_targets_int = np.rint(diff_targets_array).astype(int)

    return {
        "val_loss": float(np.mean(losses)) if losses else float("inf"),
        "topic_accuracy": accuracy_score(topic_targets, topic_preds),
        "topic_macro_f1": f1_score(
            topic_targets,
            topic_preds,
            average="macro",
            zero_division=0,
        ),
        "difficulty_mae": float(
            np.mean(np.abs(diff_targets_array - diff_preds_array))
        ),
        "difficulty_rmse": float(
            np.sqrt(np.mean((diff_targets_array - diff_preds_array) ** 2))
        ),
        "difficulty_level_accuracy": accuracy_score(
            diff_targets_int,
            diff_preds_rounded,
        ),
    }


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    ensure_dir(artifacts_dir)
    tb_log_dir = Path(args.tb_log_dir)
    ensure_dir(tb_log_dir)
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    limit = args.limit if args.limit > 0 else None
    data_result = load_math_dataframe(limit=limit)
    df = data_result.dataframe
    df["level_num"] = df["level_num"].astype("float32")
    df = df[df["problem"].str.len() >= 30].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    (
        train_loader,
        val_loader,
        topic_encoder,
        train_df,
        val_df,
    ) = _build_dataloaders(
        df=df,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_size=args.val_size,
        seed=args.seed,
    )

    device = get_device()
    print(f"Using device: {device}")

    model = MultiTaskDistilBertClassifier(
        model_name=args.model_name,
        num_topics=len(topic_encoder.classes_),
        num_difficulties=1,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )

    best_topic_f1 = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None
    history = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            topic_labels = batch["topic_labels"].to(device)
            difficulty_labels = batch["difficulty_labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                topic_labels=topic_labels,
                difficulty_labels=difficulty_labels,
            )

            if outputs.loss is None:
                continue

            outputs.loss.backward()
            optimizer.step()
            scheduler.step()

            batch_loss = float(outputs.loss.detach().cpu().item())
            running_loss += batch_loss
            progress.set_postfix({"train_loss": f"{batch_loss:.4f}"})

            writer.add_scalar("train/batch_loss", batch_loss, global_step)
            global_step += 1

        metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / max(len(train_loader), 1)
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(epoch_row)

        writer.add_scalar("train/epoch_loss", train_loss, epoch + 1)
        writer.add_scalar("val/loss", metrics["val_loss"], epoch + 1)
        writer.add_scalar(
            "val/topic_accuracy",
            metrics["topic_accuracy"],
            epoch + 1,
        )
        writer.add_scalar(
            "val/topic_macro_f1",
            metrics["topic_macro_f1"],
            epoch + 1,
        )
        writer.add_scalar(
            "val/difficulty_mae",
            metrics["difficulty_mae"],
            epoch + 1,
        )
        writer.add_scalar(
            "val/difficulty_rmse",
            metrics["difficulty_rmse"],
            epoch + 1,
        )
        writer.add_scalar(
            "val/difficulty_level_accuracy",
            metrics["difficulty_level_accuracy"],
            epoch + 1,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={metrics['val_loss']:.4f}, "
            f"topic_acc={metrics['topic_accuracy']:.4f}, "
            f"topic_f1={metrics['topic_macro_f1']:.4f}, "
            f"difficulty_mae={metrics['difficulty_mae']:.4f}, "
            f"difficulty_rmse={metrics['difficulty_rmse']:.4f}, "
            f"difficulty_level_acc={metrics['difficulty_level_accuracy']:.4f}"
        )

        improved = False
        if metrics["topic_macro_f1"] > best_topic_f1:
            improved = True
        elif (
            metrics["topic_macro_f1"] == best_topic_f1
            and metrics["val_loss"] < best_val_loss
        ):
            improved = True

        if improved:
            best_topic_f1 = metrics["topic_macro_f1"]
            best_val_loss = metrics["val_loss"]
            patience_counter = 0
            best_state_dict = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state_dict is None:
        best_state_dict = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
        }

    model_checkpoint = {
        "state_dict": best_state_dict,
        "model_name": args.model_name,
        "num_topics": len(topic_encoder.classes_),
        "num_difficulties": 1,
        "max_length": args.max_length,
    }

    torch.save(model_checkpoint, artifacts_dir / "model.pt")
    tokenizer.save_pretrained(artifacts_dir / "tokenizer")

    id_to_topic = {
        int(i): topic
        for i, topic in enumerate(topic_encoder.classes_.tolist())
    }
    topic_to_id = {topic: int(i) for i, topic in id_to_topic.items()}

    save_json(
        artifacts_dir / "label_mappings.json",
        {
            "topic_to_id": topic_to_id,
            "id_to_topic": {str(k): v for k, v in id_to_topic.items()},
            "difficulty_mode": "regression",
            "difficulty_range": [1.0, 5.0],
        },
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(artifacts_dir / "train_history.csv", index=False)

    save_json(
        artifacts_dir / "metadata.json",
        {
            "dataset_id": data_result.dataset_id,
            "dataset_columns": {
                "problem": data_result.problem_col,
                "topic": data_result.topic_col,
                "level": data_result.level_col,
            },
            "model_name": args.model_name,
            "max_length": args.max_length,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "epochs_ran": int(len(history)),
            "best_topic_macro_f1": float(best_topic_f1),
            "best_val_loss": float(best_val_loss),
            "device": str(device),
            "tb_log_dir": str(tb_log_dir),
        },
    )

    writer.flush()
    writer.close()

    print(f"Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    train()
