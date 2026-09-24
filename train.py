"""和文モールス信号CNNの学習スクリプト(Milestone 6)。

Validation Accuracyが最も高いエポックのモデルをmodels/best_model.ptへ保存し、
label_to_index・fixed_frames(前処理条件)をmodels/label_mapping.jsonへ保存する。
推論(predict.py)側はこのlabel_mapping.jsonを読み込むことで、学習時と同じ
前処理・クラス対応を再現できる。

毎エポックmodels/last_model.pt(モデル+オプティマイザ状態)と
train_history.jsonを更新するため、電源断等で学習が中断しても
--resumeで直前のエポックから再開できる。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from morse_cnn import create_model, model_from_config
from morse_dataset import DEFAULT_METADATA_CSV, build_dataloaders

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODELS_DIR = Path("models")


def evaluate(model: nn.Module, loader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer, device: str) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train(
    metadata_csv: str | Path = DEFAULT_METADATA_CSV,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    seed: int = 42,
    device: str | None = None,
    resume: bool = False,
    arch: str = "cnn",
) -> dict:
    """学習を実行し、最良モデル・label_mapping・学習履歴を保存する。

    resume=Trueの場合、models_dir内のチェックポイント(last_model.pt。
    無ければbest_model.pt)からモデル・オプティマイザ状態を復元し、
    epochsで指定した件数だけ追加で学習する。エポック番号はtrain_history.json
    に記録済みの続きから採番される。

    戻り値は best_model_path / label_mapping_path / history_path / best_val_acc を
    含むdict(テストや呼び出し元スクリプトから結果を参照しやすくするため)。
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "best_model.pt"
    last_model_path = models_dir / "last_model.pt"
    label_map_path = models_dir / "label_mapping.json"
    history_path = models_dir / "train_history.json"

    label_to_index: dict[str, int] | None = None
    fixed_frames: int | None = None
    best_val_acc = -1.0
    history: list[dict] = []
    resume_model_state = None
    resume_optimizer_state = None
    resume_model_config: dict | None = None
    start_epoch = 0

    if resume:
        if not label_map_path.exists():
            raise FileNotFoundError(f"再開に必要なlabel_mapping.jsonが見つかりません: {label_map_path}")
        with open(label_map_path, encoding="utf-8") as f:
            mapping = json.load(f)
        label_to_index = mapping["label_to_index"]
        fixed_frames = mapping["fixed_frames"]

        checkpoint_path = last_model_path if last_model_path.exists() else best_model_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"再開に必要なチェックポイントが見つかりません: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        resume_model_state = checkpoint["model_state_dict"]
        resume_optimizer_state = checkpoint.get("optimizer_state_dict")
        # 旧バージョンのチェックポイントにはmodel_configが無い場合があるため、
        # その場合はMorseCNNの現在のデフォルト構成を使う(architecture変更前の
        # チェックポイントから再開する場合は形状が一致しないため別途注意が必要)。
        resume_model_config = checkpoint.get("model_config")

        if best_model_path.exists():
            best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
            best_val_acc = best_checkpoint["val_acc"]

        if history_path.exists():
            with open(history_path, encoding="utf-8") as f:
                history = json.load(f)
        start_epoch = history[-1]["epoch"] if history else checkpoint["epoch"]

        print(
            f"epoch {start_epoch} から再開します"
            f"(読み込んだチェックポイント: {checkpoint_path.name}, これまでの最良val_acc={best_val_acc:.4f})"
        )

    train_loader, val_loader, _test_loader, label_to_index, fixed_frames = build_dataloaders(
        metadata_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        fixed_frames=fixed_frames,
        label_to_index=label_to_index,
    )

    if resume_model_config is not None:
        model = model_from_config(resume_model_config).to(device)
    else:
        model = create_model(arch, len(label_to_index)).to(device)
    if resume_model_state is not None:
        model.load_state_dict(resume_model_state)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)

    total_epochs = start_epoch + epochs
    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"epoch {epoch:02d}/{total_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # 電源断等で中断しても再開できるよう、毎エポック末尾で保存する
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model.config(),
                "num_classes": len(label_to_index),
                "fixed_frames": fixed_frames,
                "val_acc": val_acc,
                "epoch": epoch,
            },
            last_model_path,
        )
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config(),
                    "num_classes": len(label_to_index),
                    "fixed_frames": fixed_frames,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_model_path,
            )
            with open(label_map_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"label_to_index": label_to_index, "fixed_frames": fixed_frames},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    print(f"最良検証精度: {best_val_acc:.4f} (保存先: {best_model_path})")
    return {
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
        "label_map_path": label_map_path,
        "history_path": history_path,
        "best_val_acc": best_val_acc,
        "history": history,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="和文モールス信号CNNを学習する")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--arch", choices=["cnn", "crnn"], default="cnn", help="モデル構成(再開時は保存済みの構成を使う)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="既存のチェックポイントから追加でepochs分だけ学習を継続する")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        metadata_csv=args.metadata_csv,
        models_dir=args.models_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        arch=args.arch,
    )


if __name__ == "__main__":
    main()
