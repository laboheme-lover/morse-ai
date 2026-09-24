"""和文モールス信号1文字分類用のCNN(Milestone 6、Milestone 9後に容量拡大)。

Conv2D-BatchNorm-ReLU-MaxPoolを複数段重ねた後、AdaptiveAvgPoolで空間サイズを
1x1に潰し、全結合層2層で分類する。段数・チャネル数はchannelsで指定でき、
実データ検証で判明したモデル容量不足(少数クラスへの予測集中=モード崩壊)を
受け、初期構成(1->16->32->64、3段)から (1->32->64->128->256、4段) へ拡大した。
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_CHANNELS: tuple[int, ...] = (32, 64, 128, 256)
DEFAULT_HIDDEN_DIM = 256
# 容量拡大後の学習でtrain_acc 96%・val_acc 72%という過学習傾向が確認されたため、
# 0.3 -> 0.5 へ引き上げた。
DEFAULT_DROPOUT = 0.5


class MorseCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = tuple(channels)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev_channels = in_channels
        for out_channels in channels:
            layers += [
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def config(self) -> dict:
        """チェックポイント保存用のアーキテクチャ設定。

        predict.py/evaluate_model.py等が、学習時と全く同じ構造で
        MorseCNNを再構築できるようにするため、チェックポイントに含めて保存する。
        """
        return {
            "arch": "cnn",
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "channels": list(self.channels),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: dict) -> "MorseCNN":
        """config()で保存した設定からMorseCNNを再構築する。"""
        return cls(
            num_classes=config["num_classes"],
            in_channels=config.get("in_channels", 1),
            channels=tuple(config.get("channels", DEFAULT_CHANNELS)),
            hidden_dim=config.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            dropout=config.get("dropout", DEFAULT_DROPOUT),
        )


class MorseCRNN(nn.Module):
    """畳み込み + 双方向GRUによる分類モデル(時間方向の並びを保持して扱う)。

    MorseCNNは最後に時間方向を平均プールで潰すため、同じ短点・長点を並べ替えただけの
    符号(例: ス "---.-" と ア "--.--"、ル "-.--." と シ "--.-.")の区別が苦手で、
    合成テストのcleanでもル・ヒ・エの正解率が0%、実録音でも同じ組の取り違えが
    誤りの大半を占めた。畳み込みでは周波数方向を主に圧縮し、時間方向は
    4分の1までしか縮めず、双方向GRUで要素の順序を読み取ってから分類する。
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        rnn_hidden: int = 128,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = tuple(channels)
        self.rnn_hidden = rnn_hidden
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev_channels = in_channels
        for i, out_channels in enumerate(channels):
            # 最初の2段は時間・周波数とも半分、以降は周波数方向だけ半分にする
            pool = (2, 2) if i < 2 else (2, 1)
            layers += [
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            ]
            prev_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.rnn = nn.GRU(prev_channels, rnn_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 4, hidden_dim),  # 双方向 x (平均+最大) = 4 x rnn_hidden
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # [B, C, F', T']
        x = x.mean(dim=2)  # 周波数方向を平均 -> [B, C, T']
        x = x.transpose(1, 2)  # [B, T', C]
        out, _ = self.rnn(x)  # [B, T', 2*rnn_hidden]
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.classifier(pooled)

    def config(self) -> dict:
        return {
            "arch": "crnn",
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "channels": list(self.channels),
            "rnn_hidden": self.rnn_hidden,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: dict) -> "MorseCRNN":
        return cls(
            num_classes=config["num_classes"],
            in_channels=config.get("in_channels", 1),
            channels=tuple(config.get("channels", (32, 64, 128, 128))),
            rnn_hidden=config.get("rnn_hidden", 128),
            hidden_dim=config.get("hidden_dim", DEFAULT_HIDDEN_DIM),
            dropout=config.get("dropout", DEFAULT_DROPOUT),
        )


ARCHITECTURES = {"cnn": MorseCNN, "crnn": MorseCRNN}


def create_model(arch: str, num_classes: int) -> nn.Module:
    """アーキテクチャ名からモデルを新規に作る(既定の構成)。"""
    if arch not in ARCHITECTURES:
        raise ValueError(f"未対応のarchです: {arch!r} (選択肢: {sorted(ARCHITECTURES)})")
    return ARCHITECTURES[arch](num_classes=num_classes)


def model_from_config(config: dict) -> nn.Module:
    """チェックポイントに保存したconfigからモデルを再構築する。

    archキーが無い旧チェックポイントはMorseCNNとして扱う。
    """
    arch = config.get("arch", "cnn")
    if arch not in ARCHITECTURES:
        raise ValueError(f"未対応のarchです: {arch!r} (選択肢: {sorted(ARCHITECTURES)})")
    return ARCHITECTURES[arch].from_config(config)
