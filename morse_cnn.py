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

        # 【旧モデル】CNNだけの構成(v1〜v5)。現在は MorseCRNN を使っているが、
        # 比較用・旧チェックポイントの読み込み用に残している。
        #
        # 畳み込みブロックを channels の数だけ積む。1ブロック =
        #   Conv2d   : 3×3の小窓を画像全体にずらしながら当て、模様(縦線・横線など)を検出する
        #   BatchNorm: 値のばらつきを整えて学習を安定させる
        #   ReLU     : 負の値を0にする(これが無いと何層重ねても単純な掛け算1回分の表現力しかない)
        #   MaxPool2d(2): 2×2マスの最大値を1マスに → 縦横とも半分に縮める
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

        # 縦横すべてを平均して1マスにする。時間方向の並び順の情報はここで失われる
        # (並び順だけが違う ス/ア などを区別できなかった原因。MorseCRNN で解決した)。
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,1,64,T] → 畳み込み4段 → [B,256,4,T/16] → 縦横平均 → [B,256,1,1] → 分類器 → [B,クラス数]
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

        # ----- ①CNN部(模様を見つける係) -----
        # 入力の形 [B, 1, 64, T] (B=バッチの件数、1=チャンネル、64=周波数、T=時間フレーム数)
        # channels=(32, 64, 128, 128) の4ブロック。各ブロックで縦(周波数)は必ず半分にするが、
        # 横(時間)は最初の2ブロックだけ半分にし、残りは縮めない。時間方向の解像度を残して、
        # 後ろのGRUが短点・長点の「並び順」を読み取れるようにするため。
        #   ブロック1: [B,1,64,T]    → [B,32,32,T/2]
        #   ブロック2: [B,32,32,T/2] → [B,64,16,T/4]
        #   ブロック3: [B,64,16,T/4] → [B,128,8,T/4]
        #   ブロック4: [B,128,8,T/4] → [B,128,4,T/4]
        layers: list[nn.Module] = []
        prev_channels = in_channels
        for i, out_channels in enumerate(channels):
            # 最初の2段は時間・周波数とも半分、以降は周波数方向だけ半分にする
            # MaxPool2d((縦の縮小, 横の縮小)): (2, 2) = 縦横半分、(2, 1) = 縦だけ半分
            pool = (2, 2) if i < 2 else (2, 1)
            layers += [
                nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            ]
            prev_channels = out_channels
        self.features = nn.Sequential(*layers)

        # ----- ②双方向GRU部(並び順を読む係) -----
        # GRU は時系列を先頭から1コマずつ読み、「これまでに見た内容」を隠れ状態(128個の数値)に
        # 記憶しながら進むネットワーク。bidirectional=True で、前から読むGRUと後ろから読むGRUの
        # 2つを持ち、各コマで両者の結果を並べる(128×2=256個)。
        # batch_first=True: 入力を [バッチ, 時間, 特徴] の順で受け取る指定。
        self.rnn = nn.GRU(prev_channels, rnn_hidden, batch_first=True, bidirectional=True)
        # ----- 分類器(最終判定) -----
        # 512個の特徴 → 256個 → 63個(各文字のスコア)。Dropout は学習中だけ半分の値をランダムに0にし、
        # 特定の特徴に頼りすぎる(丸暗記する)のを防ぐ。推論時(model.eval())は無効になる。
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 4, hidden_dim),  # 双方向 x (平均+最大) = 4 x rnn_hidden
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力 x: [B, 1, 64, T]。例として T=424(学習時の固定長)の場合の形を書く。
        x = self.features(x)  # [B, C, F', T'] 例 [B, 128, 4, 106]
        x = x.mean(dim=2)  # 周波数方向を平均 -> [B, C, T'] 例 [B, 128, 106]
        # (周波数の情報は畳み込みで128チャンネルの特徴に織り込み済みなので、ここで1本にまとめてよい)
        x = x.transpose(1, 2)  # [B, T', C] GRUが読める「時間ごとの128次元ベクトルの列」に並べ替える
        out, _ = self.rnn(x)  # [B, T', 2*rnn_hidden] 例 [B, 106, 256] 各コマの前後双方向の読み取り結果
        # 時間方向を2通りにまとめて結合する:
        #   平均 = 全体の雰囲気、最大 = 一番強く反応した瞬間(決め手)。256+256 = 512個
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)  # [B, 512]
        # 63文字それぞれのスコア(logits)を返す。確率にするのは呼び出し側(softmax)。
        return self.classifier(pooled)  # [B, 63]

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


# モデル名 → クラスの対応表。チェックポイントには model_config["arch"] として名前を保存しておき、
# 読み込み時にこの表から同じ構造のモデルを作り直す(predict.py / evaluate_model.py / train.py の再開)。
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
