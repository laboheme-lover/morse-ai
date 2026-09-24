"""和文モールス信号の音声(WAV)を生成するモジュール。

Milestone 1では速度・周波数・音量を固定値としたシンプルな生成のみを扱った。
Milestone 2では以下のデータ拡張を追加する。

- 速度カテゴリ(slow/normal/fast)によるdot_durationのランダム化
- 周波数・音量のランダム化
- タイミングジッター(短点・長点・間隔の揺らぎ)
- ホワイトノイズ付加とSNR制御
- 生成条件のmetadata CSVへの記録
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from morse_table import char_to_units

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_RATE = 16000
DEFAULT_FREQUENCY_HZ = 700.0
DEFAULT_AMPLITUDE = 0.6
DEFAULT_DOT_DURATION = 0.10  # 短点の長さ[秒] (normal速度相当)

DASH_UNITS = 3  # 長点 = 短点の3倍
INTRA_CODE_GAP_UNITS = 1  # 同一符号内の短点・長点の間隔
INTER_CODE_GAP_UNITS = 3  # 濁点等、複数符号を連結する際の間隔(文字間隔相当)

RAMP_SECONDS = 0.005  # オン/オフ時のクリックノイズ防止用フェード時間(既定値)
# randomize時に抽出する立ち上がり/立ち下がりの長さの範囲。実録音の立ち上がりは
# 10-90%で約2.3ms(線形ランプ長で約2.9ms)と、既定の5msより急峻なため、これを
# 含む範囲とする。
RAMP_RANGE_SECONDS: tuple[float, float] = (0.002, 0.006)

# --- データ拡張パラメータ (Milestone 2) ---------------------------------

# 短点の長さの範囲[秒]。値は仕様書の速度カテゴリに対応する。
# 元の値(slow=0.13-0.17, normal=0.09-0.11, fast=0.05-0.07)は
# fast-normal間(0.07-0.09)・normal-slow間(0.11-0.13)に未学習の隙間があり、
# 実録音(短点長0.074-0.086秒)がちょうどその隙間に該当し認識精度が
# 大きく低下することが実データ検証で判明した。境界を接するよう拡張し、
# 0.05-0.17秒の範囲全体を隙間なくカバーする。
SPEED_RANGES: dict[str, tuple[float, float]] = {
    "slow": (0.12, 0.17),
    "normal": (0.08, 0.12),
    "fast": (0.05, 0.08),
}

# 実録音(モールス練習音源)のトーンは602Hzで、元の範囲(600-800Hz)の下端にあたる。
# 端に位置すると学習データが薄くなるため、下側へ拡張して範囲内に収める。
FREQUENCY_RANGE: tuple[float, float] = (500.0, 800.0)
AMPLITUDE_RANGE: tuple[float, float] = (0.4, 0.9)

# 短点・長点・間隔を基準値から±10%揺らす。短点:長点=1:3の比率に対して、
# どのdot_durationであっても短点の揺らぎ後最大値(dot*1.1)は長点の
# 揺らぎ後最小値(3*dot*0.9=2.7*dot)を常に下回るため、区別が崩れることはない。
TIMING_JITTER = 0.10

METADATA_FIELDS = [
    "filepath",
    "label",
    "morse_code",
    "speed",
    "dot_duration",
    "frequency_hz",
    "amplitude",
    "ramp_ms",
    "timing_jitter",
    "noise_type",
    "snr_db",
]


def _tone(
    duration_seconds: float,
    frequency_hz: float,
    amplitude: float,
    sample_rate: int,
    ramp_seconds: float = RAMP_SECONDS,
) -> np.ndarray:
    n_samples = max(int(round(duration_seconds * sample_rate)), 1)
    t = np.arange(n_samples) / sample_rate
    wave = amplitude * np.sin(2 * np.pi * frequency_hz * t)

    ramp_samples = min(int(round(ramp_seconds * sample_rate)), n_samples // 2)
    if ramp_samples > 0:
        ramp = np.linspace(0.0, 1.0, ramp_samples)
        wave[:ramp_samples] *= ramp
        wave[-ramp_samples:] *= ramp[::-1]
    return wave.astype(np.float32)


def _silence(duration_seconds: float, sample_rate: int) -> np.ndarray:
    n_samples = max(int(round(duration_seconds * sample_rate)), 0)
    return np.zeros(n_samples, dtype=np.float32)


def _jittered_duration(base_duration: float, timing_jitter: float, rng: np.random.Generator) -> float:
    if timing_jitter <= 0:
        return base_duration
    factor = 1.0 + rng.uniform(-timing_jitter, timing_jitter)
    return base_duration * factor


def code_to_waveform(
    code: str,
    dot_duration: float = DEFAULT_DOT_DURATION,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
    timing_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    ramp_seconds: float = RAMP_SECONDS,
) -> np.ndarray:
    """単一のモールス符号文字列(例 ".-")を波形(numpy配列)に変換する。"""
    if not 0.0 <= timing_jitter < 1.0:
        raise ValueError(f"timing_jitterは[0, 1)の範囲で指定してください: {timing_jitter!r}")
    rng = rng if rng is not None else np.random.default_rng()

    segments: list[np.ndarray] = []
    for i, symbol in enumerate(code):
        if i > 0:
            gap = _jittered_duration(dot_duration * INTRA_CODE_GAP_UNITS, timing_jitter, rng)
            segments.append(_silence(gap, sample_rate))
        if symbol == ".":
            duration = _jittered_duration(dot_duration, timing_jitter, rng)
        elif symbol == "-":
            duration = _jittered_duration(dot_duration * DASH_UNITS, timing_jitter, rng)
        else:
            raise ValueError(f"不正な符号記号です: {symbol!r}")
        segments.append(_tone(duration, frequency_hz, amplitude, sample_rate, ramp_seconds))
    return np.concatenate(segments)


def char_to_waveform(
    char: str,
    dot_duration: float = DEFAULT_DOT_DURATION,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
    timing_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    ramp_seconds: float = RAMP_SECONDS,
) -> np.ndarray:
    """1文字(濁音・半濁音の合成仮名を含む)を波形に変換する。"""
    rng = rng if rng is not None else np.random.default_rng()
    units = char_to_units(char)
    segments: list[np.ndarray] = []
    for i, code in enumerate(units):
        if i > 0:
            gap = _jittered_duration(dot_duration * INTER_CODE_GAP_UNITS, timing_jitter, rng)
            segments.append(_silence(gap, sample_rate))
        segments.append(
            code_to_waveform(
                code, dot_duration, frequency_hz, amplitude, sample_rate, timing_jitter, rng, ramp_seconds
            )
        )
    return np.concatenate(segments)


CHARACTER_GAP_UNITS = 7  # 文字と文字の間隔(語間隔相当)。文字内の最大間隔(3単位)
                          # より明確に長くし、連続音声認識(Milestone 9)で
                          # 文字境界を一意に判定できるようにする。


def sequence_to_waveform(
    text: str,
    dot_duration: float = DEFAULT_DOT_DURATION,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
    timing_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
    ramp_seconds: float = RAMP_SECONDS,
) -> np.ndarray:
    """複数文字を文字間隔(無音)で連結した連続音声の波形を生成する。"""
    rng = rng if rng is not None else np.random.default_rng()
    segments: list[np.ndarray] = []
    for i, char in enumerate(text):
        if i > 0:
            gap = _jittered_duration(dot_duration * CHARACTER_GAP_UNITS, timing_jitter, rng)
            segments.append(_silence(gap, sample_rate))
        segments.append(
            char_to_waveform(
                char, dot_duration, frequency_hz, amplitude, sample_rate, timing_jitter, rng, ramp_seconds
            )
        )
    return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)


def generate_sequence_wav(
    text: str,
    output_path: str | Path,
    dot_duration: float = DEFAULT_DOT_DURATION,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = SAMPLE_RATE,
    timing_jitter: float = 0.0,
    seed: int | None = None,
    ramp_seconds: float = RAMP_SECONDS,
) -> Path:
    """複数文字の連続モールス信号音声をWAVとして書き出す(連続音声認識のテスト・デモ用)。"""
    rng = np.random.default_rng(seed)
    waveform = sequence_to_waveform(
        text, dot_duration, frequency_hz, amplitude, sample_rate, timing_jitter, rng, ramp_seconds
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, sample_rate, subtype="PCM_16")
    return output_path


def add_white_noise(
    waveform: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """ホワイトノイズを付加する。

    信号電力はトーン区間(振幅が0でないサンプル)の平均パワーを基準とする。
    無音区間を含めた平均を基準にすると符号のマーク/スペース比によって
    同じSNR指定でも聞こえ方が変わってしまうため、トーン区間のみを用いる。
    """
    rng = rng if rng is not None else np.random.default_rng()
    active = waveform[waveform != 0]
    signal_power = float(np.mean(active**2)) if active.size > 0 else 1e-12

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return waveform + noise


def resolve_generation_params(
    rng: np.random.Generator,
    speed: str | None = None,
    dot_duration: float | None = None,
    frequency_hz: float | None = None,
    amplitude: float | None = None,
    randomize: bool = False,
    ramp_seconds: float | None = None,
) -> dict:
    """生成条件(speed, dot_duration, frequency_hz, amplitude, ramp_seconds)を決定する。

    明示的な指定があればそれを優先し、無ければ speed / randomize に応じて
    SPEED_RANGES・FREQUENCY_RANGE・AMPLITUDE_RANGE からランダムサンプリングする。
    """
    if dot_duration is not None:
        speed = speed or "custom"
    else:
        if speed is None:
            speed = "normal"
            dot_duration = DEFAULT_DOT_DURATION
        else:
            if speed not in SPEED_RANGES:
                raise ValueError(f"未対応のspeedです: {speed!r} (選択肢: {sorted(SPEED_RANGES)})")
            dot_duration = rng.uniform(*SPEED_RANGES[speed])

    if frequency_hz is None:
        frequency_hz = rng.uniform(*FREQUENCY_RANGE) if randomize else DEFAULT_FREQUENCY_HZ
    if amplitude is None:
        amplitude = rng.uniform(*AMPLITUDE_RANGE) if randomize else DEFAULT_AMPLITUDE

    if ramp_seconds is None:
        ramp_seconds = rng.uniform(*RAMP_RANGE_SECONDS) if randomize else RAMP_SECONDS

    return {
        "speed": speed,
        "dot_duration": dot_duration,
        "frequency_hz": frequency_hz,
        "amplitude": amplitude,
        "ramp_seconds": ramp_seconds,
    }


def generate_morse_wav(
    char: str,
    output_path: str | Path,
    speed: str | None = None,
    dot_duration: float | None = None,
    frequency_hz: float | None = None,
    amplitude: float | None = None,
    randomize: bool = False,
    timing_jitter: float = 0.0,
    noise_type: str = "clean",
    snr_db: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    seed: int | None = None,
    ramp_seconds: float | None = None,
) -> tuple[Path, dict]:
    """指定した1文字のモールス信号音声をWAVファイルとして書き出す。

    戻り値は (出力パス, 生成条件metadata辞書) のタプル。
    """
    if noise_type not in ("clean", "white"):
        raise ValueError(f"未対応のnoise_typeです: {noise_type!r} (選択肢: 'clean', 'white')")
    if noise_type == "white" and snr_db is None:
        raise ValueError("noise_type='white' の場合は snr_db を指定してください")

    rng = np.random.default_rng(seed)
    params = resolve_generation_params(rng, speed, dot_duration, frequency_hz, amplitude, randomize, ramp_seconds)

    waveform = char_to_waveform(
        char,
        dot_duration=params["dot_duration"],
        frequency_hz=params["frequency_hz"],
        amplitude=params["amplitude"],
        sample_rate=sample_rate,
        timing_jitter=timing_jitter,
        rng=rng,
        ramp_seconds=params["ramp_seconds"],
    )

    if noise_type == "white":
        waveform = add_white_noise(waveform, snr_db, rng)

    # ピークが±1.0を超える場合は一様スケーリングで正規化する。
    # np.clipによる波形の切り詰めは高調波歪みを生み、狙ったSNRを崩してしまうため、
    # 信号とノイズを同じ係数で縮小してSNR比を保ったまま±1.0以内に収める。
    peak = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
    if peak > 1.0:
        waveform = waveform / peak
    waveform = waveform.astype(np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, sample_rate, subtype="PCM_16")

    metadata = {
        "filepath": str(output_path).replace("\\", "/"),
        "label": char,
        "morse_code": "+".join(char_to_units(char)),
        "speed": params["speed"],
        "dot_duration": round(float(params["dot_duration"]), 6),
        "frequency_hz": round(float(params["frequency_hz"]), 3),
        "amplitude": round(float(params["amplitude"]), 4),
        "ramp_ms": round(float(params["ramp_seconds"]) * 1000, 2),
        "timing_jitter": timing_jitter,
        "noise_type": noise_type,
        "snr_db": snr_db if snr_db is not None else "",
    }
    return output_path, metadata


def append_metadata_row(csv_path: str | Path, row: dict) -> None:
    """生成条件を1行、metadata CSVへ追記する(無ければヘッダー付きで新規作成)。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="和文モールス信号の単一文字WAVを生成する")
    parser.add_argument("char", help="生成する文字(例: イ, ガ, 1, 、)")
    parser.add_argument("-o", "--output", default=None, help="出力WAVパス")
    parser.add_argument("--speed", choices=sorted(SPEED_RANGES), default=None, help="速度カテゴリ(未指定かつ--dot-duration未指定ならnormal固定値)")
    parser.add_argument("--dot-duration", type=float, default=None, help="短点の長さ[秒](指定時は--speedより優先)")
    parser.add_argument("--frequency", type=float, default=None, help="搬送波周波数[Hz](未指定かつ--randomizeならランダム)")
    parser.add_argument("--amplitude", type=float, default=None, help="振幅0-1(未指定かつ--randomizeならランダム)")
    parser.add_argument("--randomize", action="store_true", help="周波数・音量をFREQUENCY_RANGE/AMPLITUDE_RANGEからランダム化する")
    parser.add_argument("--jitter", type=float, default=0.0, help="タイミングジッター(例: 0.10 = ±10%%)")
    parser.add_argument("--noise", choices=["clean", "white"], default="clean", help="ノイズ種別")
    parser.add_argument("--snr", type=float, default=None, help="SNR[dB] (--noise white の場合必須)")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード(再現性確保用)")
    parser.add_argument("--metadata-csv", default=None, help="生成条件を追記するmetadata CSVパス")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output or f"data/clean/{args.char}.wav"
    path, metadata = generate_morse_wav(
        args.char,
        output_path,
        speed=args.speed,
        dot_duration=args.dot_duration,
        frequency_hz=args.frequency,
        amplitude=args.amplitude,
        randomize=args.randomize,
        timing_jitter=args.jitter,
        noise_type=args.noise,
        snr_db=args.snr,
        seed=args.seed,
    )
    print(f"生成しました: {path}")
    print(f"生成条件: {metadata}")
    if args.metadata_csv:
        append_metadata_row(args.metadata_csv, metadata)
        print(f"metadataを追記しました: {args.metadata_csv}")


if __name__ == "__main__":
    main()
