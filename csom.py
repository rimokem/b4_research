from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Tuple, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray

matplotlib.use("Agg")


class Config(NamedTuple):
    """CSOMの設定パラメータ"""

    # 観測データのファイルパス
    raw_file: str = "data/mine3.csv"

    # 測定パラメータ
    start_freq: float = 1.0  # GHz
    stop_freq: float = 11.0
    freq_point: int = 1601
    scale_x: int = 35
    scale_y: int = 35

    # SOM 学習パラメータ
    alpha: float = 0.4  # 学習率 (勝者)
    beta: float = 0.01  # 学習率 (近傍)
    epochs: int = 50  # 学習ループ回数
    num_classes: int = 8  # クラス数 (参照ベクトル数)

    # 特徴量抽出パラメータ
    freq_count: int = 10  # サンプリングする周波数の数
    window_size: int = 5  # 特徴抽出の窓サイズ (L)

    # 能動的推論パラメータ
    max_measurements: int = 50  # 最大測定回数
    gpr_length_scale: float = 50.0  # GPRの長さスケール
    gpr_noise_level: float = 0.01  # GPRのノイズレベル


def load_file(
    cfg: Config,
) -> Tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """
    ファイルから複素振幅データを読み込む関数
    """
    df = pd.read_csv(cfg.raw_file, header=None)
    df.columns = ["index_freq", "freq", "amp", "phase", "x", "y"]

    # 正規化と複素数変換
    min_amp = df["amp"].min()
    max_amp = df["amp"].max()
    df["amp_norm"] = (df["amp"] - min_amp) / (max_amp - min_amp)
    complex_values = df["amp_norm"] * np.exp(1j * np.deg2rad(df["phase"]))

    # 往復走査の補正
    max_y = df["y"].max()
    real_y = np.where(df["x"] % 2 != 0, max_y - df["y"], df["y"])

    data = np.zeros((cfg.scale_x, cfg.scale_y, cfg.freq_point), dtype=np.complex128)
    data[df["x"], real_y, df["index_freq"]] = complex_values

    freq_mapping = (
        df[["index_freq", "freq"]].drop_duplicates().sort_values("index_freq")
    )
    freqs = freq_mapping["freq"].values

    return data, freqs


def extract_features(
    data: NDArray[np.complex128], cfg: Config
) -> NDArray[np.complex128]:
    """
    複素振幅データから特徴量を抽出する関数
    """
    L = cfg.window_size
    rows, cols, _ = data.shape
    out_rows = rows - (L - 1)
    out_cols = cols - (L - 1)

    features = np.zeros(
        (out_rows, out_cols, (cfg.freq_count - 1) + 5), dtype=np.complex128
    )

    norm_const = L * L * cfg.freq_count

    for x in range(out_rows):
        for y in range(out_cols):
            block = data[x : x + L, y : y + L, :]

            # 1. 平均値
            features[x, y, 0] = np.sum(block) / norm_const

            # 2-5. 空間相関(0,0), (1,0), (0,1), (1,1)
            features[x, y, 1] = calc_spatial_corr(data, x, y, 0, 0, cfg)
            features[x, y, 2] = calc_spatial_corr(data, x, y, 1, 0, cfg)
            features[x, y, 3] = calc_spatial_corr(data, x, y, 0, 1, cfg)
            features[x, y, 4] = calc_spatial_corr(data, x, y, 1, 1, cfg)

            # 6-. 周波数間相関
            for i in range(cfg.freq_count - 1):
                d1 = data[x : x + L, y : y + L, i]
                d2 = data[x : x + L, y : y + L, i + 1]

                features[x, y, 5 + i] = np.sum(d1 * d2.conj()) / L * L

    return features


def calc_spatial_corr(
    data: NDArray[np.complex128],
    x: int,
    y: int,
    dx: int,
    dy: int,
    cfg: Config,
) -> np.complex128:
    L = cfg.window_size
    rows, cols, _ = data.shape

    tx, ty = x + dx, y + dy

    curr_x_end = min(x + L, rows - dx)
    curr_y_end = min(y + L, cols - dy)

    lx = curr_x_end - x
    ly = curr_y_end - y

    b1 = data[x : x + lx, y : y + ly, :]
    b2 = data[tx : tx + lx, ty : ty + ly, :]

    n_freq = cfg.freq_count
    return cast(np.complex128, np.sum(b1 * b2.conj()) / (lx * ly * n_freq))


def init_weights(dim: int, n_classes: int) -> NDArray[np.complex128]:
    w_amp = np.random.rand(dim, n_classes)
    w_phase = 2 * np.pi * np.random.rand(dim, n_classes)

    return cast(NDArray[np.complex128], w_amp * np.exp(1j * w_phase))


def find_winner(vec: NDArray[np.complex128], weights: NDArray[np.complex128]) -> int:
    dots = vec.conj() @ weights
    norms = np.linalg.norm(vec) * np.linalg.norm(weights, axis=0) + 1e-12
    return int(np.argmax(np.abs(dots) / norms))


def calculate_distribution(
    vec: NDArray[np.complex128],
    weights: NDArray[np.complex128],
    temperature: float = 0.1,
) -> NDArray[np.float64]:
    """
    入力ベクトルと重み行列とのコサイン類似度に基づき、確率分布を計算して返します。

    Args:
        vec: 入力ベクトル (Complex)
        weights: 重み行列 (Complex)
        temperature: 分布の鋭さを調整するパラメータ (小さいほどTop1が強くなる)

    Returns:
        確率分布 (合計が1になる配列)
    """
    # 1. 内積 (Hermitian inner product)
    dots = vec.conj() @ weights

    # 2. ノルム計算
    norms = np.linalg.norm(vec) * np.linalg.norm(weights, axis=0) + 1e-12

    # 3. コサイン類似度 (絶対値をとって実数にする) [0.0 ~ 1.0]
    cos_sims = np.abs(dots) / norms

    # 4. 温度付きロジットの計算
    logits = cos_sims / temperature

    # 5. ソフトマックス関数 (数値安定化のため最大値を引く)
    # exp(x) / sum(exp(x)) の計算
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    return cast(NDArray[np.float64], probs)


def update_vecotor(
    w: NDArray[np.complex128], k: NDArray[np.complex128], coef: float
) -> NDArray[np.complex128]:
    diff = np.angle(w) - np.angle(k)
    abs_k = np.abs(k)

    new_amp = (1 - coef) * np.abs(w) + coef * abs_k * np.cos(diff)
    new_angle = np.angle(w) - coef * abs_k * np.sin(diff)

    return cast(NDArray[np.complex128], new_amp * np.exp(1j * new_angle))


def train_csom(
    features: NDArray[np.complex128],
    prev_weights: NDArray[np.complex128],
    prev_class_map: NDArray[np.int_],
    cfg: Config,
) -> Tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.complex128]]:
    rows, cols, _ = features.shape
    weights = prev_weights.copy()
    class_map = prev_class_map.copy()
    distribution_map = np.zeros((rows, cols, cfg.num_classes), dtype=np.float64)

    for t in range(cfg.epochs):
        rate = (cfg.epochs - t) / cfg.epochs
        alpha_t = cfg.alpha * rate
        beta_t = cfg.beta * rate

        for x in range(rows):
            for y in range(cols):
                k = features[x, y]

                winner = find_winner(k, weights)
                class_map[x, y] = winner

                weights[:, winner] = update_vecotor(weights[:, winner], k, alpha_t)

                left = (winner - 1) % cfg.num_classes
                right = (winner + 1) % cfg.num_classes

                weights[:, left] = update_vecotor(weights[:, left], k, beta_t)
                weights[:, right] = update_vecotor(weights[:, right], k, beta_t)

    for x in range(rows):
        for y in range(cols):
            k = features[x, y]
            distribution_map[x, y] = calculate_distribution(k, weights)

    return class_map, distribution_map, weights


def create_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outupt_dir = Path("results") / timestamp
    outupt_dir.mkdir(parents=True, exist_ok=True)
    return outupt_dir


def save_class_map(
    class_map: NDArray[np.int_],
    cfg: Config,
    output_dir: Path,
    filename: str = "CSOM_map.png",
) -> None:
    filepath = output_dir / filename
    print(f"Saving {filepath}...")
    fig, ax = plt.subplots(figsize=(6, 6))

    # weights plotと同じ色マッピングを使用
    colors = plt.cm.tab10(np.linspace(0, 1, cfg.num_classes))
    cmap = matplotlib.colors.ListedColormap(colors)

    ax.imshow(class_map, cmap=cmap, vmin=0, vmax=cfg.num_classes - 1)
    plt.savefig(filepath)
    plt.close()


def save_weights_plot(
    weights: NDArray[np.complex128],
    cfg: Config,
    output_dir: Path,
    filename: str = "CSOM_weights.png",
) -> None:
    filepath = output_dir / filename
    print(f"Saving {filepath}...")
    feature_dim = weights.shape[0]
    fig, axes = plt.subplots(
        2,
        (feature_dim + 1) // 2,
        figsize=(15, 8),
        subplot_kw={"projection": "polar"},
    )
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, cfg.num_classes))

    for dim_idx in range(feature_dim):
        ax = axes[dim_idx]

        weights_dim = weights[dim_idx, :]
        magnitudes = np.abs(weights_dim)
        phases = np.angle(weights_dim)

        for class_idx in range(cfg.num_classes):
            ax.plot(
                [0, phases[class_idx]],
                [0, magnitudes[class_idx]],
                c=colors[class_idx],
                linewidth=2,
                marker="o",
                markersize=8,
                label=f"Class {class_idx}",
            )

        ax.set_title(f"Dimension {dim_idx}", pad=20)
        ax.grid(True, alpha=0.3)

    # 余った軸を非表示
    for idx in range(feature_dim, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def main() -> None:
    cfg = Config()
    data, freqs = load_file(Config())

    # 10個の周波数成分を用いて特徴量抽出
    selected_freqs = np.linspace(0, cfg.freq_point - 1, cfg.freq_count, dtype=int)
    print(selected_freqs)

    features = extract_features(
        data[:, :, selected_freqs],
        cfg,
    )

    rows, cols, dim = features.shape
    initial_class_map = np.zeros((rows, cols), dtype=np.int_)
    initial_weights = init_weights(dim, cfg.num_classes)

    class_map, distribution_map, weights = train_csom(
        features, initial_weights, initial_class_map, cfg
    )
    # 結果保存

    output_dir = create_output_dir()
    save_class_map(class_map, cfg, output_dir)
    save_weights_plot(weights, cfg, output_dir)


if __name__ == "__main__":
    main()
