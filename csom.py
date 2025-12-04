from __future__ import annotations

from typing import Any, NamedTuple, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Config(NamedTuple):
    """CSOMの設定パラメータ"""

    # 観測データのファイルパス
    raw_file: str = "data/mine2.csv"

    # 測定パラメータ
    start_freq: float = 1.0  # GHz
    stop_freq: float = 11.0
    freq_point: int = 1601
    scale_x: int = 35
    scale_y: int = 35

    # SOM 学習パラメータ
    alpha: float = 0.4  # 学習率 (勝者)
    beta: float = 0.01  # 学習率 (近傍)
    epochs: int = 100  # 学習ループ回数
    num_classes: int = 8  # クラス数 (参照ベクトル数)

    # 特徴量抽出パラメータ
    freq_count: int = 10  # サンプリングする周波数の数
    window_size: int = 5  # 特徴抽出の窓サイズ (L)


def load_file(
    cfg: Config,
) -> NDArray[np.complex128]:
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

    return data


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
    # 1. 差分ベクトルを計算 (ブロードキャストを使用)
    # vecを (D,) から (D, 1) に変形して、(D, N) の weights から引く
    diff = vec[:, np.newaxis] - weights
    
    # 2. 各列ベクトル（axis=0）ごとのノルム（距離）を計算
    dists = np.linalg.norm(diff, axis=0)
    print("Distances:", dists)
    
    # 3. 距離が「最小」のインデックスを返す
    return int(np.argmin(dists))


def update_vecotor(
    w: NDArray[np.complex128], k: NDArray[np.complex128], coef: float
) -> NDArray[np.complex128]:
    diff = np.angle(w) - np.angle(k)
    abs_k = np.abs(k)

    new_amp = (1 - coef) * np.abs(w) + coef * abs_k * np.cos(diff)
    new_angle = np.angle(w) - coef * abs_k * np.sin(diff)

    return cast(NDArray[np.complex128], new_amp * np.exp(1j * new_angle))


def train_csom(features: NDArray[np.complex128], cfg: Config) -> Any:
    rows, cols, dim = features.shape
    weights = init_weights(dim, cfg.num_classes)

    # 可視化用セットアップ
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    frames = []

    for t in range(cfg.epochs):
        rate = (cfg.epochs - t) / cfg.epochs
        alpha_t = cfg.alpha * rate
        beta_t = cfg.beta * rate

        class_map = np.zeros((rows, cols), dtype=int)

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

        img = ax.imshow(class_map, cmap="Greys", vmin=0, vmax=cfg.num_classes - 1)
        frames.append([img])

    return fig, frames


def main() -> None:
    cfg = Config()
    data = load_file(Config())

    # 10個の周波数成分を用いて特徴量抽出
    selected_freqs = np.linspace(0, cfg.freq_point - 1, cfg.freq_count, dtype=int)
    print(selected_freqs)

    features = extract_features(
        data[:, :, selected_freqs],
        cfg,
    )

    fig, frames = train_csom(features, cfg)
    # 結果保存
    print("Saving video...")
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    try:
        ani.save("CSOM_numpy.mp4", writer="ffmpeg")
    except Exception as e:
        print(f"Video save failed (ffmpeg required): {e}")
        ani.save("CSOM_numpy.gif", writer="pillow")

    plt.savefig("CSOM_numpy_final.png")


if __name__ == "__main__":
    main()
