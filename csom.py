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


# ============================================================
# データ読み込みと前処理
# ============================================================


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


# ============================================================
# 特徴量抽出
# ============================================================


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


# ============================================================
# CSOM: 重みの初期化・学習・推論
# ============================================================


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


def calculate_weights_variance(
    weights: NDArray[np.complex128],
) -> float:
    """
    重み行列のベクトル間距離に基づく分散を計算する関数
    距離は (w_i - w_j) とその複素共役の内積で定義

    Args:
        weights: 重み行列 (dim, num_classes)

    Returns:
        全次元における距離の平均分散
    """
    dim, num_classes = weights.shape
    all_distances = []

    for d in range(dim):
        # 各次元のクラスベクトルを取得
        w_d = weights[d, :]  # (num_classes,)

        # 全ペアの距離を計算
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                diff = w_d[i] - w_d[j]
                # 距離 = (w_i - w_j) · (w_i - w_j)* = |w_i - w_j|²
                distance = np.real(diff * diff.conj())
                all_distances.append(distance)

    return float(np.var(all_distances))


# ============================================================
# Active Inference: 状態推定・行動選択・学習
# ============================================================


def infer_state(obs_idx, A, B_counts, prev_qs, last_action_idx):
    """
    観測から現在の状態確率(qs)を計算する (Perception)
    """
    # 尤度: P(o|s)
    likelihood = A[obs_idx, :].reshape(-1, 1)

    # 事前分布: P(s_t | s_t-1, u)
    if prev_qs is not None and last_action_idx is not None:
        # 選んだ行動に対応するB行列(正規化済み)を取得
        B_action = normalize_counts(B_counts[last_action_idx])
        prior = np.dot(B_action, prev_qs)
    else:
        # 最初は分からないので一様分布
        prior = np.ones((A.shape[1], 1)) / A.shape[1]

    # ベイズ更新: Posterior ∝ Likelihood * Prior
    log_qs = np.log(likelihood + 1e-16) + np.log(prior + 1e-16)
    qs = softmax(log_qs)
    return qs


def softmax(x):
    e_x = np.exp(x - np.max(x))  # オーバーフロー対策
    return e_x / e_x.sum(axis=0)


def calculate_efe(A, B, C, current_state_belief) -> float:
    """
    期待自由エネルギー (EFE) を計算する関数
    """
    predcted_state = np.dot(B, current_state_belief)
    predecited_obs = np.dot(A, predcted_state)

    eps = 1e-16
    H_A = -np.sum(A * np.log(A + eps), axis=0)

    ambiguity = np.dot(H_A, predcted_state)

    risk = np.sum(predecited_obs * (np.log(predecited_obs + eps) - np.log(C + eps)))

    G = risk + ambiguity
    print(f"G: {G[0]:.4f}")

    return G[0]


def update_parameters(
    a_counts, b_counts, obs_idx, current_qs, prev_qs, policy_idx, lr=1.0
):
    """
    パラメータ（経験値カウント）を更新する関数

    a_counts: A行列のカウント (No, Ns)
    b_counts: B行列のカウント (..., Ns, Ns)
    obs_idx: 観測されたデータのインデックス (int)
    current_qs: 現在の状態推定 (Ns, 1)
    prev_qs: 1つ前の状態推定 (Ns, 1) or None
    policy_idx: 実行したポリシーのインデックス (tuple)
    lr: 学習率 (1回あたりの重み)
    """

    # --- 1. A行列（尤度）の更新 ---
    # 観測(o) と 現在の状態(s) の結びつきを強化

    # 観測をOne-hotベクトル化
    num_obs = a_counts.shape[0]
    o_onehot = np.zeros((num_obs, 1))
    o_onehot[obs_idx] = 1.0

    # 外積 (No x 1) * (1 x Ns) = (No, Ns) を計算して加算
    da = lr * np.dot(o_onehot, current_qs.T)
    a_counts += da

    # --- 2. B行列（遷移）の更新 ---
    # 前の状態(s_tm1) -> 今の状態(s_t) の遷移を強化
    # 初回ステップ(prev_qsがない場合)は更新できないためスキップ

    if prev_qs is not None:
        # 外積 (Ns x 1) * (1 x Ns) = (Ns, Ns)
        # 行:現在の状態(Next), 列:前の状態(Prev)
        db = lr * np.dot(current_qs, prev_qs.T)

        # policy_idx (タプル) を使って、該当する行動のシートだけを更新
        b_counts[policy_idx] += db

    return a_counts, b_counts


def normalize_counts(counts):
    """
    カウント行列を確率行列（合計1.0）に変換するヘルパー関数
    """
    # 列方向(axis=-2)の合計で割る
    # A行列 (No, Ns) -> 各列(状態)ごとに正規化
    # B行列 (..., Ns, Ns) -> 各列(前の状態)ごとに正規化

    # ゼロ除算回避のための小さな値
    eps = 1e-16

    # sumのaxisは、最後の2次元のうち「列」方向なので -2
    # (行列の形状定義によりますが、通常 Active Inferenceでは column-stochastic です)
    # A: sum(axis=0), B: sum(axis=-2) ですが、
    # ここでは汎用的に書くため手動で指定します。

    if counts.ndim == 2:  # A行列の場合
        return counts / (counts.sum(axis=0, keepdims=True) + eps)
    else:  # B行列の場合
        return counts / (counts.sum(axis=-2, keepdims=True) + eps)


# ============================================================
# 可視化・保存
# ============================================================


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


def save_variance_plot(
    weights: NDArray[np.complex128],
    cfg: Config,
    output_dir: Path,
    filename: str = "CSOM_variance.png",
) -> None:
    """
    重みベクトル間の距離統計をプロットして保存する関数
    """
    filepath = output_dir / filename
    print(f"Saving {filepath}...")

    # 各次元ごとの統計を計算
    dim, num_classes = weights.shape
    mean_distances = np.zeros(dim, dtype=np.float64)
    variance_distances = np.zeros(dim, dtype=np.float64)

    for d in range(dim):
        w_d = weights[d, :]
        distances = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                diff = w_d[i] - w_d[j]
                distance = np.real(diff * diff.conj())
                distances.append(distance)

        distances_array = np.array(distances)
        mean_distances[d] = np.mean(distances_array)
        variance_distances[d] = np.var(distances_array)

    # 全体の分散
    overall_variance = calculate_weights_variance(weights)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 平均距離
    ax1.bar(range(dim), mean_distances)
    ax1.set_xlabel("Feature Dimension")
    ax1.set_ylabel("Mean Distance")
    ax1.set_title("Mean Distance Between Weight Vectors")
    ax1.grid(True, alpha=0.3)

    # 距離の分散
    ax2.bar(range(dim), variance_distances)
    ax2.set_xlabel("Feature Dimension")
    ax2.set_ylabel("Variance of Distances")
    ax2.set_title("Variance of Distances Between Weight Vectors")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

    # 統計情報を表示
    print(f"Overall variance: {overall_variance:.4f}")
    print(
        f"Mean distance - Mean: {mean_distances.mean():.4f}, Std: {mean_distances.std():.4f}"
    )
    print(
        f"Variance of distances - Mean: {variance_distances.mean():.4f}, Std: {variance_distances.std():.4f}"
    )


# ============================================================
# メイン実行関数
# ============================================================


def active_inference():
    C = np.array([0.4, 0.3, 0.2, 0.1, 0.0]).reshape(5, 1)
    qs = np.array([0.5, 0.5]).reshape(2, 1)

    a = np.array([[0.4, 0.0], [0.3, 0.1], [0.2, 0.2], [0.1, 0.3], [0.0, 0.4]])
    state_dims = [3] * 10
    b = np.zeros(state_dims + [2, 2], dtype=np.float64)
    for idx in np.ndindex(*state_dims):
        b[idx] = np.array([[0.5, 0.5], [0.5, 0.5]])

    prev_qs = None
    last_action = None

    cfg = Config()
    data, freqs = load_file(Config())

    # 10個の周波数成分を用いて特徴量抽出
    selected_freqs = np.linspace(0, cfg.freq_point - 1, cfg.freq_count, dtype=int)
    print(freqs[selected_freqs])

    print("-" * 60)
    print("Start Simulation Loop")
    print("-" * 60)

    for t in range(100):
        print(f"--- Step {t + 1} ---")

        if last_action is not None:
            for i, item in enumerate(last_action):
                selected_freqs[i] = min(
                    max(0, selected_freqs[i] + (item - 1) * 100), cfg.freq_point - 1
                )
        # 選択された周波数を表示
        print("Selected frequencies (GHz): ", freqs[selected_freqs])

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

        obs = calculate_weights_variance(weights)
        obs_idx = 0
        if obs > 3.36:
            obs_idx = 0
        elif obs > 3.13:
            obs_idx = 1
        elif obs > 2.93:
            obs_idx = 2
        elif obs > 2.73:
            obs_idx = 3
        else:
            obs_idx = 4

        A = normalize_counts(a)

        prev_qs_backup = qs.copy()
        qs = infer_state(obs_idx, A, b, prev_qs, last_action)

        a, b = update_parameters(a, b, obs_idx, qs, prev_qs, last_action, lr=1.0)

        prev_qs = qs.copy()

        best_efe = float("inf")
        best_action = None

        A = normalize_counts(a)

        for action_idx in np.ndindex(*state_dims):
            B = normalize_counts(b[action_idx])

            G = calculate_efe(A, B, C, qs)

            if G < best_efe:
                best_efe = G
                best_action = action_idx

        print(
            f"Time {t}: Obs={obs_idx} | Belief(s0)={qs[0, 0]:.2f} | Action={best_action} | EFE={best_efe:.2f}"
        )

        last_action = best_action

        # 結果保存
        output_dir = create_output_dir()
        save_class_map(class_map, cfg, output_dir, filename=f"step{t + 1}_map.png")
        save_weights_plot(weights, cfg, output_dir, filename=f"step{t + 1}_weights.png")


def csom() -> None:
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
    save_variance_plot(weights, cfg, output_dir)  # 追加


def main() -> None:
    active_inference()
    # csom()


if __name__ == "__main__":
    main()
