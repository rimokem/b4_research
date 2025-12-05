from __future__ import annotations

from typing import List, NamedTuple, Tuple, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray

# ガウス過程回帰用
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from tqdm import tqdm

matplotlib.use("Agg")


class Config(NamedTuple):
    """CSOMの設定パラメータ"""

    raw_file: str = "data/mine3.csv"

    # 測定パラメータ
    freq_point: int = 1601
    scale_x: int = 35
    scale_y: int = 35

    # SOM 学習パラメータ
    alpha: float = 0.4
    beta: float = 0.01
    epochs: int = 50
    num_classes: int = 8

    # 特徴量抽出パラメータ
    target_freq_count: int = 10
    window_size: int = 5

    # 能動的推論パラメータ
    n_measurements: int = 10  # 常に維持する計測点数
    n_iterations: int = 10  # 入れ替えを行う回数
    gpr_length_scale: float = 50.0
    gpr_noise_level: float = 0.01


# -----------------------------------------------------------------------------
#  既存の関数群 (前回と同じものは省略せずに記述)
# -----------------------------------------------------------------------------


def load_file(cfg: Config) -> NDArray[np.complex128]:
    """ファイルから複素振幅データを読み込む関数"""
    try:
        df = pd.read_csv(cfg.raw_file, header=None)
    except FileNotFoundError:
        print("Warning: File not found. Generating dummy data.")
        return np.random.rand(
            cfg.scale_x, cfg.scale_y, cfg.freq_point
        ) + 1j * np.random.rand(cfg.scale_x, cfg.scale_y, cfg.freq_point)

    df.columns = ["index_freq", "freq", "amp", "phase", "x", "y"]

    min_amp = df["amp"].min()
    max_amp = df["amp"].max()
    df["amp_norm"] = (df["amp"] - min_amp) / (max_amp - min_amp)
    complex_values = df["amp_norm"] * np.exp(1j * np.deg2rad(df["phase"]))

    max_y = df["y"].max()
    real_y = np.where(df["x"] % 2 != 0, max_y - df["y"], df["y"])

    data = np.zeros((cfg.scale_x, cfg.scale_y, cfg.freq_point), dtype=np.complex128)

    try:
        data[df["x"], real_y, df["index_freq"]] = complex_values
    except IndexError:
        pass

    return data


def extract_features(
    data: NDArray[np.complex128], cfg: Config
) -> NDArray[np.complex128]:
    L = cfg.window_size
    rows, cols, _ = data.shape
    out_rows = rows - (L - 1)
    out_cols = cols - (L - 1)

    feat_dim = 1 + 4 + (cfg.target_freq_count - 1)
    features = np.zeros((out_rows, out_cols, feat_dim), dtype=np.complex128)
    norm_const = L * L * cfg.target_freq_count

    for x in range(out_rows):
        for y in range(out_cols):
            block = data[x : x + L, y : y + L, :]

            # 平均
            features[x, y, 0] = np.sum(block) / norm_const
            # 空間相関
            features[x, y, 1] = calc_spatial_corr(data, x, y, 0, 0, cfg)
            features[x, y, 2] = calc_spatial_corr(data, x, y, 1, 0, cfg)
            features[x, y, 3] = calc_spatial_corr(data, x, y, 0, 1, cfg)
            features[x, y, 4] = calc_spatial_corr(data, x, y, 1, 1, cfg)
            # 周波数間相関
            for i in range(cfg.target_freq_count - 1):
                d1 = block[:, :, i]
                d2 = block[:, :, i + 1]
                features[x, y, 5 + i] = np.sum(d1 * d2.conj()) / (L * L)

    return features


def calc_spatial_corr(
    data: NDArray[np.complex128], x: int, y: int, dx: int, dy: int, cfg: Config
) -> np.complex128:
    L = cfg.window_size
    rows, cols, _ = data.shape
    tx, ty = x + dx, y + dy
    curr_x_end = min(x + L, rows - dx)
    curr_y_end = min(y + L, cols - dy)
    lx = curr_x_end - x
    ly = curr_y_end - y
    if lx <= 0 or ly <= 0:
        return 0j
    b1 = data[x : x + lx, y : y + ly, :]
    b2 = data[tx : tx + lx, ty : ty + ly, :]
    n_freq = data.shape[2]
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
    dots = vec.conj() @ weights
    norms = np.linalg.norm(vec) * np.linalg.norm(weights, axis=0) + 1e-12
    cos_sims = np.abs(dots) / norms
    logits = cos_sims / temperature
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
    features: NDArray[np.complex128], prev_weights: NDArray[np.complex128], cfg: Config
) -> Tuple[NDArray[np.complex128], NDArray[np.int_]]:
    rows, cols, _ = features.shape
    weights = prev_weights.copy()
    class_map = np.zeros((rows, cols), dtype=np.int_)

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

    return weights, class_map


# -----------------------------------------------------------------------------
#  GPR & Active Inference Classes
# -----------------------------------------------------------------------------


class ComplexGPR:
    def __init__(self, cfg: Config):
        kernel = C(1.0) * RBF(length_scale=cfg.gpr_length_scale) + WhiteKernel(
            noise_level=cfg.gpr_noise_level
        )

        self.models_real = [
            [
                GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
                for _ in range(cfg.scale_y)
            ]
            for _ in range(cfg.scale_x)
        ]
        self.models_imag = [
            [
                GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
                for _ in range(cfg.scale_y)
            ]
            for _ in range(cfg.scale_x)
        ]
        self.cfg = cfg

    def fit(self, full_data: NDArray[np.complex128], current_indices: List[int]):
        X = np.array(current_indices).reshape(-1, 1)
        print("Fitting GPR models...")
        for x in tqdm(range(self.cfg.scale_x)):
            for y in range(self.cfg.scale_y):
                y_vals = full_data[x, y, current_indices]
                self.models_real[x][y].fit(X, y_vals.real)
                self.models_imag[x][y].fit(X, y_vals.imag)

    def predict_full_spectrum(
        self, target_indices: NDArray[np.int_]
    ) -> Tuple[NDArray[np.complex128], NDArray[np.float64]]:
        X_target = target_indices.reshape(-1, 1)
        pred_mean = np.zeros(
            (self.cfg.scale_x, self.cfg.scale_y, len(target_indices)),
            dtype=np.complex128,
        )
        pred_var = np.zeros(
            (self.cfg.scale_x, self.cfg.scale_y, len(target_indices)), dtype=np.float64
        )

        for x in range(self.cfg.scale_x):
            for y in range(self.cfg.scale_y):
                mu_r, sigma_r = self.models_real[x][y].predict(
                    X_target, return_std=True
                )
                mu_i, sigma_i = self.models_imag[x][y].predict(
                    X_target, return_std=True
                )
                pred_mean[x, y, :] = mu_r + 1j * mu_i
                pred_var[x, y, :] = sigma_r**2 + sigma_i**2
        return pred_mean, pred_var

    def predict_at_candidate(
        self, candidate_idx: int
    ) -> Tuple[NDArray[np.complex128], NDArray[np.float64]]:
        X_cand = np.array([[candidate_idx]])
        means = np.zeros((self.cfg.scale_x, self.cfg.scale_y), dtype=np.complex128)
        vars_ = np.zeros((self.cfg.scale_x, self.cfg.scale_y), dtype=np.float64)

        for x in range(self.cfg.scale_x):
            for y in range(self.cfg.scale_y):
                mu_r, sigma_r = self.models_real[x][y].predict(X_cand, return_std=True)
                mu_i, sigma_i = self.models_imag[x][y].predict(X_cand, return_std=True)
                means[x, y] = mu_r[0] + 1j * mu_i[0]
                vars_[x, y] = sigma_r[0] ** 2 + sigma_i[0] ** 2
        return means, vars_


def calculate_entropy_map(
    features: NDArray[np.complex128], som_weights: NDArray[np.complex128], cfg: Config
) -> NDArray[np.float64]:
    """現在の特徴量とSOM重みに基づいて、各地点のエントロピー（迷い度）を計算"""
    rows, cols, _ = features.shape
    entropy_map = np.zeros((rows, cols))

    for x in range(rows):
        for y in range(cols):
            probs = calculate_distribution(features[x, y], som_weights)
            # Entropy = - sum(p * log(p))
            entropy_map[x, y] = -np.sum(probs * np.log(probs + 1e-10))

    return entropy_map


def select_best_frequency_to_add(
    gpr: ComplexGPR,
    entropy_map: NDArray[np.float64],
    candidate_indices: List[int],
    cfg: Config,
) -> int:
    """
    【追加用】期待自由エネルギー（情報利得）が最大になる周波数を選ぶ
    Score = Entropy(現在の迷い) * Variance(GPRの未知度)
    """
    best_score = -np.inf
    best_freq = candidate_indices[0]

    rows, cols = entropy_map.shape

    print("Selecting Best Frequency to ADD...")
    for cand_idx in tqdm(candidate_indices):
        _, variance_map = gpr.predict_at_candidate(cand_idx)

        # window_sizeによるサイズ違いを補正
        valid_var = variance_map[:rows, :cols]

        # 自由エネルギー項 (大きいほど良い = 情報が得られる)
        score = np.sum(entropy_map * valid_var)

        if score > best_score:
            best_score = score
            best_freq = cand_idx

    print(f"  -> Best Add Candidate: {best_freq} (Score: {best_score:.4f})")
    return best_freq


def select_worst_frequency_to_drop(
    full_data: NDArray[np.complex128], current_indices: List[int], cfg: Config
) -> int:
    """
    【削除用】既存のリストの中で、最も「情報量が少ない」周波数を選ぶ
    ここでは『空間分散 (Spatial Variance)』を自由エネルギー貢献度の代理指標とする。
    論理: マップ全体で値が平坦な周波数は、地雷と土壌の識別に寄与していない。
    """
    worst_score = np.inf
    worst_freq = current_indices[0]

    print("Selecting Worst Frequency to DROP...")
    for idx in current_indices:
        # その周波数における全地点のデータを取り出す
        # shape: (35, 35)
        amp_map = np.abs(full_data[:, :, idx])

        # 空間分散を計算 (値のバラつきが大きいほど、何か特徴を捉えている可能性が高い)
        spatial_variance = np.var(amp_map)

        # 分散が小さい(=情報量が小さい)ものを探す
        if spatial_variance < worst_score:
            worst_score = spatial_variance
            worst_freq = idx

    print(f"  -> Worst Drop Candidate: {worst_freq} (Spatial Var: {worst_score:.4f})")
    return worst_freq


def main() -> None:
    cfg = Config()
    full_data = load_file(cfg)
    print(f"Data loaded: {full_data.shape}")

    gpr = ComplexGPR(cfg)
    target_freq_indices = np.linspace(
        0, cfg.freq_point - 1, cfg.target_freq_count, dtype=int
    )

    # --- 1. 初期化: ランダムに10個選んでスタート ---
    all_indices = np.arange(cfg.freq_point)
    measured_indices = list(
        np.random.choice(all_indices, cfg.n_measurements, replace=False)
    )
    measured_indices.sort()

    print(f"Initial Frequencies: {measured_indices}")

    # 初期のGPR学習とSOM構築
    gpr.fit(full_data, measured_indices)
    pred_spectrum, _ = gpr.predict_full_spectrum(target_freq_indices)
    features = extract_features(pred_spectrum, cfg)

    # SOM重み初期化
    som_weights = init_weights(features.shape[2], cfg.num_classes)
    som_weights, class_map = train_csom(features, som_weights, cfg)

    # --- 2. 入れ替えループ (Optimization Loop) ---
    for step in range(cfg.n_iterations):
        print(f"\n=== Iteration {step + 1} / {cfg.n_iterations} ===")
        print(f"Current Set: {measured_indices}")

        # A. 現在の状態でのエントロピーマップ計算
        entropy_map = calculate_entropy_map(features, som_weights, cfg)

        # B. 追加候補の選択 (Active Selection)
        # 既に選ばれているもの以外から候補を作成
        candidates = np.setdiff1d(all_indices, measured_indices)
        np.random.shuffle(candidates)
        search_subset = candidates[:20]  # 高速化のため20個だけ評価

        freq_to_add = select_best_frequency_to_add(gpr, entropy_map, search_subset, cfg)

        # C. 削除候補の選択 (Least Contribution)
        freq_to_drop = select_worst_frequency_to_drop(full_data, measured_indices, cfg)

        # D. 入れ替え実行
        if freq_to_add != freq_to_drop:
            measured_indices.remove(freq_to_drop)
            measured_indices.append(freq_to_add)
            measured_indices.sort()
            print(f"Swapped: Drop {freq_to_drop} <-> Add {freq_to_add}")
        else:
            print("No swap needed (unlikely).")

        # E. 新しいセットでモデル更新
        gpr.fit(full_data, measured_indices)
        pred_spectrum, _ = gpr.predict_full_spectrum(target_freq_indices)
        features = extract_features(pred_spectrum, cfg)

        print("Retraining CSOM...")
        som_weights, class_map = train_csom(features, som_weights, cfg)

        # 可視化保存
        plt.figure(figsize=(6, 6))
        plt.imshow(class_map, cmap="jet", vmin=0, vmax=cfg.num_classes - 1)
        plt.title(f"Iter {step + 1}: Swapped {freq_to_drop}->{freq_to_add}")
        plt.savefig(f"iter_{step:02d}.png")
        plt.close()

    # --- 最終結果 ---
    print("\nOptimization Completed.")
    print("Final Frequency Set:", measured_indices)


if __name__ == "__main__":
    main()
