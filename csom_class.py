from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Tuple, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import entropy

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
    freq_count: int = 10
    window_size: int = 5

    # 能動的推論パラメータ
    n_measurements: int = 10  # 常に維持する計測点数
    n_iterations: int = 10  # 入れ替えを行う回数
    gpr_length_scale: float = 50.0
    gpr_noise_level: float = 0.01


class DataLoader:
    """データ読み込みクラス"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> NDArray[np.complex128]:
        """ファイルから複素振幅データを読み込む"""
        df = pd.read_csv(self.cfg.raw_file, header=None)

        df.columns = ["index_freq", "freq", "amp", "phase", "x", "y"]
        return self._process_dataframe(df)

    def _process_dataframe(self, df: pd.DataFrame) -> NDArray[np.complex128]:
        """DataFrameを複素数配列に変換"""
        min_amp = df["amp"].min()
        max_amp = df["amp"].max()
        df["amp_norm"] = (df["amp"] - min_amp) / (max_amp - min_amp)
        complex_values = df["amp_norm"] * np.exp(1j * np.deg2rad(df["phase"]))

        max_y = df["y"].max()
        real_y = np.where(df["x"] % 2 != 0, max_y - df["y"], df["y"])

        data = np.zeros(
            (self.cfg.scale_x, self.cfg.scale_y, self.cfg.freq_point),
            dtype=np.complex128,
        )

        data[df["x"], real_y, df["index_freq"]] = complex_values

        return data


class FeatureExtractor:
    """特徴量抽出クラス"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def extract(self, data: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """特徴量を抽出"""
        L = self.cfg.window_size
        rows, cols, _ = data.shape
        out_rows = rows - (L - 1)
        out_cols = cols - (L - 1)

        feat_dim = 1 + 4 + (self.cfg.freq_count - 1)
        features = np.zeros((out_rows, out_cols, feat_dim), dtype=np.complex128)
        norm_const = L * L * self.cfg.freq_count

        for x in range(out_rows):
            for y in range(out_cols):
                features[x, y] = self._extract_block_features(data, x, y, L, norm_const)

        return features

    def _extract_block_features(
        self,
        data: NDArray[np.complex128],
        x: int,
        y: int,
        L: int,
        norm_const: float,
    ) -> NDArray[np.complex128]:
        """単一ブロックの特徴量を抽出"""
        block = data[x : x + L, y : y + L, :]
        feat_dim = 1 + 4 + (self.cfg.freq_count - 1)
        features = np.zeros(feat_dim, dtype=np.complex128)

        # 平均
        features[0] = np.sum(block) / norm_const

        # 空間相関
        features[1] = self._calc_spatial_corr(data, x, y, 0, 0)
        features[2] = self._calc_spatial_corr(data, x, y, 1, 0)
        features[3] = self._calc_spatial_corr(data, x, y, 0, 1)
        features[4] = self._calc_spatial_corr(data, x, y, 1, 1)

        # 周波数間相関
        for i in range(self.cfg.freq_count - 1):
            d1 = block[:, :, i]
            d2 = block[:, :, i + 1]
            features[5 + i] = np.sum(d1 * d2.conj()) / (L * L)

        return features

    def _calc_spatial_corr(
        self, data: NDArray[np.complex128], x: int, y: int, dx: int, dy: int
    ) -> np.complex128:
        """空間相関を計算"""
        L = self.cfg.window_size
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


class CSOM:
    """Complex Self-Organizing Mapクラス"""

    def __init__(self, cfg: Config, feature_dim: int):
        self.cfg = cfg
        self.feature_dim = feature_dim
        self.weights = self._init_weights()

    def _init_weights(self) -> NDArray[np.complex128]:
        """重みを初期化"""
        w_amp = np.random.rand(self.feature_dim, self.cfg.num_classes)
        w_phase = 2 * np.pi * np.random.rand(self.feature_dim, self.cfg.num_classes)
        return cast(NDArray[np.complex128], w_amp * np.exp(1j * w_phase))

    def find_winner(self, vec: NDArray[np.complex128]) -> int:
        """勝者ユニットを見つける"""
        dots = vec.conj() @ self.weights
        norms = np.linalg.norm(vec) * np.linalg.norm(self.weights, axis=0) + 1e-12
        return int(np.argmax(np.abs(dots) / norms))

    def calculate_distribution(
        self, vec: NDArray[np.complex128], temperature: float = 0.1
    ) -> NDArray[np.float64]:
        """クラス分布を計算"""
        dots = vec.conj() @ self.weights
        norms = np.linalg.norm(vec) * np.linalg.norm(self.weights, axis=0) + 1e-12
        cos_sims = np.abs(dots) / norms
        logits = cos_sims / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return cast(NDArray[np.float64], probs)

    def detect_salient_targets(
        self, features: NDArray[np.complex128], top_k: int = 10
    ) -> list[Tuple[int, int]]:
        """顕著なターゲットを検出"""
        rows, cols, _ = features.shape

        # 全地点の分布を計算
        all_distributions = np.zeros((rows, cols, self.cfg.num_classes))
        for x in range(rows):
            for y in range(cols):
                vec = features[x, y]
                all_distributions[x, y] = self.calculate_distribution(
                    vec, temperature=0.1
                )

        # 平均分布を計算
        mean_distribution = np.mean(
            all_distributions.reshape(-1, self.cfg.num_classes), axis=0
        )

        # 各地点のKLダイバージェンスを計算
        kl_divergences = []
        for x in range(rows):
            for y in range(cols):
                kl_div = entropy(all_distributions[x, y], qk=mean_distribution)
                kl_divergences.append((kl_div, x, y))

        # KLダイバージェンスが大きい順にソート
        kl_divergences.sort(reverse=True, key=lambda item: item[0])

        # 上位top_k個の座標を返す
        salient_targets = [(x, y) for _, x, y in kl_divergences[:top_k]]

        return salient_targets

    def _update_vector(
        self, w: NDArray[np.complex128], k: NDArray[np.complex128], coef: float
    ) -> NDArray[np.complex128]:
        """重みベクトルを更新"""
        diff = np.angle(w) - np.angle(k)
        abs_k = np.abs(k)
        new_amp = (1 - coef) * np.abs(w) + coef * abs_k * np.cos(diff)
        new_angle = np.angle(w) - coef * abs_k * np.sin(diff)
        return cast(NDArray[np.complex128], new_amp * np.exp(1j * new_angle))

    def train(
        self, features: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], NDArray[np.int_]]:
        """CSOMを学習"""
        rows, cols, _ = features.shape
        class_map = np.zeros((rows, cols), dtype=np.int_)

        for t in range(self.cfg.epochs):
            rate = (self.cfg.epochs - t) / self.cfg.epochs
            alpha_t = self.cfg.alpha * rate
            beta_t = self.cfg.beta * rate

            for x in range(rows):
                for y in range(cols):
                    k = features[x, y]
                    winner = self.find_winner(k)
                    class_map[x, y] = winner

                    # 勝者と隣接ノードを更新
                    self.weights[:, winner] = self._update_vector(
                        self.weights[:, winner], k, alpha_t
                    )
                    left = (winner - 1) % self.cfg.num_classes
                    right = (winner + 1) % self.cfg.num_classes
                    self.weights[:, left] = self._update_vector(
                        self.weights[:, left], k, beta_t
                    )
                    self.weights[:, right] = self._update_vector(
                        self.weights[:, right], k, beta_t
                    )

        return self.weights, class_map


class FrequencyOptimizer:
    def __init__(self, cfg: Config, all_data: NDArray[np.complex128]):
        self.cfg = cfg
        self.all_data = all_data
        self.total_freqs = all_data.shape[2]

    def calculate_efe(
        self,
        current_freqs: NDArray[np.int_],
        target_coords: list[Tuple[int, int]],
        csom: CSOM,
        target_class: int = None,
    ) -> float:
        """
        期待自由エネルギー(EFE)を計算
        G = Risk + Ambiguity
        """
        slice_data = self.all_data[:, :, current_freqs]

        features = FeatureExtractor(self.cfg).extract(slice_data)

        total_efe = 0.0

        for x, y in target_coords:
            if x >= features.shape[0] or y >= features.shape[1]:
                continue
            vec = features[x, y]

            probs = csom.calculate_distribution(vec, temperature=0.1)

            ambiguity = entropy(probs)

            risk = 0.0

            if target_class is not None:
                epsilon = 1e-10
                preffed_dist = np.full_like(probs, epsilon)
                preffed_dist[target_class] = 1.0 - (epsilon * (len(probs) - 1))

                risk = entropy(probs, qk=preffed_dist)

            total_efe += risk + ambiguity

            return total_efe

        def optimize(
            self,
            initial_freqs: NDArray[np.int_],
            csom: CSOM,
            target_coords: list[Tuple[int, int]],
            target_class: int = None,
            n_trials: int = 20,
        ) -> NDArray[np.int_]:
            """周波数選択を最適化"""
            current_freqs = initial_freqs.copy()
            current_efe = self.calculate_efe(
                current_freqs, target_coords, csom, target_class
            )

            print(f"Initial EFE: {current_efe:.4f}")

            for i in range(n_trials):
                proposal_freqs = current_freqs.copy()
                idx_to_change = np.random.randint(0, len(proposal_freqs))
                new_freq_val = np.random.randint(0, self.total_freqs)

                while new_freq_val in proposal_freqs:
                    new_freq_val = np.random.randint(0, self.total_freqs)

                proposal_freqs[idx_to_change] = new_freq_val
                proposal_freqs.sort()

                propsal_efe = self.calculate_efe(
                    proposal_freqs, target_coords, csom, target_class
                )

                if propsal_efe < current_efe:
                    print(
                        f"Trial {i + 1}: EFE improved from {current_efe:.4f} to {propsal_efe:.4f}"
                    )
                    current_efe = propsal_efe
                    current_freqs = proposal_freqs

            return current_freqs


class Visualizer:
    """可視化クラス"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.output_dir = self._create_output_dir()

    def _create_output_dir(self) -> Path:
        """タイムスタンプ付き出力ディレクトリを作成"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = Path("results") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_class_map(
        self, class_map: NDArray[np.int_], filename: str = "CSOM_numpy_final.png"
    ) -> None:
        """クラスマップを保存"""
        filepath = self.output_dir / filename
        print(f"Saving {filepath}...")
        fig, ax = plt.subplots(figsize=(6, 6))

        # weights plotと同じ色マッピングを使用
        colors = plt.cm.tab10(np.linspace(0, 1, self.cfg.num_classes))
        cmap = matplotlib.colors.ListedColormap(colors)

        ax.imshow(class_map, cmap=cmap, vmin=0, vmax=self.cfg.num_classes - 1)
        plt.savefig(filepath)
        plt.close()

    def save_distribution_plot(
        self,
        features: NDArray[np.complex128],
        x: int,
        y: int,
        csom: CSOM,
        step: int,
    ) -> None:
        """クラス分布をプロット"""
        target_vec = features[x, y]
        probs = csom.calculate_distribution(target_vec, temperature=0.1)

        filepath = self.output_dir / f"dist_{x}_{y}_iter_{step:02d}.png"
        plt.figure(figsize=(6, 4))
        plt.bar(range(self.cfg.num_classes), probs, color="skyblue", edgecolor="black")
        plt.xlabel("Class ID")
        plt.ylabel("Probability")
        plt.title(f"Class Distribution at ({x}, {y}) - Iter {step:02d}")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(range(self.cfg.num_classes))
        plt.savefig(filepath)
        plt.close()

    def save_weights_complex_plot(
        self, csom: CSOM, filename: str = "weights_complex_space.png"
    ) -> None:
        """重みベクトルを複素平面上にプロット"""
        filepath = self.output_dir / filename
        print(f"Saving {filepath}...")

        fig, axes = plt.subplots(2, (csom.feature_dim + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        for dim_idx in range(csom.feature_dim):
            ax = axes[dim_idx]

            # 各クラスの重みを複素平面上にプロット
            weights_dim = csom.weights[dim_idx, :]
            real_parts = np.real(weights_dim)
            imag_parts = np.imag(weights_dim)

            colors = plt.cm.tab10(np.linspace(0, 1, self.cfg.num_classes))

            for class_idx in range(self.cfg.num_classes):
                ax.scatter(
                    real_parts[class_idx],
                    imag_parts[class_idx],
                    c=[colors[class_idx]],
                    s=100,
                    label=f"Class {class_idx}",
                    edgecolors="black",
                    linewidths=1.5,
                )
                # 原点からの矢印を描画
                ax.arrow(
                    0,
                    0,
                    real_parts[class_idx],
                    imag_parts[class_idx],
                    head_width=0.02,
                    head_length=0.03,
                    fc=colors[class_idx],
                    ec=colors[class_idx],
                    alpha=0.5,
                )

            ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")
            ax.set_title(f"Dimension {dim_idx}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        # 余った軸を非表示
        for idx in range(csom.feature_dim, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

    def save_weights_polar_plot(
        self, csom: CSOM, filename: str = "weights_polar_space.png"
    ) -> None:
        """重みベクトルを極座標系でプロット"""
        filepath = self.output_dir / filename
        print(f"Saving {filepath}...")

        fig, axes = plt.subplots(
            2,
            (csom.feature_dim + 1) // 2,
            figsize=(15, 8),
            subplot_kw={"projection": "polar"},
        )
        axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, self.cfg.num_classes))

        for dim_idx in range(csom.feature_dim):
            ax = axes[dim_idx]

            weights_dim = csom.weights[dim_idx, :]
            magnitudes = np.abs(weights_dim)
            phases = np.angle(weights_dim)

            for class_idx in range(self.cfg.num_classes):
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
        for idx in range(csom.feature_dim, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()


def main() -> None:
    cfg = Config()

    # データ読み込み
    loader = DataLoader(cfg)
    data = loader.load()

    # 周波数選択
    # selected_freqs = np.linspace(0, cfg.freq_point - 1, cfg.freq_count, dtype=int)
    selected_freqs = np.linspace(400, 800, cfg.freq_count, dtype=int)
    print(selected_freqs)

    # 特徴量抽出
    extractor = FeatureExtractor(cfg)
    features = extractor.extract(data[:, :, selected_freqs])

    # CSOM学習
    _, _, dim = features.shape
    csom = CSOM(cfg, dim)
    weights, class_map = csom.train(features)
    print(csom.detect_salient_targets(features, top_k=100))

    # 可視化
    visualizer = Visualizer(cfg)
    visualizer.save_class_map(class_map)
    visualizer.save_distribution_plot(features, 17, 20, csom, 0)
    visualizer.save_weights_complex_plot(csom)
    visualizer.save_weights_polar_plot(csom)


if __name__ == "__main__":
    main()
