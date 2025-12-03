from __future__ import annotations

from typing import Any, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


class CSOMConfig(BaseModel):
    map_size: int
    num_iterations: int
    alpha: int
    beta: int
    window_size: int


class ImagingData(BaseModel):
    raw_data: Any
    freq_count: int
    height: int
    width: int

    @classmethod
    def from_array(cls, array: np.ndarray) -> ImagingData:
        freq_count, height, width = array.shape
        return cls(
            raw_data=array,
            freq_count=freq_count,
            height=height,
            width=width,
        )


class FeatureData(BaseModel):
    vectors: np.ndarray  # (N_samples, input_dim)
    positions: List[Tuple[int, int]]
    original_shape: Tuple[int, int]


def normalize_data(raw_data: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    複素振幅データの対数変換と正規化を行う関数
    """
    amp = np.abs(raw_data)
    phase = np.angle(raw_data)

    amp_db = 20 * np.log10(amp)

    min_val = np.min(amp_db)
    max_val = np.max(amp_db)

    amp_norm = (amp_db - min_val) / (max_val - min_val)

    result = amp_norm * np.exp(1j * phase)

    return cast(NDArray[np.complex128], result)


def calculate_local_features(window: np.ndarray, n_freq: int) -> np.ndarray:
    """
    1つのウィンドウから特徴ベクトル(1D)を計算する純粋関数
    """
    L = window.shape[0]  # window_size

    M = np.mean(window)

    # ラグ: (0,0), (1,0), (0,1), (1,1)
    lags = [(0, 0), (1, 0), (0, 1), (1, 1)]
    K_s = []
    for di, dj in lags:
        w_curr = window[0 : L - di, 0 : L - dj, :]
        w_shift = window[di:L, dj:L, :]
        # 複素共役との積の平均
        K_s.append(np.mean(np.conj(w_curr) * w_shift))

    K_f = []
    for n in range(n_freq - 1):
        z_n = window[:, :, n]
        z_n1 = window[:, :, n + 1]
        K_f.append(np.mean(np.conj(z_n) * z_n1))

    return np.concatenate(([M], K_s, K_f))


def extract_features(raw_data: np.ndarray, config: CSOMConfig) -> FeatureData:
    """
    画像全体から特徴ベクトルを抽出する関数
    """
    H, W, N_freq = raw_data.shape
    L = config.window_size
    offset = L // 2

    norm_data = normalize_data(raw_data)

    coords = [
        (y, x) for y in range(offset, H - offset) for x in range(offset, W - offset)
    ]

    def process_coord(coord: Tuple[int, int]) -> np.ndarray:
        y, x = coord
        window = norm_data[y - offset : y + offset + 1, x - offset : x + offset + 1, :]
        feature_vector = calculate_local_features(window, N_freq)
        return feature_vector

    vectors = np.array([process_coord(coord) for coord in coords])

    return FeatureData(
        vectors=vectors,
        positions=coords,
        original_shape=(H, W),
    )
