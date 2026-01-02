import numpy as np


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    x /= 32768.0
    return x


def float_to_pcm16(x: np.ndarray) -> bytes:
    y = np.clip(x, -1.0, 1.0)
    y = (y * 32767.0).astype(np.int16)
    return y.tobytes()


def resample_float(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    src_len = len(x)
    dst_len = int(round(src_len * (dst_sr / src_sr)))
    if dst_len <= 1:
        return np.zeros(1, dtype=np.float32)
    src_idx = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float32)
    left = np.floor(src_idx).astype(np.int32)
    right = np.minimum(left + 1, src_len - 1)
    frac = src_idx - left
    y = (1.0 - frac) * x[left] + frac * x[right]
    return y.astype(np.float32)

