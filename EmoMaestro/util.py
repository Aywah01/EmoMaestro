import torch
from torch import Tensor
import numpy as np

TEXT_W = 0.3

def fuse(text_vad, face_vad, text_w: float = TEXT_W):
    if not text_vad:
        return face_vad

    if not face_vad:
        return text_vad

    vad = (
        text_vad[0] * text_w +  face_vad[0] * (1 - text_w),
        text_vad[1] * text_w +  face_vad[1] * (1 - text_w),
        text_vad[2] * text_w +  face_vad[2] * (1 - text_w),
    )

    return vad

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vec_softmax(t: Tensor) -> list[float]:

    vec = [float(el) for el in t[0]]

    vec_max = np.max(vec)

    reduce_vec = [val - vec_max for val in vec]

    e_vec = np.exp(reduce_vec)

    return list(e_vec / e_vec.sum())

def feat_size(tot_len: int, stride: int, padding: int, kernel_len: int):
    return (tot_len - kernel_len + 2 * padding) // stride + 1