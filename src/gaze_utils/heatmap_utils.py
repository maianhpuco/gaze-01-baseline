#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import cv2


def _to_numpy(arr: torch.Tensor | np.ndarray | Iterable) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def create_static_heatmap(
    xy: torch.Tensor | np.ndarray,
    dwell: torch.Tensor | np.ndarray,
    *,
    h: int = 224,
    w: int = 224,
    threshold: float | None = None,
) -> np.ndarray:
    xy_np = _to_numpy(xy)
    dwell_np = _to_numpy(dwell).astype(np.float32)

    heatmap = np.zeros((h, w), dtype=np.float32)
    if xy_np.size > 0 and dwell_np.size > 0:
        xs = np.clip(np.round(xy_np[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(xy_np[:, 1]).astype(int), 0, h - 1)
        dwell_np = dwell_np[: len(xs)]
        for x, y, d in zip(xs, ys, dwell_np):
            heatmap[y, x] += d
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
    if threshold is not None:
        heatmap = (heatmap >= threshold).astype(np.float32)
    return heatmap.astype(np.float32)


def create_temporal_heatmaps(
    xy: torch.Tensor | np.ndarray,
    dwell: torch.Tensor | np.ndarray,
    *,
    h: int = 224,
    w: int = 224,
    max_seq: int = 10,
) -> np.ndarray:
    xy_np = _to_numpy(xy)
    dwell_np = _to_numpy(dwell).astype(np.float32)
    heatmaps = []
    cumulative = np.zeros((h, w), dtype=np.float32)

    if xy_np.size > 0 and dwell_np.size > 0:
        xs = np.clip(np.round(xy_np[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(xy_np[:, 1]).astype(int), 0, h - 1)
        dwell_np = dwell_np[: len(xs)]
        for x, y, d in zip(xs[:max_seq], ys[:max_seq], dwell_np[:max_seq]):
            cumulative[y, x] += d
            if cumulative.max() > 0:
                norm = cumulative / cumulative.max()
            else:
                norm = cumulative
            heatmaps.append(norm.astype(np.float32))

    while len(heatmaps) < max_seq:
        heatmaps.append(np.zeros((h, w), dtype=np.float32))
    return np.stack(heatmaps[:max_seq], axis=0)


def create_gradia_heatmap(
    xy: torch.Tensor | np.ndarray,
    dwell: torch.Tensor | np.ndarray,
    *,
    h: int = 224,
    w: int = 224,
    out_size: Tuple[int, int] = (7, 7),
) -> np.ndarray:
    xy_np = _to_numpy(xy)
    dwell_np = _to_numpy(dwell).astype(np.float32)
    heatmap = np.zeros((h, w), dtype=np.float32)

    if xy_np.size > 0 and dwell_np.size > 0:
        xs = np.clip(np.round(xy_np[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(xy_np[:, 1]).astype(int), 0, h - 1)
        dwell_np = dwell_np[: len(xs)]
        for x, y, d in zip(xs, ys, dwell_np):
            heatmap[y, x] += d
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

    resized = cv2.resize(heatmap, out_size, interpolation=cv2.INTER_AREA)
    resized = cv2.GaussianBlur(resized, (3, 3), 0)
    if resized.max() > resized.min():
        resized = (resized - resized.min()) / (resized.max() - resized.min())
    return resized.astype(np.float32)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

