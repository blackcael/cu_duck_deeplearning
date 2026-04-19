#!/usr/bin/env python3
"""Utilities for visualizing DuckieDriveDataset samples with direction arrows."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_image(img: torch.Tensor) -> np.ndarray:
    """Convert tensor image [C,H,W] to numpy image for matplotlib."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float()
    else:
        img = torch.as_tensor(img).float()

    if img.ndim != 3:
        raise ValueError(f"Expected image shape [C,H,W], got {tuple(img.shape)}")

    channels, _, _ = img.shape
    if channels == 1:
        return img[0].numpy()
    if channels == 3:
        return img.permute(1, 2, 0).numpy()

    raise ValueError(f"Expected 1 or 3 channels, got {channels}")


def _label_to_unit_arrow(label: torch.Tensor) -> Tuple[float, float, str]:
    """Map dataset label to a unit arrow vector (dx, dy) and display text.

    Supported label formats:
    - shape [2]: interpreted as a coordinate-like pair.
    - scalar: interpreted as steering bin in [-22, 22].
    """
    label = torch.as_tensor(label).detach().cpu().float().flatten()

    if label.numel() == 2:
        dx = float(label[0].item())
        dy = float(label[1].item())
        mag = float(np.hypot(dx, dy))
        if mag < 1e-8:
            return 0.0, 0.0, f"pair=({dx:.3f}, {dy:.3f})"
        return dx / mag, dy / mag, f"pair=({dx:.3f}, {dy:.3f})"

    if label.numel() == 1:
        angle_bin = float(label.item())
        # Map [-22,22] steering bins to heading offset [-pi/2, pi/2].
        theta = (angle_bin / 22.0) * (np.pi / 2.0)
        dx = float(np.sin(theta))
        dy = float(np.cos(theta))
        return dx, dy, f"angle_bin={angle_bin:.1f}"

    raise ValueError(f"Expected label with 1 or 2 values, got shape {tuple(label.shape)}")


def _model_output_to_unit_arrow(model_output: torch.Tensor) -> Tuple[float, float, str]:
    """Map model output to a unit arrow vector (dx, dy) and display text.

    Supported output formats:
    - [45] or [1,45]: logits over steering classes -> argmax mapped to [-22,22]
    - scalar: steering bin in [-22,22]
    - [2]: coordinate-like pair
    """
    out = torch.as_tensor(model_output).detach().cpu().float()

    if out.ndim == 2 and out.shape[0] == 1:
        out = out.squeeze(0)

    if out.ndim == 1 and out.numel() == 45:
        pred_class = int(torch.argmax(out).item())
        angle_bin = float(pred_class - 22)
        return _label_to_unit_arrow(torch.tensor([angle_bin]))

    if out.ndim == 0 or (out.ndim == 1 and out.numel() in (1, 2)):
        return _label_to_unit_arrow(out)

    raise ValueError(f"Unsupported model output shape: {tuple(out.shape)}")


def play_dataset_with_arrow(
    dataset,
    model=None,
    model_name: Optional[str] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    step: int = 1,
    fps: float = 15.0,
    arrow_scale: float = 0.35,
    arrow_color: str = "lime",
    pred_arrow_color: str = "dodgerblue",
    model_device: Optional[torch.device] = None,
    repeat: bool = True,
):
    """Play a DuckieDriveDataset image stream with an overlaid direction arrow.

    Args:
        dataset: A DuckieDriveDataset-like object returning (image, label).
        model: Optional torch model. If provided, a second arrow is drawn for prediction.
        model_name: Optional model label to include in the animation title.
        start_idx: Starting index in the dataset.
        end_idx: End index (exclusive). Defaults to len(dataset).
        step: Step size through dataset indices.
        fps: Playback speed in frames per second.
        arrow_scale: Arrow size relative to min(image_width, image_height).
        arrow_color: Matplotlib color for ground-truth arrow.
        pred_arrow_color: Matplotlib color for model-prediction arrow.
        model_device: Optional torch device for model inference.
        repeat: Whether playback repeats.

    Returns:
        (fig, anim): Matplotlib figure and FuncAnimation. Keep a reference to
            `anim` alive while showing, e.g. `fig, anim = play_dataset_with_arrow(...)`.
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty")

    if end_idx is None:
        end_idx = n

    start_idx = max(0, start_idx)
    end_idx = min(end_idx, n)
    if start_idx >= end_idx:
        raise ValueError(f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}")

    indices = list(range(start_idx, end_idx, step))
    if not indices:
        raise ValueError("No indices to play after applying range/step")

    first_img, first_label = dataset[indices[0]]
    img_np = _to_numpy_image(first_img)
    dx, dy, label_text = _label_to_unit_arrow(first_label)
    pred_dx, pred_dy, pred_text = None, None, None

    def _predict_arrow(image_tensor: torch.Tensor) -> Tuple[float, float, str]:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pred_logits = model(image_tensor.unsqueeze(0).to(model_device))
        if was_training:
            model.train()
        return _model_output_to_unit_arrow(pred_logits)

    if model is not None:
        if model_device is None:
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")
        pred_dx, pred_dy, pred_text = _predict_arrow(first_img)

    fig, ax = plt.subplots(figsize=(7, 7))
    im_artist = ax.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)

    h, w = img_np.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    arrow_len = arrow_scale * min(w, h)

    (arrow_artist,) = ax.plot(
        [cx, cx + dx * arrow_len],
        [cy, cy - dy * arrow_len],
        color=arrow_color,
        linewidth=3,
    )
    marker_artist = ax.scatter([cx + dx * arrow_len], [cy - dy * arrow_len], c=arrow_color, s=30)
    pred_arrow_artist = None
    pred_marker_artist = None
    if model is not None:
        (pred_arrow_artist,) = ax.plot(
            [cx, cx + pred_dx * arrow_len],
            [cy, cy - pred_dy * arrow_len],
            color=pred_arrow_color,
            linewidth=3,
        )
        pred_marker_artist = ax.scatter(
            [cx + pred_dx * arrow_len],
            [cy - pred_dy * arrow_len],
            c=pred_arrow_color,
            s=30,
        )

    title_prefix = f"model: {model_name} | " if model_name else ""
    title_text = f"{title_prefix}idx={indices[0]} | gt: {label_text}"
    if model is not None:
        title_text += f" | pred: {pred_text}"
    ax.set_title(title_text)
    ax.set_axis_off()

    def _update(frame_i: int):
        idx = indices[frame_i]
        image, label = dataset[idx]
        frame_np = _to_numpy_image(image)
        dx_i, dy_i, text_i = _label_to_unit_arrow(label)
        pred_dx_i, pred_dy_i, pred_text_i = None, None, None
        if model is not None:
            pred_dx_i, pred_dy_i, pred_text_i = _predict_arrow(image)

        im_artist.set_data(frame_np)

        h_i, w_i = frame_np.shape[:2]
        cx_i, cy_i = w_i / 2.0, h_i / 2.0
        arrow_len_i = arrow_scale * min(w_i, h_i)

        x2 = cx_i + dx_i * arrow_len_i
        y2 = cy_i - dy_i * arrow_len_i

        arrow_artist.set_data([cx_i, x2], [cy_i, y2])
        marker_artist.set_offsets(np.array([[x2, y2]]))
        if model is not None:
            pred_x2 = cx_i + pred_dx_i * arrow_len_i
            pred_y2 = cy_i - pred_dy_i * arrow_len_i
            pred_arrow_artist.set_data([cx_i, pred_x2], [cy_i, pred_y2])
            pred_marker_artist.set_offsets(np.array([[pred_x2, pred_y2]]))
            ax.set_title(f"{title_prefix}idx={idx} | gt: {text_i} | pred: {pred_text_i}")
        else:
            ax.set_title(f"{title_prefix}idx={idx} | gt: {text_i}")

        artists = [im_artist, arrow_artist, marker_artist]
        if model is not None:
            artists.extend([pred_arrow_artist, pred_marker_artist])
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(indices),
        interval=1000.0 / fps,
        blit=False,
        repeat=repeat,
    )

    return fig, anim
