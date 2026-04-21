#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from utils.DuckieDriveDataset import DuckieDriveDataset
from utils.model_utils import save_checkpoint
from utils.plotting import plot_training

INPUT_SIZE_H = 32
INPUT_SIZE_W = 32

OUTPUT_LAYER_SIZE = 45
DEF_DATA_DIR = "data/"


class ALVINNITA(nn.Module):
    """Temporal ALVINN with transformer decoder over the last T frames.

    Input shapes:
      - [B, T, C, H, W] preferred
      - [B, C, H, W] compatibility mode (treated as T=1)
    """

    def __init__(
        self,
        imagesize_hw=(INPUT_SIZE_H, INPUT_SIZE_W),
        color_channels=1,
        history_frames=6,
        d_model=128,
        nhead=4,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        output_size=OUTPUT_LAYER_SIZE,
    ):
        super().__init__()
        self.name = "alvinnita"
        self.color_channels = color_channels
        self.history_frames = history_frames

        h, w = imagesize_hw
        self.input_area = h * w
        self.input_dim = color_channels * self.input_area

        self.frame_encoder = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.position_embedding = nn.Parameter(torch.randn(history_frames, d_model) * 0.02)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W] or [B,C,H,W], got {tuple(x.shape)}")

        b, t, c, h, w = x.shape

        if c != self.color_channels:
            raise ValueError(f"Expected {self.color_channels} channels, got {c}")
        if h * w != self.input_area:
            raise ValueError(f"Expected H*W={self.input_area}, got H*W={h*w}")
        if t > self.history_frames:
            raise ValueError(
                f"Input T={t} exceeds history_frames={self.history_frames}; "
                "increase history_frames or pass fewer frames."
            )

        x = x.reshape(b, t, -1)
        memory = self.frame_encoder(x)

        # Right-align positional embeddings so shorter sequences represent the latest frames.
        pos = self.position_embedding[-t:].unsqueeze(0)
        memory = memory + pos

        query = self.query_token.expand(b, 1, -1)
        decoded = self.decoder(tgt=query, memory=memory)
        return self.head(decoded[:, 0, :])

    def get_name(self):
        return self.name

    def set_name(self, new_name):
        self.name = new_name


class TemporalWindowDataset(Dataset):
    """Wraps a frame dataset to return sliding temporal windows."""

    def __init__(self, base_dataset, history_frames):
        super().__init__()
        if history_frames <= 0:
            raise ValueError("history_frames must be > 0")
        self.base_dataset = base_dataset
        self.history_frames = history_frames
        self.num_windows = max(0, len(base_dataset) - history_frames + 1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(f"idx={idx} out of bounds for {self.num_windows} windows")
        frames = []
        target_angle = None
        for i in range(idx, idx + self.history_frames):
            img, angle = self.base_dataset[i]
            frames.append(img)
            target_angle = angle
        return torch.stack(frames, dim=0), target_angle


def _to_numpy_image(img: torch.Tensor) -> np.ndarray:
    img = torch.as_tensor(img).detach().cpu().float()
    if img.ndim != 3:
        raise ValueError(f"Expected image shape [C,H,W], got {tuple(img.shape)}")
    channels, _, _ = img.shape
    if channels == 1:
        return img[0].numpy()
    if channels == 3:
        return img.permute(1, 2, 0).numpy()
    raise ValueError(f"Expected 1 or 3 channels, got {channels}")


def _label_to_unit_arrow(label: torch.Tensor):
    label = torch.as_tensor(label).detach().cpu().float().flatten()
    if label.numel() != 1:
        raise ValueError(f"Expected scalar steering bin label, got shape {tuple(label.shape)}")
    angle_bin = float(label.item())
    theta = (angle_bin / 22.0) * (np.pi / 2.0)
    dx = float(np.sin(theta))
    dy = float(np.cos(theta))
    return dx, dy, f"angle_bin={angle_bin:.1f}"


def _angle_to_class(truth_angle):
    return torch.clamp(torch.round(truth_angle + 22), 0, 44).long().view(-1)


def _logits_to_angle(logits):
    return torch.argmax(logits, dim=1) - 22


def _loss_fn_v2(logits, truth_angle, angle_loss_weight=0.2):
    """Hybrid CE + angle-distance loss for ordered steering bins."""
    truth_angle = truth_angle.float().view(-1)
    truth_class = _angle_to_class(truth_angle)
    class_loss = F.cross_entropy(logits, truth_class)

    probs = F.softmax(logits, dim=1)
    class_bins = torch.arange(45, device=logits.device, dtype=probs.dtype) - 22.0
    pred_angle = torch.sum(probs * class_bins.unsqueeze(0), dim=1)
    angle_loss = F.smooth_l1_loss(pred_angle, truth_angle)
    return class_loss + angle_loss_weight * angle_loss


@torch.no_grad()
def _validate(model, dataloader, device, angle_loss_weight):
    model.eval()
    losses = []
    class_accs = []
    bin_maes = []
    for seq, truth_angle in dataloader:
        seq = seq.to(device, non_blocking=True)
        truth_angle = truth_angle.to(device, non_blocking=True)
        logits = model(seq)
        loss = _loss_fn_v2(logits, truth_angle, angle_loss_weight=angle_loss_weight)
        losses.append(loss.item())
        pred_class = torch.argmax(logits, dim=1)
        class_accs.append((pred_class == _angle_to_class(truth_angle)).float().mean().item())
        bin_maes.append(torch.mean(torch.abs(_logits_to_angle(logits).float() - truth_angle.float())).item())

    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    avg_acc = sum(class_accs) / len(class_accs) if class_accs else float("nan")
    avg_mae = sum(bin_maes) / len(bin_maes) if bin_maes else float("nan")
    return avg_loss, avg_acc, avg_mae


def play_temporal_dataset_with_arrow(
    temporal_dataset,
    model,
    model_device,
    model_name="alvinnita",
    start_idx=0,
    end_idx=None,
    step=1,
    fps=15.0,
    arrow_scale=0.35,
    arrow_color="lime",
    pred_arrow_color="dodgerblue",
    repeat=True,
):
    """Animate temporal validation samples, showing last frame and gt/pred arrows."""
    if step <= 0:
        raise ValueError("step must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    n = len(temporal_dataset)
    if n == 0:
        raise ValueError("Temporal dataset is empty")
    if end_idx is None:
        end_idx = n
    start_idx = max(0, start_idx)
    end_idx = min(end_idx, n)
    if start_idx >= end_idx:
        raise ValueError(f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}")
    indices = list(range(start_idx, end_idx, step))
    if not indices:
        raise ValueError("No indices to play after applying range/step")

    def _predict_arrow(seq_tensor: torch.Tensor):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            pred_logits = model(seq_tensor.unsqueeze(0).to(model_device))
        if was_training:
            model.train()
        pred_angle = float(_logits_to_angle(pred_logits)[0].item())
        return _label_to_unit_arrow(torch.tensor([pred_angle]))

    first_seq, first_label = temporal_dataset[indices[0]]
    first_img = first_seq[-1]
    img_np = _to_numpy_image(first_img)
    dx, dy, label_text = _label_to_unit_arrow(first_label)
    pred_dx, pred_dy, pred_text = _predict_arrow(first_seq)

    fig, ax = plt.subplots(figsize=(7, 7))
    im_artist = ax.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)

    h, w = img_np.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    arrow_len = arrow_scale * min(w, h)

    (arrow_artist,) = ax.plot([cx, cx + dx * arrow_len], [cy, cy - dy * arrow_len], color=arrow_color, linewidth=3)
    marker_artist = ax.scatter([cx + dx * arrow_len], [cy - dy * arrow_len], c=arrow_color, s=30)
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

    title_prefix = f"model: {model_name} | "
    ax.set_title(f"{title_prefix}idx={indices[0]} | gt: {label_text} | pred: {pred_text}")
    ax.set_axis_off()

    def _update(frame_i: int):
        idx = indices[frame_i]
        seq, label = temporal_dataset[idx]
        frame_np = _to_numpy_image(seq[-1])
        dx_i, dy_i, text_i = _label_to_unit_arrow(label)
        pred_dx_i, pred_dy_i, pred_text_i = _predict_arrow(seq)

        im_artist.set_data(frame_np)
        h_i, w_i = frame_np.shape[:2]
        cx_i, cy_i = w_i / 2.0, h_i / 2.0
        arrow_len_i = arrow_scale * min(w_i, h_i)

        x2 = cx_i + dx_i * arrow_len_i
        y2 = cy_i - dy_i * arrow_len_i
        pred_x2 = cx_i + pred_dx_i * arrow_len_i
        pred_y2 = cy_i - pred_dy_i * arrow_len_i

        arrow_artist.set_data([cx_i, x2], [cy_i, y2])
        marker_artist.set_offsets(np.array([[x2, y2]]))
        pred_arrow_artist.set_data([cx_i, pred_x2], [cy_i, pred_y2])
        pred_marker_artist.set_offsets(np.array([[pred_x2, pred_y2]]))
        ax.set_title(f"{title_prefix}idx={idx} | gt: {text_i} | pred: {pred_text_i}")

        return im_artist, arrow_artist, marker_artist, pred_arrow_artist, pred_marker_artist

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(indices),
        interval=1000.0 / fps,
        blit=False,
        repeat=repeat,
    )
    return fig, anim


def train_ALVINNITA(
    image_size=(INPUT_SIZE_H, INPUT_SIZE_W),
    history_frames=6,
    n_minibatch_steps=10000,
    batch_size=128,
    learning_rate=3e-4,
    weight_decay=1e-2,
    preload_images=True,
    preload_num_workers=8,
    num_workers=8,
    use_blue=True,
    turn_data_smoothing_window=9,
    val_ratio=0.1,
    log_interval=20,
    val_interval=100,
    angle_loss_weight=0.2,
    model_name="alvinnita",
):
    """
    Dedicated training pipeline for ALVINNITA temporal decoder model.
    """
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dataset = DuckieDriveDataset(
        data_dir=DEF_DATA_DIR,
        preload_images=preload_images,
        cache_preloaded_images=True,
        preload_num_workers=preload_num_workers,
        image_size=image_size,
        use_angles=True,
        use_blue_channel=use_blue,
        smooth_labels=True,
        smoothing_window=turn_data_smoothing_window,
    )

    training_base = base_dataset
    validation_base = training_base.slice_off_validation_dataset(vt_ratio=val_ratio)
    train_dataset = TemporalWindowDataset(training_base, history_frames=history_frames)
    validation_dataset = TemporalWindowDataset(validation_base, history_frames=history_frames)

    if len(train_dataset) == 0:
        raise ValueError("Training window dataset is empty; reduce history_frames or increase dataset size.")
    if len(validation_dataset) == 0:
        raise ValueError("Validation window dataset is empty; reduce history_frames or increase dataset size.")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        validation_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    sample_seq, _ = train_dataset[0]
    input_channels = sample_seq.shape[1]
    model = ALVINNITA(
        imagesize_hw=image_size,
        color_channels=input_channels,
        history_frames=history_frames,
    ).to(device)
    model.set_name(model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    training_loss = []
    val_loss = []
    training_class_acc = []
    validation_class_acc = []
    training_bin_mae = []
    validation_bin_mae = []
    val_ts = []

    i = 0
    model.train()
    while i < n_minibatch_steps:
        for seq, truth_angle in train_loader:
            seq = seq.to(device, non_blocking=True)
            truth_angle = truth_angle.to(device, non_blocking=True)

            logits = model(seq)
            loss = _loss_fn_v2(logits, truth_angle, angle_loss_weight=angle_loss_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item())
            pred_class = torch.argmax(logits, dim=1)
            training_class_acc.append((pred_class == _angle_to_class(truth_angle)).float().mean().item())
            training_bin_mae.append(torch.mean(torch.abs(_logits_to_angle(logits).float() - truth_angle.float())).item())

            if i % val_interval == 0:
                v_loss, v_acc, v_mae = _validate(model, val_loader, device, angle_loss_weight)
                val_loss.append(v_loss)
                validation_class_acc.append(v_acc)
                validation_bin_mae.append(v_mae)
                val_ts.append(i)
                model.train()

            if i % log_interval == 0:
                latest_val = val_loss[-1] if val_loss else float("nan")
                print(
                    f"Progress: {i / n_minibatch_steps:.2f}; Step: {i}/{n_minibatch_steps}; "
                    f"train loss={training_loss[-1]:.3f}; val loss={latest_val:.3f}",
                    end="\r",
                )

            i += 1
            if i >= n_minibatch_steps:
                break

    print("\nCompleted Training!")

    checkpoint_path, latest_path = save_checkpoint(
        model=model,
        model_name=model_name,
        optimizer=optimizer,
        image_size=image_size,
        n_optimization_steps=n_minibatch_steps,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Updated latest checkpoint: {latest_path}")

    plot_training(
        training_loss,
        training_class_acc,
        val_loss,
        validation_class_acc,
        training_bin_mae,
        validation_bin_mae,
        val_ts,
    )

    fig, anim = play_temporal_dataset_with_arrow(
        validation_dataset,
        model=model,
        model_device=device,
        model_name=model_name,
        fps=30,
    )
    plt.show()

    return model
