#!/usr/bin/env python3

import torch
from datetime import datetime
from pathlib import Path
from utils.ALVINN import ALVINN

DEF_MODELS_DIR = "models"

def save_checkpoint(model, optimizer, image_size, n_optimization_steps, save_dir=DEF_MODELS_DIR, model_name="alvinn"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = save_dir / f"{model_name}_{timestamp}.pt"
    latest_path = save_dir / f"{model_name}_latest.pt"

    checkpoint = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": tuple(image_size),
        "n_optimization_steps": n_optimization_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)
    return ckpt_path, latest_path

def load_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ALVINN(imagesize_hw=tuple(checkpoint["image_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint