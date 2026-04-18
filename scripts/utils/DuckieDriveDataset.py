#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import os
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

DEF_DATA_DIR = "data/"

IMG_NAME_COL = "image_file"
VEL_LEFT_COL = "vel_left"
VEL_RIGHT_COL = "vel_right"
DATA_PAIRS_CSV = "samples.csv"

AXEL_L_MM = 50
ZERO_DEADBAND_BINS = 0.75

# In-process caches to avoid repeated CSV parsing and image decoding on repeated
# training runs from the same Python process (e.g., notebook/REPL loops).
_SAMPLE_CACHE = {}
_PRELOADED_IMAGE_CACHE = {}

def _load_and_transform_image(path: str, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


def filenames_to_color_tensors(
    file_list,
    image_size=(512, 512),
    use_blue_channel=False,
    preload_num_workers: int = 1,
):
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),  # [3, H, W], values in [0,1]
        T.Lambda(lambda img: img[2:3, :, :]) if use_blue_channel else T.Lambda(lambda img: img),
    ])

    if preload_num_workers <= 1:
        tensors = []
        for path in file_list:
            tensors.append(_load_and_transform_image(path, transform))
        return tensors

    # Parallel image decode/resize using thread helpers.
    with ThreadPoolExecutor(max_workers=preload_num_workers) as executor:
        tensors = list(executor.map(_load_and_transform_image, file_list, repeat(transform)))
    return tensors


def load_drive_samples(data_folder_path: str, filter_idle_frames: bool = True):
    """Load csv metadata and return (image_paths, velocity_tensor)."""
    cache_key = (data_folder_path, filter_idle_frames)
    cached = _SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        image_paths, velocity_tensor = cached
        # Return shallow copies so per-dataset mutations do not affect cache.
        return list(image_paths), velocity_tensor.clone()

    data_pairs_csv = os.path.join(data_folder_path, DATA_PAIRS_CSV)
    csv_data = pd.read_csv(data_pairs_csv)

    if filter_idle_frames:
        before = len(csv_data)
        csv_data = csv_data[(csv_data[VEL_LEFT_COL] != 0.0) | (csv_data[VEL_RIGHT_COL] != 0.0)]
        print(
            f"Length before filtering: {before}, length after filtering: {len(csv_data)}, removed {before - len(csv_data)} elements"
        )

    image_paths = [
        os.path.join(data_folder_path, rel_path)
        for rel_path in csv_data[IMG_NAME_COL].astype(str).tolist()
    ]
    velocity_np = csv_data[[VEL_LEFT_COL, VEL_RIGHT_COL]].to_numpy(dtype="float32")
    velocity_tensor = torch.tensor(velocity_np, dtype=torch.float32)

    _SAMPLE_CACHE[cache_key] = (list(image_paths), velocity_tensor.clone())
    return image_paths, velocity_tensor


def clear_dataset_cache():
    """Clear module-level in-memory caches for samples and preloaded images."""
    _SAMPLE_CACHE.clear()
    _PRELOADED_IMAGE_CACHE.clear()


def csv_and_images_to_tensors(
    data_folder_path: str,
    filter_idle_frames=True,
    image_size=(512, 512),
    use_blue_channel=False,
    preload_num_workers: int = 1,
):
    """
    Eager path: load *all* images immediately into one tensor.
    This is convenient, but slower at startup and uses much more memory.
    """
    image_paths, velocity_tensor = load_drive_samples(data_folder_path, filter_idle_frames)

    print("starting assembling image tensor")
    image_tensor = torch.stack(
        filenames_to_color_tensors(
            image_paths,
            image_size=image_size,
            use_blue_channel=use_blue_channel,
            preload_num_workers=preload_num_workers,
        )
    )
    print("end of assembling image tensor")

    print(f"image_tensor shape: {tuple(image_tensor.shape)}")
    print(f"velocity_tensor shape: {tuple(velocity_tensor.shape)}")

    return image_tensor, velocity_tensor


class DuckieDriveDataset(Dataset):
    """
    Paired driving dataset. Returns (image, velocity) per sample.

    Fast path:
      DuckieDriveDataset(data_dir="data/", preload_images=False)

    Legacy/eager path:
      DuckieDriveDataset(data_dir="data/", preload_images=True)
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        image_tensor: Optional[torch.Tensor] = None,
        velocity_tensor: Optional[torch.Tensor] = None,
        image_paths = None,
        filter_idle_frames: bool = True,
        image_size=(512, 512),
        preload_images: bool = False,
        use_angles: bool = False,
        use_blue_channel: bool = False,
        smooth_labels: bool = False,
        smoothing_window: int = 5,
        cache_preloaded_images: bool = True,
        preload_num_workers: int = 1,
    ):
        super().__init__()

        self.image_tensor = None
        self.image_paths = None
        self.use_blue_channel = use_blue_channel
        self.smooth_labels = smooth_labels
        self.smoothing_window = smoothing_window
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Lambda(lambda img: img[2:3, :, :]) if self.use_blue_channel else T.Lambda(lambda img: img),
        ])

        if data_dir is not None or image_paths is not None:
            if image_paths is None:
                image_paths, velocity_tensor = load_drive_samples(
                    data_dir,
                    filter_idle_frames=filter_idle_frames,
                )
            if preload_images:
                print("preloading all images into memory")
                cache_key = None
                if cache_preloaded_images and data_dir is not None and image_paths is not None:
                    cache_key = (
                        os.path.abspath(data_dir),
                        bool(filter_idle_frames),
                        tuple(image_size),
                        bool(self.use_blue_channel),
                    )
                    cached_tensor = _PRELOADED_IMAGE_CACHE.get(cache_key)
                    if cached_tensor is not None:
                        print("using cached preloaded images")
                        self.image_tensor = cached_tensor

                if self.image_tensor is None:
                    self.image_tensor = torch.stack(
                        filenames_to_color_tensors(
                            image_paths,
                            image_size=image_size,
                            use_blue_channel=self.use_blue_channel,
                            preload_num_workers=preload_num_workers,
                        )
                    )
                    if cache_key is not None:
                        _PRELOADED_IMAGE_CACHE[cache_key] = self.image_tensor
            else:
                self.image_paths = image_paths

        elif image_tensor is not None and velocity_tensor is not None:
            self.image_tensor = image_tensor

        else:
            raise ValueError(
                "Provide either data_dir, or both image_tensor and velocity_tensor"
            )

        if velocity_tensor is None:
            raise ValueError("velocity_tensor is required")

        if self.smooth_labels:
            velocity_tensor = self._smooth_velocity_tensor(velocity_tensor, self.smoothing_window)

        self.velocity_tensor = velocity_tensor

        if self.image_tensor is not None and len(self.image_tensor) != len(self.velocity_tensor):
            raise ValueError(
                f"Length mismatch: {len(self.image_tensor)} images vs {len(self.velocity_tensor)} velocities"
            )

        if self.image_paths is not None and len(self.image_paths) != len(self.velocity_tensor):
            raise ValueError(
                f"Length mismatch: {len(self.image_paths)} image paths vs {len(self.velocity_tensor)} velocities"
            )
        
        self.use_angles = use_angles
        if self.use_angles:
            max_w, min_w = self.get_max_min_w(AXEL_L_MM)
            max_mag_w = max_w if max_w > abs(min_w) else abs(min_w)
            self.angle_tensor = self.scale_v_pair_to_45(self.velocity_tensor, max_mag_w, AXEL_L_MM)
            

    def __len__(self):
        return len(self.velocity_tensor)

    def __getitem__(self, idx):
        if self.image_tensor is not None:
            image = self.image_tensor[idx]
        else:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.transform(image)

        w = self.angle_tensor[idx] if self.use_angles else self.velocity_tensor[idx]
        return image, w
    

    def slice_off_validation_dataset(self, vt_ratio=0.1):
        slice_idx = int((1-vt_ratio) * len(self.velocity_tensor))
        
        validation_vels = self.velocity_tensor[slice_idx:]
        self.velocity_tensor = self.velocity_tensor[:slice_idx]


        if self.image_tensor is not None:
            validation_imgs = self.image_tensor[slice_idx:]
            self.image_tensor = self.image_tensor[:slice_idx]
            validation_dataset = DuckieDriveDataset(
                image_tensor=validation_imgs,
                velocity_tensor=validation_vels,
                use_angles=self.use_angles,
                use_blue_channel=self.use_blue_channel,
                smooth_labels=False,
                smoothing_window=self.smoothing_window,
            )
        else:
            validation_imgs = self.image_paths[slice_idx:]
            self.image_paths = self.image_paths[:slice_idx]
            validation_dataset = DuckieDriveDataset(
                image_paths=validation_imgs,
                velocity_tensor=validation_vels,
                image_size=self.transform.transforms[0].size,
                use_angles=self.use_angles,
                use_blue_channel=self.use_blue_channel,
                smooth_labels=False,
                smoothing_window=self.smoothing_window,
            )

        return validation_dataset
    
    def get_average_speed(self):
        v = 0
        for v_l, v_r in self.velocity_tensor:
            v += (v_l + v_r) / 2
        return v / len(self.velocity_tensor)
    
    def get_max_min_w(self, AXEL_L):
        max = 0
        min = 0
        for v_l, v_r in self.velocity_tensor:
            w = (v_r - v_l) / AXEL_L
            if w > max: max = w
            if w < min: min = w
        return(max, min) 
    
    def scale_v_pair_to_45(self, v_pair, max_mag_w, axel_len_mm):
        """
        takes a value between 0 and 45. 23 is the middle value, so in reality we are scaling from -22 to 0 to 22
        """
        v_pair = torch.as_tensor(v_pair, dtype=torch.float32)

        if v_pair.ndim == 1:
            # Single (v_l, v_r) pair.
            w = (v_pair[0] - v_pair[1]) / axel_len_mm
        elif v_pair.ndim == 2 and v_pair.shape[1] == 2:
            # Batch of velocity pairs shaped [N, 2].
            w = (v_pair[:, 0] - v_pair[:, 1]) / axel_len_mm
        else:
            raise ValueError(f"Expected shape [2] or [N, 2], got {tuple(v_pair.shape)}")

        max_mag_w = torch.as_tensor(max_mag_w, dtype=v_pair.dtype)
        unit_w = w / torch.clamp(max_mag_w, min=torch.finfo(v_pair.dtype).eps)
        scaled_bins = unit_w * 22.0

        # Keep small near-zero steering values at 0 to reduce jitter from sensor noise.
        scaled_bins = torch.where(
            torch.abs(scaled_bins) < ZERO_DEADBAND_BINS,
            torch.zeros_like(scaled_bins),
            scaled_bins,
        )
        return torch.clamp(torch.round(scaled_bins), -22, 22)

    def _smooth_velocity_tensor(self, velocity_tensor, window_size):
        """
        Smooth sequential [N,2] wheel-velocity labels with a centered moving average.
        """
        v = torch.as_tensor(velocity_tensor, dtype=torch.float32)

        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(f"Expected velocity tensor shape [N, 2], got {tuple(v.shape)}")

        if window_size <= 1:
            return v

        if window_size % 2 == 0:
            window_size += 1

        # [N,2] -> [1,2,N] so avg_pool1d can smooth along time dimension.
        vt = v.transpose(0, 1).unsqueeze(0)
        pad = window_size // 2
        vt = F.pad(vt, (pad, pad), mode="replicate")
        vt = F.avg_pool1d(vt, kernel_size=window_size, stride=1)
        return vt.squeeze(0).transpose(0, 1)




if __name__ == "__main__":
    # Fast startup: lazy image loading.
    dataset = DuckieDriveDataset(data_dir=DEF_DATA_DIR, preload_images=False)
    print(f"dataset length: {len(dataset)}")

    training_dataset = dataset
    validation_dataset = training_dataset.slice_off_validation_dataset()
    print(f"Training dataset length: {len(training_dataset)}, Validation dataset length:  {len(validation_dataset)}")

    # If you still want eager tensors, keep using this API:
    # img, vel = csv_and_images_to_tensors(DEF_DATA_DIR)
