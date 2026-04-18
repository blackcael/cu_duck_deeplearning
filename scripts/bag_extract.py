#!/usr/bin/env python3

import os
import csv
import bisect
import argparse
from pathlib import Path

import cv2
import numpy as np

from rosbags.highlevel import AnyReader


ROOT_DIR = Path(__file__).resolve().parents[1]
BAG_DIR = ROOT_DIR / "rosbags"
IMAGE_TOPIC = "/turbogoose/camera_node/image/compressed"
CMD_TOPIC = "/turbogoose/wheels_driver_node/wheels_cmd"
OUTPUT_DIR = str(ROOT_DIR / "data")

# Set to either:
#   "nearest"     -> nearest command in time
#   "latest_prior"-> most recent command at or before image time
PAIRING_MODE = "latest_prior"

# Maximum allowed time gap between image and command, in seconds.
# If no command is close enough, the sample is skipped.
MAX_DT = 0.2


def stamp_to_sec(stamp):
    sec = getattr(stamp, "sec", None)
    nsec = getattr(stamp, "nanosec", None)
    if sec is None:
        sec = getattr(stamp, "secs", None)
    if nsec is None:
        nsec = getattr(stamp, "nsec", None)
    if nsec is None:
        nsec = getattr(stamp, "nsecs", None)
    if sec is None or nsec is None:
        return None
    return float(sec) + float(nsec) * 1e-9


def get_msg_time(msg, bag_time_ns):
    """Prefer message header stamp if present, otherwise bag record time."""
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        try:
            ts = stamp_to_sec(msg.header.stamp)
            if ts > 0:
                return ts
        except Exception:
            pass
    return float(bag_time_ns) * 1e-9


def get_msg_type(msg, fallback=""):
    return (
        getattr(msg, "_type", None)
        or getattr(msg, "__msgtype__", None)
        or fallback
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def choose_image_filename(images_dir, ts):
    """Avoid overwriting if a timestamp filename already exists."""
    stem = f"{ts:.6f}"
    candidate = f"{stem}.png"
    i = 1
    while os.path.exists(os.path.join(images_dir, candidate)):
        candidate = f"{stem}_{i}.png"
        i += 1
    return candidate


def image_from_raw_msg(msg):
    """Decode sensor_msgs/Image into a BGR OpenCV image for common encodings."""
    encoding = str(getattr(msg, "encoding", "")).lower()
    height = int(getattr(msg, "height"))
    width = int(getattr(msg, "width"))
    data = np.frombuffer(bytes(getattr(msg, "data")), dtype=np.uint8)

    if encoding in {"bgr8", "rgb8"}:
        expected = height * width * 3
        if data.size != expected:
            return None
        img = data.reshape((height, width, 3))
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if encoding in {"bgra8", "rgba8"}:
        expected = height * width * 4
        if data.size != expected:
            return None
        img = data.reshape((height, width, 4))
        if encoding == "rgba8":
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if encoding == "mono8":
        expected = height * width
        if data.size != expected:
            return None
        gray = data.reshape((height, width))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if encoding == "mono16":
        data16 = np.frombuffer(bytes(getattr(msg, "data")), dtype=np.uint16)
        expected = height * width
        if data16.size != expected:
            return None
        gray16 = data16.reshape((height, width))
        gray8 = cv2.convertScaleAbs(gray16, alpha=255.0 / 65535.0)
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    return None


def save_image_from_msg(msg, msg_type, out_path):
    """Handles sensor_msgs/Image and sensor_msgs/CompressedImage without cv_bridge."""
    if msg_type in {
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/CompressedImage",
    }:
        compressed = np.frombuffer(bytes(getattr(msg, "data")), dtype=np.uint8)
        img = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode compressed image payload.")
            return False
        cv2.imwrite(out_path, img)
        return True

    if msg_type in {"sensor_msgs/msg/Image", "sensor_msgs/Image"}:
        img = image_from_raw_msg(msg)
        if img is None:
            enc = getattr(msg, "encoding", "unknown")
            print(f"Unsupported or malformed raw image encoding: {enc}")
            return False
        cv2.imwrite(out_path, img)
        return True

    print(f"Unsupported image message type: {msg_type}")
    return False


def extract_wheel_values(msg):
    """
    Customize this for your wheel command message type.

    Returns a dict of values to store in CSV.
    """
    msg_type = get_msg_type(msg)

    # Example: geometry_msgs/Twist
    if msg_type == "geometry_msgs/Twist":
        return {
            "linear_x": msg.linear.x,
            "linear_y": msg.linear.y,
            "linear_z": msg.linear.z,
            "angular_x": msg.angular.x,
            "angular_y": msg.angular.y,
            "angular_z": msg.angular.z,
        }

    # Example: geometry_msgs/TwistStamped
    elif msg_type == "geometry_msgs/TwistStamped":
        return {
            "linear_x": msg.twist.linear.x,
            "linear_y": msg.twist.linear.y,
            "linear_z": msg.twist.linear.z,
            "angular_x": msg.twist.angular.x,
            "angular_y": msg.twist.angular.y,
            "angular_z": msg.twist.angular.z,
        }

    # Example: Duckietown WheelsCmdStamped-like message
    # Adjust field names if needed.
    elif hasattr(msg, "vel_left") and hasattr(msg, "vel_right"):
        return {
            "vel_left": msg.vel_left,
            "vel_right": msg.vel_right,
        }

    # Fallback: dump visible numeric-ish fields
    result = {}
    for attr in dir(msg):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(msg, attr)
            if isinstance(value, (int, float, bool, str)):
                result[attr] = value
        except Exception:
            pass
    return result


def pair_command(image_ts, cmd_times, mode):
    if not cmd_times:
        return None, None

    idx = bisect.bisect_left(cmd_times, image_ts)

    if mode == "latest_prior":
        if idx == 0:
            return None, None
        chosen = idx - 1
        return chosen, abs(cmd_times[chosen] - image_ts)

    elif mode == "nearest":
        candidates = []
        if idx < len(cmd_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        if not candidates:
            return None, None

        chosen = min(candidates, key=lambda i: abs(cmd_times[i] - image_ts))
        return chosen, abs(cmd_times[chosen] - image_ts)

    else:
        raise ValueError(f"Unknown pairing mode: {mode}")


def load_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def merge_fieldnames(rows):
    seen = set()
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def dedupe_rows(rows, key_fields):
    seen = set()
    unique = []
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def resolve_bag_paths(bag_dir, bag_names):
    if bag_names is None:
        discovered = sorted(Path(bag_dir).glob("*.bag"))
        bag_paths = [p.expanduser().resolve() for p in discovered if p.is_file()]
        if not bag_paths:
            raise FileNotFoundError(f"No .bag files found in: {Path(bag_dir).expanduser().resolve()}")
        return bag_paths

    bag_paths = []
    for bag_name in bag_names:
        bag_path = Path(bag_name)
        if not bag_path.is_absolute():
            bag_path = Path(bag_dir) / bag_name
        bag_path = bag_path.expanduser().resolve()
        if not bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")
        bag_paths.append(bag_path)
    return bag_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract images + wheel commands from one or more ROS bag files."
    )
    parser.add_argument(
        "--bags",
        nargs="+",
        default=None,
        help="Bag filenames or paths to process.",
    )
    parser.add_argument(
        "--bag-dir",
        default=str(BAG_DIR),
        help="Directory used for relative --bags values.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Dataset directory containing images/, samples.csv, and wheel_cmds.csv.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace CSV files instead of appending to existing dataset rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bag_paths = resolve_bag_paths(args.bag_dir, args.bags)
    print(f"Found {len(bag_paths)} bag file(s) to process")

    ensure_dir(args.output_dir)
    images_dir = os.path.join(args.output_dir, "images")
    ensure_dir(images_dir)

    images = []
    commands = []

    for bag_path in bag_paths:
        print(f"Opening bag: {bag_path}")
        with AnyReader([bag_path]) as reader:
            selected_connections = [
                conn for conn in reader.connections
                if conn.topic in {IMAGE_TOPIC, CMD_TOPIC}
            ]
            for connection, t_ns, rawdata in reader.messages(connections=selected_connections):
                topic = connection.topic
                msg = reader.deserialize(rawdata, connection.msgtype)
                msg_type = get_msg_type(msg, fallback=connection.msgtype)
                ts = get_msg_time(msg, t_ns)

                if topic == IMAGE_TOPIC:
                    filename = choose_image_filename(images_dir, ts)
                    out_path = os.path.join(images_dir, filename)

                    ok = save_image_from_msg(msg, msg_type, out_path)
                    if ok:
                        images.append({
                            "image_ts": ts,
                            "image_file": f"images/{filename}",
                        })

                elif topic == CMD_TOPIC:
                    values = extract_wheel_values(msg)
                    row = {"cmd_ts": ts}
                    row.update(values)
                    commands.append(row)

    images.sort(key=lambda x: x["image_ts"])
    commands.sort(key=lambda x: x["cmd_ts"])

    print(f"Saved {len(images)} new images")
    print(f"Read {len(commands)} new commands")

    # Write all commands
    wheel_csv = os.path.join(args.output_dir, "wheel_cmds.csv")
    if commands:
        existing_commands = [] if args.replace else load_csv_rows(wheel_csv)
        all_commands = dedupe_rows(existing_commands + commands, key_fields=["cmd_ts"])
        all_commands.sort(key=lambda x: float(x["cmd_ts"]))

        cmd_fieldnames = merge_fieldnames(all_commands)
        with open(wheel_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cmd_fieldnames)
            writer.writeheader()
            writer.writerows(all_commands)
        print(f"Wrote {wheel_csv} ({len(all_commands)} rows)")

    # Build paired samples for current extraction
    cmd_times = [float(c["cmd_ts"]) for c in commands]
    samples = []

    for img in images:
        idx, dt = pair_command(float(img["image_ts"]), cmd_times, PAIRING_MODE)
        if idx is None:
            continue
        if dt > MAX_DT:
            continue

        sample = {
            "image_ts": float(img["image_ts"]),
            "image_file": img["image_file"],
            "cmd_ts": float(commands[idx]["cmd_ts"]),
            "dt": dt,
        }

        for k, v in commands[idx].items():
            if k != "cmd_ts":
                sample[k] = v

        samples.append(sample)

    samples_csv = os.path.join(args.output_dir, "samples.csv")
    if samples:
        existing_samples = [] if args.replace else load_csv_rows(samples_csv)
        all_samples = dedupe_rows(existing_samples + samples, key_fields=["image_file"])
        all_samples.sort(key=lambda x: float(x["image_ts"]))

        fieldnames = merge_fieldnames(all_samples)
        with open(samples_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_samples)
        print(f"Wrote {samples_csv} ({len(all_samples)} rows)")
        print(f"Paired {len(samples)} new samples")
    else:
        print("No paired samples were created. Check topic names or MAX_DT.")


if __name__ == "__main__":
    main()
