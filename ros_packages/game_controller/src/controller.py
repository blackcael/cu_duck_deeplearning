#!/usr/bin/env python3
"""
Compatibility entrypoint.

Preferred node path for ROS is:
  ros_packages/game_controller/scripts/controller.py
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "controller.py"
    runpy.run_path(str(script_path), run_name="__main__")
