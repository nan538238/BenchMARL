"""Validated task configuration for the VMAS area-defense training pilot."""

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    guidance_mode: str = MISSING
    opponent_style: str = MISSING
    sensor_range: float = MISSING
    red_speed: float = MISSING
    capture_radius: float = MISSING
