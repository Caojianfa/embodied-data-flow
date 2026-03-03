"""YAML 配置加载器，支持点号属性访问"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    配置对象，将字典包装为可用点号访问的形式。

    示例：
        cfg = load_config("configs/pipeline.yaml")
        cfg.data.sequence          # "MH_01_easy"
        cfg.model.clip_model       # "ViT-B/32"
        cfg.quality.to_dict()      # 返回原始 dict
    """

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key not in data:
            raise AttributeError(f"配置项不存在: '{key}'")
        val = data[key]
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        val = data[key]
        return Config(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        data = object.__getattribute__(self, "_data")
        val = data.get(key, default)
        return Config(val) if isinstance(val, dict) else val

    def to_dict(self) -> dict:
        return object.__getattribute__(self, "_data")

    def __repr__(self) -> str:
        return f"Config({object.__getattribute__(self, '_data')})"


def load_config(path: str | Path) -> Config:
    """
    加载 YAML 配置文件

    Args:
        path: YAML 文件路径

    Returns:
        Config 对象
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data or {})
