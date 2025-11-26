from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import yaml
from pydantic import ValidationError

from .config import AppConfig, ObjectConfig, Vector3

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ConfigError(RuntimeError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ObjectInjectionError(RuntimeError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


_DEFAULT_CONFIG_CANDIDATES: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2] / "configs" / "config.yaml",
    Path(__file__).resolve().parents[2] / "config" / "config.yaml",
)


def _iter_config_candidates(config_path: str | Path | None) -> Iterable[Path]:
    if config_path is not None:
        yield Path(config_path).expanduser()
    else:
        yield from _DEFAULT_CONFIG_CANDIDATES


def _locate_config_path(config_path: str | Path | None) -> Path:
    candidates = list(_iter_config_candidates(config_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = [str(c) for c in candidates]
    raise ConfigError("Configuration file not found.", {"searched_paths": searched})


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read configuration file: {path}") from exc


def load_config(config_path: str | Path | None = None) -> AppConfig:
    target_path = _locate_config_path(config_path)
    raw = _read_yaml(target_path)
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc


def _resolve_path(base_dir: Path, candidate: str | Path) -> Path:
    candidate_path = Path(candidate).expanduser()
    return candidate_path if candidate_path.is_absolute() else base_dir / candidate_path


def _resolve_object_dir(
    obj_cfg: ObjectConfig, config_dir: Path, object_base_dir: Path | None
) -> Path:
    object_dir = Path(obj_cfg.object_dir_path).expanduser()
    if object_dir.is_absolute():
        return object_dir
    if object_base_dir is not None:
        return object_base_dir / object_dir
    return config_dir / object_dir


def _relpath(target: Path, start: Path) -> str:
    return os.path.relpath(target, start)


def _format_float(value: float) -> str:
    text = f"{value:.6f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _format_position(position: Vector3 | None) -> str:
    x, y, z = position if position is not None else (0.0, 0.0, 0.0)
    return " ".join(_format_float(v) for v in (x, y, z))


def _make_unique(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    idx = 1
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def _collect_existing_mesh_names(asset: ET.Element) -> set[str]:
    names: set[str] = set()
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        if name:
            names.add(name)
    return names


def build_env_xml(config: AppConfig, config_dir: Path) -> Path:
    base_xml = _resolve_path(config_dir, config.base_xml_path).resolve()
    out_xml = _resolve_path(config_dir, config.out_xml_path or "").resolve()
    object_base_dir = (
        _resolve_path(config_dir, config.object_base_dir_path).resolve()
        if config.object_base_dir_path
        else None
    )

    if not base_xml.exists():
        raise ObjectInjectionError(f"Base XML not found: {base_xml}")

    try:
        tree = ET.parse(base_xml)
    except ET.ParseError as exc:
        raise ObjectInjectionError(f"Failed to parse base XML: {base_xml}") from exc

    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ObjectInjectionError("Base XML is missing a <worldbody> element.")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    existing_mesh_names = _collect_existing_mesh_names(asset)

    insert_at = 0
    for obj_cfg in config.objects:
        obj_dir = _resolve_object_dir(obj_cfg, config_dir, object_base_dir).resolve()
        if not obj_dir.exists():
            raise ObjectInjectionError(f"Object directory not found: {obj_dir}")

        xml_candidates = sorted(obj_dir.glob("*.xml"))
        if xml_candidates:
            include_path = _relpath(xml_candidates[0], out_xml.parent)
            include_elem = ET.Element("include", {"file": include_path})
            worldbody.insert(insert_at, include_elem)
            insert_at += 1
            logger.info("Included object XML: %s", include_path)
            continue

        mesh_candidates = sorted(list(obj_dir.glob("*.obj")) + list(obj_dir.glob("*.stl")))
        if not mesh_candidates:
            raise ObjectInjectionError(
                "No supported geometry files found in object directory.",
                {"object_dir": str(obj_dir)},
            )

        mesh_file = mesh_candidates[0]
        mesh_name = _make_unique(obj_cfg.name or mesh_file.stem, existing_mesh_names)
        existing_mesh_names.add(mesh_name)

        rel_mesh_path = _relpath(mesh_file, out_xml.parent)
        mesh_elem = ET.Element("mesh", {"name": mesh_name, "file": rel_mesh_path})
        asset.append(mesh_elem)

        body = ET.Element("body", {"name": mesh_name, "pos": _format_position(obj_cfg.position)})
        ET.SubElement(body, "freejoint")
        geom_attrs = {
            "name": f"{mesh_name}_geom",
            "type": "mesh",
            "mesh": mesh_name,
            "contype": "1",
            "conaffinity": "1",
            "density": "1000",
            "rgba": "0.8 0.8 0.8 1",
        }
        ET.SubElement(body, "geom", geom_attrs)
        worldbody.insert(insert_at, body)
        insert_at += 1
        logger.info("Injected mesh object: %s at %s", mesh_name, _format_position(obj_cfg.position))

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    logger.info("Generated environment XML: %s", out_xml)
    return out_xml


def build_env_from_config(config_path: str | Path | None = None) -> Path:
    resolved_config_path = _locate_config_path(config_path)
    config = load_config(resolved_config_path)
    return build_env_xml(config, resolved_config_path.parent)


__all__ = [
    "build_env_from_config",
    "build_env_xml",
    "ConfigError",
    "ObjectInjectionError",
    "load_config",
]
