from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vector3 = Tuple[float, float, float]


class ObjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_dir_path: str
    position: Vector3 | None = Field(default=None)
    name: str | None = None

    @field_validator("position")
    @classmethod
    def validate_position(cls, value: Vector3 | None) -> Vector3 | None:
        if value is None:
            return None
        if len(value) != 3:
            raise ValueError("position must be a sequence of three numbers.")
        return tuple(float(v) for v in value)  # type: ignore[return-value]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_xml_path: str
    out_xml_path: str | None = None
    object_base_dir_path: str | None = None
    objects: list[ObjectConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_default_out_path(self) -> "AppConfig":
        if self.out_xml_path is None:
            base = Path(self.base_xml_path)
            self.out_xml_path = str(base.with_name(f"{base.stem}_env.xml"))
        return self


__all__ = ["AppConfig", "ObjectConfig", "Vector3"]
