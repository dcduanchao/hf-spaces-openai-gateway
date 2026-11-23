from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, model_validator


class ExtraSpec(BaseModel):
    target: str
    kind: str = "str"  # str, int, float, bool
    default: Any = None
    min: float | None = None
    max: float | None = None
    choices: list[Any] | None = None


class Profile(BaseModel):
    id: str
    adapter: str = "generic"
    space_id: str
    api_name: str | None = None
    enabled: bool = False
    description: str = ""
    modes: list[str] = Field(default_factory=lambda: ["generations"])
    default_size: str = "1024x1024"
    allowed_sizes: list[str] = Field(default_factory=lambda: ["1024x1024"])
    bindings: dict[str, str] = Field(default_factory=dict)
    extras: dict[str, ExtraSpec] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)
    result_path: list[str | int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_profile(self):
        if "/" not in self.space_id or self.space_id.startswith("/"):
            raise ValueError("space_id must be owner/space")
        if self.enabled and not self.api_name:
            raise ValueError(f"{self.id}: enabled model requires api_name")
        if len(set(self.extras)) != len(self.extras):
            raise ValueError("duplicate extra names")
        return self


def load_profiles() -> dict[str, Profile]:
    path = Path(os.getenv("MODEL_CATALOG", Path(__file__).with_name("catalog.json")))
    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles = [Profile.model_validate(x) for x in raw["models"]]
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError("duplicate model id")
    return {p.id: p for p in profiles if p.enabled}


def validate_extra(raw: Any, spec: ExtraSpec):
    if raw is None:
        return None
    types = {"str": str, "bool": bool, "int": int, "float": (int, float)}
    if spec.kind not in types or (isinstance(raw, bool) and spec.kind != "bool") or not isinstance(raw, types[spec.kind]):
        raise ValueError(f"expected {spec.kind}")
    if spec.choices is not None and raw not in spec.choices:
        raise ValueError(f"must be one of {spec.choices}")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if spec.min is not None and raw < spec.min:
            raise ValueError(f"must be >= {spec.min}")
        if spec.max is not None and raw > spec.max:
            raise ValueError(f"must be <= {spec.max}")
    return raw
