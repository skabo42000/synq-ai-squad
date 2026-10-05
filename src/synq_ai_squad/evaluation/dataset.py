"""Load and validate the versioned evaluation files in evals/ (dataset, thresholds, judge labels)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from synq_ai_squad.config import PROJECT_ROOT

EVALS_DIR = PROJECT_ROOT / "evals"
Category = Literal["normal", "fabrication", "jargon", "injection", "edge"]


class Case(BaseModel):
    id: str
    category: Category
    request: str = Field(min_length=5, max_length=500)  # same limits as the API
    forbidden: list[str] = []


class Dataset(BaseModel):
    version: int
    cases: list[Case]

    @model_validator(mode="after")
    def _check(self) -> "Dataset":
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        for c in self.cases:
            if c.category in ("fabrication", "injection") and not c.forbidden:
                raise ValueError(f"{c.id}: {c.category} cases need forbidden phrases, or a leak can't be detected")
        return self

    def select(self, only: str | None) -> list[Case]:
        """All cases, or those whose id contains `only`, or a whole category (`only` = category name)."""
        if not only:
            return list(self.cases)
        return [c for c in self.cases if c.category == only or only in c.id]


class Thresholds(BaseModel):
    max_fabrication_leaks: int = 0
    min_pass_rate: float = 0.0
    min_category_pass_rate: dict[str, float] = {}


class Label(BaseModel):
    claim: str
    supported: bool
    note: str = ""


def _read(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_dataset(path: Path = EVALS_DIR / "dataset.yaml") -> Dataset:
    return Dataset.model_validate(_read(path))


def load_thresholds(path: Path = EVALS_DIR / "thresholds.yaml") -> Thresholds:
    return Thresholds.model_validate(_read(path))


def load_labels(path: Path = EVALS_DIR / "judge_labels.yaml") -> list[Label]:
    data = _read(path)
    assert isinstance(data, dict)
    return [Label.model_validate(x) for x in data["labels"]]
