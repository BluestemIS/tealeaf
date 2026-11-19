"""Suites and cases. A case is an async function that drives the model through a Ctx."""
from __future__ import annotations

import fnmatch
import inspect
from dataclasses import dataclass, field
from typing import Awaitable, Callable

CaseFn = Callable[..., Awaitable[None]]


@dataclass
class Case:
    suite: str
    name: str
    title: str
    fn: CaseFn
    kind: str  # "traditional" (scored benchmark) or "probe" (exploratory, judge-annotated)
    description: str = ""
    look_for: str = ""  # guidance for the judge's annotation pass
    tags: tuple = ()
    annotate: bool = False
    pass_threshold: float = 0.5

    @property
    def id(self) -> str:
        return f"{self.suite}.{self.name}"

    def info(self) -> dict:
        return {"id": self.id, "suite": self.suite, "name": self.name, "title": self.title, "kind": self.kind,
                "description": self.description, "look_for": self.look_for, "tags": list(self.tags)}


@dataclass
class Suite:
    name: str
    kind: str
    description: str
    cases: list[Case] = field(default_factory=list)

    def case(self, name: str, title: str | None = None, *, look_for: str = "", tags=(), annotate: bool | None = None,
             pass_threshold: float = 0.5, kind: str | None = None):
        def deco(fn: CaseFn) -> CaseFn:
            self.add(name, fn, title=title, look_for=look_for, tags=tags, annotate=annotate,
                     pass_threshold=pass_threshold, kind=kind)
            return fn
        return deco

    def add(self, name: str, fn: CaseFn, *, title=None, look_for="", tags=(), annotate=None, pass_threshold=0.5,
            kind=None, description=None):
        k = kind or self.kind
        self.cases.append(Case(
            suite=self.name, name=name, title=title or name.replace("_", " ").capitalize(), fn=fn, kind=k,
            description=description if description is not None else inspect.cleandoc(fn.__doc__ or ""),
            look_for=look_for, tags=tuple(tags), annotate=(k == "probe") if annotate is None else annotate,
            pass_threshold=pass_threshold,
        ))

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "description": self.description,
                "cases": [c.info() for c in self.cases]}


SUITES: dict[str, Suite] = {}


def suite(name: str, kind: str, description: str) -> Suite:
    s = Suite(name, kind, description)
    SUITES[name] = s
    return s


def all_cases() -> list[Case]:
    from . import suites  # noqa: F401  (registers everything)
    return [c for s in SUITES.values() for c in s.cases]


def select_cases(suite_names=None, patterns=None, kinds=None, exclude=None) -> list[Case]:
    """Select by suite name, glob on case id (e.g. 'beliefs.*', '*.identity'), and kind."""
    cases = all_cases()
    if suite_names:
        unknown = set(suite_names) - set(SUITES)
        if unknown:
            raise ValueError(f"unknown suite(s): {', '.join(sorted(unknown))}. Known: {', '.join(SUITES)}")
        cases = [c for c in cases if c.suite in suite_names]
    if patterns:
        cases = [c for c in cases if any(fnmatch.fnmatch(c.id, p) for p in patterns)]
    if kinds:
        cases = [c for c in cases if c.kind in kinds]
    if exclude:
        cases = [c for c in cases if not any(fnmatch.fnmatch(c.id, p) for p in exclude)]
    return cases
