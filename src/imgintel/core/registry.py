"""Analyzer discovery.

Three layers, in increasing order of trust required:

1. **Built-in**  — modules inside ``imgintel.analyzers``.
2. **Installed** — the ``imgintel.analyzers`` entry-point group, so third
   parties can ``pip install imgintel-anpr`` and have it appear automatically.
3. **Local**     — loose ``.py`` files in the user plugin directory. These are
   arbitrary code from a directory, so they load only on explicit opt-in.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import sys
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from platformdirs import user_config_dir

from imgintel.core.analyzer import Analyzer

ENTRY_POINT_GROUP = "imgintel.analyzers"


def user_plugin_dir() -> Path:
    return Path(user_config_dir("imgintel")) / "plugins"


def user_rules_dir() -> Path:
    """Where declarative rule files live.

    Separate from the Python plugin directory, and loaded without the
    ``--allow-local-plugins`` opt-in: rule files are data that cannot execute
    anything, so they do not carry the code-execution risk that gates loose
    ``.py`` files.
    """
    return Path(user_config_dir("imgintel")) / "rules"


@dataclass(slots=True)
class LoadError:
    source: str
    error: str


class Registry:
    """Holds the set of known analyzers and where each came from."""

    def __init__(self) -> None:
        self._analyzers: dict[str, Analyzer] = {}
        self._origins: dict[str, str] = {}
        self.errors: list[LoadError] = []

    # -- population --------------------------------------------------------

    def register(self, analyzer: Analyzer, origin: str = "manual") -> None:
        if not analyzer.name:
            raise ValueError(f"{type(analyzer).__name__} has no name")
        if analyzer.name in self._analyzers:
            existing = self._origins[analyzer.name]
            raise ValueError(
                f"duplicate analyzer {analyzer.name!r} from {origin} "
                f"(already registered by {existing})"
            )
        self._analyzers[analyzer.name] = analyzer
        self._origins[analyzer.name] = origin

    def _register_module(self, module, origin: str) -> None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, Analyzer) or obj is Analyzer:
                continue
            if inspect.isabstract(obj) or not obj.name:
                continue
            # Skip classes merely imported into this module from elsewhere.
            if obj.__module__ != module.__name__:
                continue
            try:
                self.register(obj(), origin)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(LoadError(f"{origin}:{obj.__name__}", str(exc)))

    def load_builtin(self) -> None:
        import imgintel.analyzers as pkg

        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            mod_name = f"{pkg.__name__}.{info.name}"
            try:
                self._register_module(importlib.import_module(mod_name), "builtin")
            except Exception as exc:  # noqa: BLE001
                self.errors.append(LoadError(mod_name, f"{type(exc).__name__}: {exc}"))

    def load_entry_points(self) -> None:
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                obj = ep.load()
                origin = f"package:{ep.value.split(':')[0].split('.')[0]}"
                if inspect.isclass(obj) and issubclass(obj, Analyzer):
                    self.register(obj(), origin)
                elif inspect.ismodule(obj):
                    self._register_module(obj, origin)
                else:
                    self.errors.append(
                        LoadError(ep.name, "entry point is neither an Analyzer nor a module")
                    )
            except Exception as exc:  # noqa: BLE001
                self.errors.append(LoadError(ep.name, f"{type(exc).__name__}: {exc}"))

    def load_local(self, directory: Path | None = None) -> None:
        """Load loose .py files. Caller must have obtained explicit consent."""
        directory = directory or user_plugin_dir()
        if not directory.is_dir():
            return
        for py in sorted(directory.glob("*.py")):
            if py.name.startswith("_"):
                continue
            mod_name = f"imgintel_local_{py.stem}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, py)
                if spec is None or spec.loader is None:
                    raise ImportError("could not build module spec")
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
                self._register_module(module, f"local:{py.name}")
            except Exception as exc:  # noqa: BLE001
                self.errors.append(LoadError(str(py), f"{type(exc).__name__}: {exc}"))

    def load_rules(self, directory: Path | None = None) -> None:
        """Load declarative rule files as analyzers.

        Not gated behind ``--allow-local-plugins``: a rule file is data, is
        parsed rather than executed, and cannot do anything a condition
        evaluator does not already do. The opt-in exists for arbitrary Python,
        and applying it here would only discourage the safer option.
        """
        from imgintel.plugins.declarative import load_rule_analyzers

        directory = directory or user_rules_dir()
        analyzers, errors = load_rule_analyzers(directory)
        for analyzer in analyzers:
            try:
                self.register(analyzer, f"rules:{Path(analyzer.ruleset.source or '').name}")
            except ValueError as exc:
                self.errors.append(LoadError(str(analyzer.ruleset.source), str(exc)))
        for error in errors:
            self.errors.append(LoadError("rules", error))

    @classmethod
    def discover(cls, *, allow_local: bool = False, rules: bool = True) -> Registry:
        reg = cls()
        reg.load_builtin()
        reg.load_entry_points()
        if rules:
            reg.load_rules()
        if allow_local:
            reg.load_local()
        return reg

    # -- queries -----------------------------------------------------------

    def get(self, name: str) -> Analyzer | None:
        return self._analyzers.get(name)

    def origin(self, name: str) -> str:
        return self._origins.get(name, "unknown")

    def names(self) -> list[str]:
        return sorted(self._analyzers)

    def all(self) -> list[Analyzer]:
        return [self._analyzers[n] for n in self.names()]

    def __len__(self) -> int:
        return len(self._analyzers)

    def __contains__(self, name: object) -> bool:
        return name in self._analyzers

    def resolve(self, names: list[str]) -> tuple[list[Analyzer], list[str]]:
        """Map names to analyzers, pulling in ``requires`` transitively.

        Returns (analyzers, unknown_names).
        """
        selected: dict[str, Analyzer] = {}
        unknown: list[str] = []
        stack = list(names)
        while stack:
            name = stack.pop()
            if name in selected:
                continue
            analyzer = self._analyzers.get(name)
            if analyzer is None:
                if name not in unknown:
                    unknown.append(name)
                continue
            selected[name] = analyzer
            stack.extend(analyzer.requires)
        return [selected[n] for n in sorted(selected)], unknown
