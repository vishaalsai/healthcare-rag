"""
Phase 2 – Production Quality
Prompt Manager: loads versioned prompts from prompts.yaml,
provides template substitution and version tracking for auditability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class PromptManager:
    """
    Reads prompt templates from a YAML registry.
    Supports variable substitution via .format(**kwargs).

    The prompt registry is a single source of truth — every prompt
    change is reflected by bumping its version field, enabling
    reproducibility in evaluations.
    """

    DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts.yaml"

    def __init__(self, prompts_path: str | Path | None = None) -> None:
        self._path = Path(prompts_path) if prompts_path else self.DEFAULT_PROMPTS_PATH
        self._registry: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get(self, name: str, **variables: Any) -> str:
        """
        Retrieve a prompt by name and substitute {variables}.

        Args:
            name:      Key in the YAML prompts section (e.g. 'healthcare_rag_system')
            variables: Named variables to substitute into the template.

        Returns:
            Rendered prompt string.

        Raises:
            KeyError: if the prompt name is not found.
        """
        entry = self._registry.get(name)
        if entry is None:
            raise KeyError(
                f"Prompt '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )

        content: str = entry["content"]
        if variables:
            try:
                content = content.format(**variables)
            except KeyError as exc:
                raise KeyError(
                    f"Prompt '{name}' is missing variable {exc}. "
                    f"Required variables: {self._extract_variables(content)}"
                ) from exc

        return content.strip()

    def version(self, name: str) -> str:
        """Return the version string for a prompt."""
        entry = self._registry.get(name)
        if entry is None:
            raise KeyError(f"Prompt '{name}' not found.")
        return entry.get("version", "unknown")

    def schema_version(self) -> str:
        return self._raw.get("schema_version", "unknown")

    def list_prompts(self) -> list[str]:
        return list(self._registry.keys())

    def reload(self) -> None:
        """Hot-reload prompts from disk without restarting the application."""
        self._load()
        logger.info(f"Prompts reloaded from '{self._path}'")

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self._path}")
        with open(self._path, encoding="utf-8") as fh:
            self._raw = yaml.safe_load(fh)
        self._registry = self._raw.get("prompts", {})
        logger.debug(
            f"Loaded {len(self._registry)} prompts "
            f"(schema_version={self._raw.get('schema_version', '?')})"
        )

    @staticmethod
    def _extract_variables(template: str) -> list[str]:
        import re
        return re.findall(r"\{(\w+)\}", template)
