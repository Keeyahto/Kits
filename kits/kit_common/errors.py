from __future__ import annotations


class KitError(Exception):
    """Base exception for kits."""


class ConfigError(KitError):
    """Configuration related error."""


class ExternalServiceError(KitError):
    """External dependency error (network, API, etc.)."""


class ValidationError(KitError):
    """Validation error."""

