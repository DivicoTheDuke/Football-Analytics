"""Adapters for external football data providers."""

from .statsbomb import normalise_statsbomb_events

__all__ = ["normalise_statsbomb_events"]
