"""Matbench extension hooks.

The prototype intentionally keeps Matbench optional. A future implementation can add task
adapters that expose the same dataset, split, metric, and prediction-export interfaces used
by the simple CSV workflow.
"""

from __future__ import annotations


class MatbenchAdapterNotImplemented(NotImplementedError):
    """Raised when Matbench integration is requested before implementation."""
