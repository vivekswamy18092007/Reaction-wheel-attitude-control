"""
Common shape every controller in control/registry.REGISTRY must satisfy, so
sim/runner.py can drive any of them through one code path instead of an
if/elif per type. Adding a new controller (e.g. lqr.py going from stub to
real) means writing a class matching this Protocol and registering it in
registry.py -- no changes anywhere in sim/.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Controller(Protocol):

    Ki: Optional[np.ndarray]
    u_max: Optional[float]

    def compute_torque(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        q_target: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Body control torque [N m] for the current state.

        q, q_target: scalar-first quaternions (4,). omega: body rate (3,).
        dt: pass only when integral action (Ki) is active, and only if it
        matches the propagator's step -- see each controller's docstring.
        """
        ...

    def reset(self) -> None:
        """Clear any per-run state (sign latch, integral accumulator).
        Call before reusing one controller instance across multiple runs."""
        ...
