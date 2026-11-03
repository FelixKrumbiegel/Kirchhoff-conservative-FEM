"""Implements everything related to the Kirchhoff PDE.
"""
from dataclasses import dataclass

from .config import ThetaConfig
from .data import TimeData, ScalarFunction


@dataclass(frozen=True, kw_only=True)
class KirchhoffConfig(ThetaConfig):
    """Configuration class for Kirchhoff equation.
    """
    gamma: float

    def __post_init__(self) -> None:
        super().__post_init__()
        # assert self.theta >= 1/4 or self.theta == 0., \
        #     'The scheme is unstable for theta < 1/4. Theta = 0. may be used.'
        assert self.gamma <= 1/2 and self.gamma >= 0., \
            'The scheme uses gamma in [0, 1/2].'


@dataclass(frozen=True, kw_only=True)
class KirchhoffData(TimeData):
    """Class that holds the data functions for the PDE.

    Attributes:
        source -- Right-hand-side source term.
        u_dirichlet -- Boundary condition function.
        u0 -- Initial condition of the solution.
        v0 -- Initial condition for the first time derivative of the
        solution.
        alpha -- Coefficient for the Kirchhoff equation.
        beta -- Coefficient for the Kirchhoff equation.
    """
    alpha: ScalarFunction
    beta: ScalarFunction
