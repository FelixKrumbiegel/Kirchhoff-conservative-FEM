"""Implements classes that hold information about the time
discretization.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class Config:
    """Configuration class for spatial discretization.
    """
    nx: int
    ny: int


@dataclass(frozen=True, kw_only=True)
class ThetaConfig(Config):
    """Configuration class for spatial and temporal discretization.
    """
    ntime: int
    final_time: float
    theta: float
    eta: float
    tau: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'tau', self.final_time / self.ntime)
        assert self.theta <= 1/2 and self.theta >= 0., \
            'The scheme uses theta in [0, 1/2].'
        assert self.eta <= 1/2 and self.eta >= 0., \
            'The scheme uses eta in [0, 1/2].'
