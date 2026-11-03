"""Implements data and function classes to use for FEM with dolfinx.
"""
from __future__ import annotations
import copy
from collections.abc import Callable
from dataclasses import dataclass

import dolfinx as dfx
from ufl.core.expr import Expr


@dataclass(frozen=True, kw_only=True)
class Data:
    """Class that holds the data functions for the PDE.

    Attributes:
        source -- Right-hand-side source term.
        u_dirichlet -- Boundary condition function.
    """
    source: Function
    u_dirichlet: Function


@dataclass(frozen=True, kw_only=True)
class TimeData(Data):
    """Class that holds the data functions for the PDE.

    Attributes:
        source -- Right-hand side source term.
        u_dirichlet -- Boundary condition function.
        u0 -- Initial condition of the solution.
        v0 -- Initial velocity du/dt(0).
    """
    u0: Function
    v0: Function

    def __post_init__(self) -> None:
        if not self.u0.is_constant:
            raise ValueError('u0 must be constant in time (is_constant=True).')
        if not self.v0.is_constant:
            raise ValueError('v0 must be constant in time (is_constant=True).')


class Function:
    """Maps a dfx.fem.Constant time value to a UFL expression, and
    keeps track of shifting that time value forward. Needs to be
    initialized for a dfx.mesh.Mesh.
    """

    def __init__(
        self,
        expression: Callable[[dfx.mesh.Mesh, dfx.fem.Constant], Expr],
        mesh: dfx.mesh.Mesh,
        *,
        t: float = 0.,
        is_constant: bool = False
    ) -> None:
        self._expression = expression
        self._mesh = mesh
        self._is_constant = is_constant
        self._t = dfx.fem.Constant(mesh, t)
        self._evaluation = expression(mesh, self._t)

        self._fem_evaluation: dfx.fem.Expression | None = None

    def __copy__(self) -> Function:
        return Function._from_copy(self)

    @property
    def is_constant(self) -> bool:
        """Returns whether the function is constant in time.

        Returns:
            bool -- Whether the function is constant.
        """
        return self._is_constant

    @property
    def evaluation(self) -> Expr:
        """Returns the function evaluated at the current time step as an
        ufl expression.

        Returns:
            Expr -- Function evaluated at the current time step as an
            ufl expression.
        """
        return self._evaluation

    @property
    def fem_evaluation(self) -> dfx.fem.Expression:
        """Returns the function as an dfx.fem.Expression that can be 
        interpolated.

        Raises:
            RuntimeError: If set_fem_evaluation has not been called.

        Returns:
            dfx.fem.Expression -- Expression that can be interpolated.
        """
        if self._fem_evaluation is None:
            raise RuntimeError('Fem Expression has not been set.')
        return self._fem_evaluation

    def set_fem_evaluation(
        self,
        function_space: dfx.fem.FunctionSpace
    ) -> None:
        """Sets a dfx.fem.Expression that can be interpolated. May be
        used for boundary or initial conditions. Note that interpolating
        right-hand sides may yield worse errors.
        """
        self._fem_evaluation = dfx.fem.Expression(
            self._evaluation,
            function_space.element.interpolation_points
        )

    def shift(
        self,
        tau: float
    ) -> None:
        """Shifts the time by tau.

        Arguments:
            tau (float) -- The time step size.
        """
        if not self._is_constant:
            self._t.value += tau

    def constant(
        self,
        value: float
    ) -> dfx.fem.Constant:
        """Creates a dfx.fem.Constant from value on the same mesh as
        self. Mainly used for (weighted) finite differences of Function
        types.

        Arguments:
            value -- Value to be casted to dfx.fem.Constant.

        Returns:
            dfx.fem.Constant that holds value.
        """
        return dfx.fem.Constant(self._mesh, value)

    @classmethod
    def _from_copy(cls, other: Function) -> Function:
        new = cls.__new__(cls)
        new._expression = other._expression
        new._mesh = other._mesh
        new._is_constant = other._is_constant
        new._t = dfx.fem.Constant(
            other._mesh,
            float(other._t.value)
        )
        new._evaluation = other._expression(
            other._mesh, new._t
        )
        new._fem_evaluation = None
        return new


class ScalarFunction:
    """Maps a time value to a float. Represents a function that is 
    constant in space.
    """

    def __init__(
        self,
        expression: Callable[[float], float],
        *,
        t: float = 0.,
        is_constant: bool = False
    ) -> None:
        self._expression = expression
        self._is_constant = is_constant
        self._t = t
        self._value = expression(self._t)

    def __copy__(self) -> ScalarFunction:
        return ScalarFunction._from_copy(self)

    @property
    def is_constant(self) -> bool:
        """Returns whether the function is constant in time.

        Returns:
            bool -- Whether the function is constant.
        """
        return self._is_constant

    @property
    def value(self) -> float:
        """Returns the function evaluated at the current time step.

        Returns:
            float -- Function evaluated at the current time step.
        """
        return self._value

    def shift(
        self,
        tau: float
    ) -> None:
        """Shifts the time by tau.

        Arguments:
            tau (float) -- The time step size.
        """
        if self._is_constant:
            return
        self._t += tau
        self._value = self._expression(self._t)

    @classmethod
    def _from_copy(cls, other: ScalarFunction) -> ScalarFunction:
        return cls(
            other._expression, t=other._t, is_constant=other._is_constant
        )


class HyperbolicThetaMean:
    """Implements the three-term theta mean of a function.

    theta * f(t - tau)
    + (1 - 2*theta) * f(t)
    + theta * f(t + tau)
    """

    def __init__(
        self,
        theta: float,
        tau: float,
        function: Function
    ) -> None:
        self._theta: dfx.fem.Constant = function.constant(theta)
        self._center: dfx.fem.Constant = function.constant(1. - 2.*theta)
        self._tau: float = tau

        self._current: Function = copy.copy(function)

        self._previous: Function = copy.copy(function)
        self._previous.shift(-tau)
        self._next: Function = copy.copy(function)
        self._next.shift(tau)

    @property
    def evaluation(self) -> Expr:
        """Returns the theta mean as an ufl expression.

        Returns:
            Expr -- Theta mean as an ufl expression.
        """
        return (
            self._theta * self._previous.evaluation
            + self._center * self._current.evaluation
            + self._theta * self._next.evaluation
        )

    def shift(self) -> None:
        """Shifts the mean by the time step size.
        """
        self._current.shift(self._tau)
        self._previous.shift(self._tau)
        self._next.shift(self._tau)


class ParabolicThetaMean:
    """Implements the three-term theta mean of a function.

    (1 - theta) * f(t)
    + theta * f(t + tau)
    """

    def __init__(
        self,
        theta: float,
        tau: float,
        function: Function
    ) -> None:
        self._theta: dfx.fem.Constant = function.constant(theta)
        self._center: dfx.fem.Constant = function.constant(1. - theta)
        self._tau: float = tau

        self._current: Function = function

        self._next: Function = copy.copy(function)
        self._next.shift(tau)

    @property
    def evaluation(self) -> Expr:
        """Returns the theta mean as an ufl expression.

        Returns:
            Expr -- Theta mean as an ufl expression.
        """
        return (
            self._center * self._current.evaluation
            + self._theta * self._next.evaluation
        )

    def shift(self) -> None:
        """Shifts the mean by the time step size.
        """
        self._current.shift(self._tau)
        self._next.shift(self._tau)
