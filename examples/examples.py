"""Test cases for the Kirchhoff equation

    u_tt - Gamma(t) * Laplace(u) = f,
    Gamma(t) = alpha(t) + beta(t)*||grad u||^2 .

Each example is one function returning an `Example`.

Each example is computed as follows. It is assumed that the exact
solution is
    u(t, x, y) = g(t) sin(lam pi x) sin(lam pi y).
Then, the right-hand side is given by
    f(t, x, y) = [ g''(t) + mu (alpha(t) + beta(t) G g(t)^2) g(t) ] phi(x, y),
where mu = 2 (lam pi)^2 is the eigenvalue of phi and G = (lam pi)^2 / 2
is the squared H^1-seminorm.


Adding an example
-----------------
Copy an existing builder and edit it. Everything defining the case lives
in that one function. Each builder opens by binding the three quantities
above to short local names, so the source term reads like the formula:

    def _example_5(lam: int, final_time: float) -> Example:
        \"\"\"u = ...,  alpha(t) = 1 + 4t,  beta(t) = 0\"\"\"
        mu, cap_g = eigenvalue(lam), grad_seminorm(lam)
        phi = eigenfunction(lam)

        def alpha(t: float) -> float:
            return ...

        def beta(t: float) -> float:
            return ...

        def source(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
            g = ...
            ddg = ...
            a = ...          # alpha(t), in UFL
            b = ...          # beta(t), in UFL
            return (ddg + mu * (a + b * cap_g * g * g) * g) * phi(mesh)

        def solution(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
            return ...

        def u0(mesh: dfx.mesh.Mesh, _t: dfx.fem.Constant) -> Expr:
            return ...

        def v0(mesh: dfx.mesh.Mesh, _t: dfx.fem.Constant) -> Expr:
            return ...

        return Example(name='5', source=source, solution=solution,
                       u0=u0, v0=v0, alpha=alpha, beta=beta,
                       autonomous=False, final_time=final_time)

then add one `case` to `example()` and the number to `ALL`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import dolfinx as dfx
import ufl
from ufl.core.expr import Expr

from kirchhoff.data import Function, ScalarFunction
from kirchhoff.kirchhoff_data import KirchhoffData


def eigenfunction(lam: int) -> Callable[[dfx.mesh.Mesh], Expr]:
    """phi = sin(lam*pi*x) sin(lam*pi*y).

    Returns the eigenfunction of the Dirichlet Laplacian on the unit
    square as a ufl expression. It has the eigenvalue
        mu = 2 * (lam * pi)^2
    and norm
        ||grad phi||^2 = (lam * pi)^2 / 2.
    """
    def _phi(mesh: dfx.mesh.Mesh) -> Expr:
        x = ufl.SpatialCoordinate(mesh)
        return (ufl.sin(lam * ufl.pi * x[0])
                * ufl.sin(lam * ufl.pi * x[1]))
    return _phi


def eigenvalue(lam: int) -> float:
    """mu = 2 * (lam * pi)^2.

    Returns the eigenvalue of the corresponding eigenfunction phi on the
    unit square.
    """
    return 2. * lam**2 * ufl.pi**2


def grad_seminorm(lam: int) -> float:
    """G = ||grad phi||^2 = (lam * pi)^2 / 2.

    Returns the squared H^1-seminorm of the corresponding eigenfunction
    phi on the unit square.
    """
    return lam**2 * ufl.pi**2 / 2.


def return_constant_expression(
    value: float
) -> Callable[[dfx.mesh.Mesh, dfx.fem.Constant], Expr]:
    """Returns a function that ignores the mesh and time and returns a
    constant expression.
    """
    def _constant_expression(
        mesh: dfx.mesh.Mesh,
        _t: dfx.fem.Constant
    ) -> Expr:
        return dfx.fem.Constant(mesh, dfx.default_scalar_type(value))
    return _constant_expression


# ----------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Example:
    """One test case: an exact solution, its data, and its coefficients.

    Attributes:
        name -- Label identifying the case.
        source -- The right-hand side f(t, x).
        solution -- The exact solution u(t, x).
        u0 -- u(0, x).
        v0 -- u_t(0, x).
        alpha, beta -- The coefficients of the model.
        dirichlet -- The boundary values. Zero unless given.
        autonomous -- Whether alpha and beta are independent of t.
        source_autonomous -- Whether the source is autonomous.
        dirichlet_autonomous -- Whether the boundary values are autonomous.
        final_time -- The time the run ends at.
    """
    name: str
    source: Callable[[dfx.mesh.Mesh, dfx.fem.Constant], Expr]
    solution: Callable[[dfx.mesh.Mesh, dfx.fem.Constant], Expr]
    u0: Callable[[dfx.mesh.Mesh, dfx.fem.Constant], Expr]
    v0: Callable[[dfx.mesh.Mesh, dfx.fem.Constant], Expr]
    alpha: Callable[[float], float]
    beta: Callable[[float], float]
    dirichlet: Callable[[dfx.mesh.Mesh, dfx.fem.Constant],
                        Expr] = return_constant_expression(0.)
    autonomous: bool = True
    source_autonomous: bool = False
    dirichlet_autonomous: bool = True
    final_time: float = ufl.pi / 2

    def data(self, mesh: dfx.mesh.Mesh) -> KirchhoffData:
        """Assembles everything the solver needs."""
        return KirchhoffData(
            source=Function(self.source, mesh,
                            is_constant=self.source_autonomous),
            u_dirichlet=Function(self.dirichlet, mesh,
                                 is_constant=self.dirichlet_autonomous),
            u0=Function(self.u0, mesh, is_constant=True),
            v0=Function(self.v0, mesh, is_constant=True),
            alpha=ScalarFunction(self.alpha,
                                 is_constant=self.autonomous),
            beta=ScalarFunction(self.beta,
                                is_constant=self.autonomous),
        )

    def exact(self, mesh: dfx.mesh.Mesh) -> Function:
        """The exact solution, for the error computation."""
        return Function(self.solution, mesh, is_constant=False)


# ----------------------------------------------------------------------
# The examples
# ----------------------------------------------------------------------
def _example_1(lam: int, final_time: float) -> Example:
    """Example 1.

        u     = sin(t) sin(lam pi x) sin(lam pi y)
        alpha = 1,   beta = 1

    With g = sin(t), g'' = -sin(t) = -g.
    """
    mu, cap_g = eigenvalue(lam), grad_seminorm(lam)
    phi = eigenfunction(lam)

    def alpha(t: float) -> float:
        return 1.

    def beta(t: float) -> float:
        return 1.

    def source(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        g = ufl.sin(t)
        ddg = -g
        # alpha = beta = 1
        return (
            (ddg + mu * (1. + cap_g * g * g) * g) * phi(mesh)
        )

    def solution(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        return ufl.sin(t) * phi(mesh)

    def v0(mesh: dfx.mesh.Mesh, _t: dfx.fem.Constant) -> Expr:
        # g'(0) = cos(0) = 1.
        return phi(mesh)

    return Example(
        name=f'1 (lam={lam})',
        source=source, solution=solution,
        u0=return_constant_expression(0.), v0=v0,
        alpha=alpha, beta=beta,
        autonomous=True, final_time=final_time
    )


def _example_2(lam: int, final_time: float) -> Example:
    """Example 2, nonautonomous.

        u     = exp(pi t) sin(lam pi x) sin(lam pi y)
        alpha = 1 + 4t,   beta = 2 - 4t

    With g = exp(pi t), g'' = pi^2 g.
    """
    mu, cap_g = eigenvalue(lam), grad_seminorm(lam)
    phi = eigenfunction(lam)

    def alpha(t: float) -> float:
        return 1. + 4. * t

    def beta(t: float) -> float:
        return 2. - 4. * t

    def source(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        g = ufl.exp(ufl.pi * t)
        ddg = ufl.pi**2 * g
        a = 1. + 4. * t           # alpha(t), in UFL
        b = 2. - 4. * t           # beta(t), in UFL
        return (
            (ddg + mu * (a + b * cap_g * g * g) * g) * phi(mesh)
        )

    def solution(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        return ufl.exp(ufl.pi * t) * phi(mesh)

    def u0(mesh: dfx.mesh.Mesh, _t: dfx.fem.Constant) -> Expr:
        # g(0) = 1.
        return phi(mesh)

    def v0(mesh: dfx.mesh.Mesh, _t: dfx.fem.Constant) -> Expr:
        # g'(0) = pi.
        return ufl.pi * phi(mesh)

    return Example(
        name=f'2 (lam={lam})',
        source=source, solution=solution, u0=u0, v0=v0,
        alpha=alpha, beta=beta,
        autonomous=False, final_time=final_time
    )


def _example_3(lam: int, lam1: int, final_time: float) -> Example:
    """Example 3.

        u     = (1 + cos(omega t)) sin(lam pi x) sin(lam pi y)
        alpha = 1,   beta = 1,
        omega = lam1 * pi / (2 * final_time)

    With g = 1 + cos(omega t) and g'' = -omega^2 cos(omega t).
    """
    mu, cap_g = eigenvalue(lam), grad_seminorm(lam)
    phi = eigenfunction(lam)
    omega = lam1 * ufl.pi / (2. * final_time)

    def alpha(t: float) -> float:
        return 1.

    def beta(t: float) -> float:
        return 1.

    def source(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        g = 1. + ufl.cos(omega * t)
        ddg = -omega**2 * ufl.cos(omega * t)
        # alpha = beta = 1
        return (
            (ddg + mu * (1. + cap_g * g * g) * g) * phi(mesh)
        )

    def solution(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        return (
            (1. + ufl.cos(omega * t)) * phi(mesh))

    def u0(mesh: dfx.mesh.Mesh, _t: dfx.fem.Constant) -> Expr:
        # g(0) = 2.
        return 2. * phi(mesh)

    return Example(
        name=f'3 (lam={lam}, lam1={lam1})',
        source=source, solution=solution,
        u0=u0, v0=return_constant_expression(0.),
        alpha=alpha, beta=beta,
        autonomous=True, final_time=final_time
    )


def _example_4(lam: int, final_time: float) -> Example:
    """Example 4, nonautonomous.

        u     = sin(t) sin(lam pi x) sin(lam pi y)
        alpha = 1 + t^2,   beta = 2 - sin(t)

    With g = sin(t), g'' = -g.
    """
    mu, cap_g = eigenvalue(lam), grad_seminorm(lam)
    phi = eigenfunction(lam)

    def alpha(t: float) -> float:
        return 1. + t * t

    def beta(t: float) -> float:
        return 2. - math.sin(t)

    def source(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        g = ufl.sin(t)
        ddg = -g
        a = 1. + t * t            # alpha(t), in UFL
        b = 2. - ufl.sin(t)       # beta(t), in UFL
        return (
            (ddg + mu * (a + b * cap_g * g * g) * g) * phi(mesh)
        )

    def solution(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        return ufl.sin(t) * phi(mesh)

    def v0(mesh: dfx.mesh.Mesh, _t: dfx.fem.Constant) -> Expr:
        # g'(0) = 1.
        return phi(mesh)

    return Example(
        name=f'4 (lam={lam})',
        source=source, solution=solution,
        u0=return_constant_expression(0.), v0=v0,
        alpha=alpha, beta=beta,
        autonomous=False, final_time=final_time
    )


def _example_5(lam: int, final_time: float) -> Example:
    """Example 5.

        u     = g(t) sin(lam pi x) sin(lam pi y)
        alpha = 1,   beta = 1

    With g(t) given by the solution of the Duffing-equation using that 
    the spatial part of u is a eigenvector of the Laplace operator. We 
    further have the following
        u(0)    = 0
        u_t(0)  = sin(lam pi x) sin(lam pi y)
        f       = 0
    This example does not contain the correct exact solution. It shows
    the energy conservation only and no convergence.
    """
    phi = eigenfunction(lam)

    def alpha(t: float) -> float:
        return 1.

    def beta(t: float) -> float:
        return 1.

    print('Note that for example 5, the exact solution provided is '
          + 'only a placeholder. Conervegence studies may be viewed '
          + 'with caution.')

    def solution(mesh: dfx.mesh.Mesh, t: dfx.fem.Constant) -> Expr:
        return ufl.sin(t) * phi(mesh)

    def v0(mesh: dfx.mesh.Mesh, _t: dfx.fem.Constant) -> Expr:
        return phi(mesh)

    return Example(
        name=f'5 (lam={lam})',
        source=return_constant_expression(0.), solution=solution,
        u0=return_constant_expression(0.), v0=v0,
        alpha=alpha, beta=beta,
        autonomous=True, final_time=final_time
    )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
ALL: tuple[int, ...] = (
    1, 11, 12, 13,
    21, 22, 23,
    311, 313, 315, 321, 323, 325, 331, 333, 335,
    41, 42, 43,
    5,
)


def example(ex_nr: int, final_time: float = ufl.pi/2) -> Example:
    """Returns the example with the given number.

    The last digit is lam, except in the 3xx family where the last digit
    selects the pulse frequency and the middle digit is lam.

    Raises:
        ValueError: If the example does not exist.
    """
    match ex_nr:
        case 1 | 11 | 12 | 13:
            return _example_1(ex_nr % 10, final_time)
        case 21 | 22 | 23:
            return _example_2(ex_nr % 10, final_time)
        case 311 | 313 | 315 | 321 | 323 | 325 | 331 | 333 | 335:
            return _example_3((ex_nr // 10) % 10, ex_nr % 10, final_time)
        case 41 | 42 | 43:
            return _example_4(ex_nr % 10, final_time)
        case 5:
            return _example_5(1, final_time)
        case _:
            raise ValueError(f'Example {ex_nr} does not exist.')


# ----------------------------------------------------------------------
# Compatibility shims for the existing call sites
# ----------------------------------------------------------------------
def get_data(
    ex_nr: int,
    mesh: dfx.mesh.Mesh,
    final_time: float = ufl.pi/2
) -> KirchhoffData:
    """Returns the data for the Kirchhoff PDE."""
    return example(ex_nr, final_time=final_time).data(mesh)


def get_solution(
    ex_nr: int,
    mesh: dfx.mesh.Mesh,
    final_time: float = ufl.pi/2
) -> Function:
    """Returns the exact solution of the Kirchhoff PDE."""
    return example(ex_nr, final_time=final_time).exact(mesh)
