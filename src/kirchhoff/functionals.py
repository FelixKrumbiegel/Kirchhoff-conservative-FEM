"""Implements functionals evaluated on discrete functions. This includes
discrete L^2-norm and H^1-seminorm, as well as an energy functional for
the Kirchhoff equation.
"""
from __future__ import annotations

from petsc4py import PETSc
import dolfinx as dfx
from dolfinx.fem.forms import Form

from .space_discretization import SpaceDiscretization


class QuadraticForm:
    """Evaluates v -> a(v, v) for a symmetric bilinear form a.
    """

    def __init__(
        self,
        space: SpaceDiscretization,
        form: Form
    ) -> None:
        """
        Arguments:
            space -- The space the functions live in.
            form -- A symmetric bilinear form on that space.
        """
        # The form is assembled without boundary conditions. The reason
        # is the fact that the (semi-)norms do not holds bcs.
        self._operator: PETSc.Mat = space.assemble(form)
        # Allocates the memory for the vector Av.
        self._work: PETSc.Vec = self._operator.createVecLeft()

    def __call__(self, u: dfx.fem.Function) -> float:
        """Returns a(u, u).

        Arguments:
            u -- Function in the space this was built from.

        Returns:
            float -- The value of the form.
        """
        vector = u.x.petsc_vec
        self._operator.mult(vector, self._work)
        return float(self._work.dot(vector))

    def destroy(self) -> None:
        """Destroys the PETSc objects."""
        self._work.destroy()
        self._operator.destroy()


def gradient_seminorm(space: SpaceDiscretization) -> QuadraticForm:
    """Returns the functional u -> ||grad u||^2 in L2.

    Arguments:
        space -- Underlying function space.

    Returns:
        QuadraticForm -- The gradient seminorm functional.
    """
    return QuadraticForm(space, space.stiffness_form)


def l2_norm_squared(space: SpaceDiscretization) -> QuadraticForm:
    """Returns the functional u -> ||u||^2 in L2.

    Arguments:
        space -- Underlying function space.

    Returns:
        QuadraticForm -- The l2-norm functional.
    """

    return QuadraticForm(space, space.mass_form)


class KirchhoffEnergy:
    """Discrete energy of the Kirchhoff PDE

        u_tt - (alpha + beta ||grad u||^2) Laplace(u) = f .

    For vanishing right-hand side and constant alpha = beta = const the
    discrete energy is given by

        E_{n} = 1/2 ||(u_{n} - u_{n-1})/tau||_M^2
                + alpha/4 (||u_{n}||_S^2 + ||u_{n-1}||_S^2)
                + beta/4 (||u_{n}||_S^2 * ||u_{n-1}||_S^2)
                + (beta gamma)/4 (||u_{n}||_S^2 - ||u_{n-1}||_S^2).
    """

    def __init__(
        self,
        space: SpaceDiscretization,
        tau: float,
        gamma: float,
        mass: QuadraticForm,
        stiffness: QuadraticForm
    ) -> None:
        """
        Arguments:
            space -- The space the functions live in.
            tau -- The time step size.
            mass -- The L2 norm functional.
            stiffness -- The gradient seminorm functional.

        For the Kirchhoff solver the gradient seminorm is required
        anyway, so passing the functional results in less matrices being
        stored. This may be changed
        """
        self._tau = float(tau)
        self._gamma = float(gamma)
        self._mass = mass
        self._stiffness = stiffness

        self._difference: dfx.fem.Function = dfx.fem.Function(
            space.function_space)

    def __call__(
        self,
        u_current: dfx.fem.Function,
        u_prev: dfx.fem.Function,
        *,
        alpha: float,
        beta: float
    ) -> float:
        """Returns the discrete energy between two consecutive steps.

        Arguments:
            u_current -- The solution at the later step.
            u_prev -- The solution at the earlier step.
            alpha, beta -- The coefficients, at the time between them.

        Returns:
            float -- The discrete energy.
        """
        later = u_current.x.petsc_vec
        earlier = u_prev.x.petsc_vec

        # u_{n} - u_{n-1}
        self._difference.x.petsc_vec.waxpy(-1., earlier, later)

        kinetic = .5 * self._mass(self._difference) / self._tau / self._tau

        seminorm_later = self._stiffness(u_current)
        seminorm_earlier = self._stiffness(u_prev)
        potential = .25 * (
            alpha * (seminorm_later + seminorm_earlier)
            + beta * (seminorm_later * seminorm_earlier
                      + self._gamma * (seminorm_later - seminorm_earlier)**2))
        return kinetic + potential
