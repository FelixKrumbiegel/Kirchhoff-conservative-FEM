"""Implements the nonlinear Kirchhoff theta scheme.
"""
from dataclasses import dataclass

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from numpy.typing import NDArray
import dolfinx as dfx
import dolfinx.fem.petsc
import ufl

from .domain import Domain
from .data import HyperbolicThetaMean
from .space_discretization import SpaceDiscretization
from .kirchhoff_data import KirchhoffConfig, KirchhoffData
from .linear_system import LinearSystem, WaveSystem
from .functionals import QuadraticForm, gradient_seminorm, KirchhoffEnergy


@dataclass(frozen=True)
class StepReport:
    """What one time step cost.

    Attributes:
        picard -- Nonlinear iterations taken.
        gamma -- The nonlinear coefficient at the end of the step.
        converged -- Whether the Picard tolerance was reached.
    """
    picard: int
    gamma: float
    converged: bool


class NonlinearKirchhoffThetaScheme:
    """Class that implements a Kirchhoff solver. It initializes the
    system and solves the PDE using Picard iteration in every time step.
    Implements a step() method that makes one time step.
    """

    def __init__(
        self,
        config: KirchhoffConfig,
        data: KirchhoffData,
        domain: Domain,
        space: SpaceDiscretization,
        *,
        picard_tolerance: float = 1e-10,
        picard_max_iterations: int = 10,
        ksp_rtol: float = 1e-10,
        preconditioner: str = 'amg'
    ) -> None:
        # assert config.gamma != 0., \
        #     'If gamma == 0, the scheme reduces to a linear scheme. '\
        #     + 'This is beyond the scope of this implementation.'
        assert config.theta != 0., \
            'If theta == 0, the scheme is nonlinear only on the '\
            + 'right-hand side. This is beyond the scope of this '\
            + 'implementation'
        self._config = config
        self._data = data
        self._domain = domain
        self._space = space
        self._picard_tolerance = picard_tolerance
        self._picard_max_iterations = picard_max_iterations

        function_space = self._space.function_space
        v = ufl.TestFunction(function_space)

        # u0, u1 are the solution at the last time steps
        self._u0: dfx.fem.Function = dfx.fem.Function(function_space)
        self._u1: dfx.fem.Function = dfx.fem.Function(function_space)
        # uh is solution at the next time step
        self._uh: dfx.fem.Function = dfx.fem.Function(function_space)
        # dtuh is a placeholder for the discrete time derivative at the
        # current step
        self._dtuh: dfx.fem.Function = dfx.fem.Function(function_space)
        # vh is the artificial function we solve for
        self._vh: dfx.fem.Function = dfx.fem.Function(function_space)
        # uh_dirichlet is the (Dirichlet) boundary condition
        self._uh_dirichlet: dfx.fem.Function = dfx.fem.Function(
            function_space)

        self._domain.set_boundary_dofs(function_space)

        data.u_dirichlet.set_fem_evaluation(function_space)
        self._uh_dirichlet.interpolate(
            self._data.u_dirichlet.fem_evaluation)
        self._bc = dfx.fem.dirichletbc(
            self._uh_dirichlet, self._domain.boundary_dofs)

        # the following class calculates the seminorm inside the
        # nonlinearity without assembly
        self._seminorm: QuadraticForm = gradient_seminorm(space)

        # solve for the first step
        self._initial_step()

        # Initializing the solution at the next time step used in the
        # nonlinearity of the Picard iteration as the solution at the
        # previous time step.
        self._uh.x.array[:] = self._u1.x.array
        self._uh.x.scatter_forward()
        # Initializing the iterate of the Picard iteration as the
        # solution at the previous time step. It is used for the warm
        # start of the cg-solver only.
        self._vh.x.array[:] = self._u1.x.array
        self._vh.x.scatter_forward()

        # right-hand side assembly
        tau = dfx.fem.Constant(self._domain.mesh, self._config.tau)
        theta = dfx.fem.Constant(self._domain.mesh, self._config.theta)
        self._source_mean = HyperbolicThetaMean(
            self._config.eta, self._config.tau, self._data.source)
        self._rhs: dfx.fem.Form = dfx.fem.form(
            tau * tau * theta
            * self._source_mean.evaluation * v * ufl.dx
            + self._u1 * v * ufl.dx
        )

        # compute the value of the nonlinearity
        self._gamma_value: float = self._compute_gamma(
            self._data.alpha.value, self._data.beta.value,
            self._fixed_seminorm())

        # create right-hand side vectors
        self._b: PETSc.Vec = self._space.create_vector()
        self._b_clean: PETSc.Vec = self._space.create_vector()

        # create the system matrix
        self._system = WaveSystem(
            space, [self._bc],
            coefficient=self._scaled_coefficient(self._gamma_value),
            preconditioner=preconditioner,
            rtol=ksp_rtol,
            options_prefix='kirchhoff_')

    @property
    def uh(self) -> dfx.fem.Function:
        """Solution at the current time step.

        Returns:
            dfx.fem.Function -- Solution at the current time step.
        """
        return self._uh

    @property
    def dtuh(self) -> dfx.fem.Function:
        """Discrete time derivative at the current time step.

        Returns:
            dfx.fem.Function -- Backward difference of the solution.
        """
        self._dtuh.x.array[:] = (
            self._u1.x.array
            - self._u0.x.array
        ) / self._config.tau
        self._dtuh.x.scatter_forward()
        return self._dtuh

    @property
    def gamma(self) -> float:
        """The nonlinear coefficient of the current step.

        Returns:
            float -- alpha + beta * ||grad u||^2, theta-averaged.
        """
        return self._gamma_value

    def energy(self, functional: KirchhoffEnergy) -> float:
        """Evaluates an energy functional between the last two steps.

        Arguments:
            functional -- The discrete energy.

        Returns:
            float -- The energy between u^{n} and u^{n-1}.
        """
        return functional(self._u1, self._u0,
                          alpha=self._data.alpha.value,
                          beta=self._data.beta.value)

    def step(self) -> StepReport:
        """Executes one time step using Picard iteration.

        Returns:
            StepReport -- Number of Picard iterations taken, so a
            caller can see whether the loop is converging.
        """
        # updating the boundary conditions
        if not self._data.u_dirichlet.is_constant:
            self._data.u_dirichlet.shift(self._config.tau)
            self._uh_dirichlet.interpolate(
                self._data.u_dirichlet.fem_evaluation
            )
        # updating alpha and beta
        self._data.alpha.shift(self._config.tau)
        self._data.beta.shift(self._config.tau)

        # The lifting of the Dirichlet data depends on the boundary
        # values but not on the coefficient, so it is recomputed once
        # per step and reused by every Picard iteration.
        self._system.update_lifting()

        # updating the source mean and rhs vector
        self._source_mean.shift()
        with self._b.localForm() as loc_b:
            loc_b.set(0)
        dolfinx.fem.petsc.assemble_vector(self._b, self._rhs)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
        self._b.copy(result=self._b_clean)

        # fixed values for the nonlinearity during the Picard iteration
        alpha = self._data.alpha.value
        beta = self._data.beta.value
        fixed = self._fixed_seminorm()

        iterations = self._picard_max_iterations
        converged = False
        for iteration in range(self._picard_max_iterations):
            gamma_old = self._gamma_value
            previous: NDArray[np.float64] = self._uh.x.array.copy()
            scale = max(self._domain.comm.allreduce(
                np.max(np.abs(previous)), MPI.MAX
            ), 1.)

            self._gamma_value = self._compute_gamma(alpha, beta, fixed)

            self._system.set_coefficient(
                self._scaled_coefficient(self._gamma_value))

            self._b_clean.copy(result=self._b)
            self._system.apply_bcs(self._b)
            self._system.solve(self._b, self._vh.x.petsc_vec)
            self._vh.x.scatter_forward()

            self._uh.x.array[:] = (
                self._vh.x.array
                - (1 - 2*self._config.theta) * self._u1.x.array
                - self._config.theta * self._u0.x.array
            ) / self._config.theta
            self._uh.x.scatter_forward()

            change_gamma = (abs(gamma_old - self._gamma_value)
                            / max(abs(self._gamma_value), 1.))
            change_u = self._domain.comm.allreduce(
                np.max(np.abs((self._uh.x.array - previous) / scale)),
                MPI.MAX
            )
            if max(change_gamma, change_u) < self._picard_tolerance:
                iterations = iteration + 1
                converged = True
                break
        else:
            if self._domain.comm.rank == 0:
                print(f'Picard iteration did not converge within '
                      f'{self._picard_max_iterations} iterations.')

        self._u0.x.array[:] = self._u1.x.array
        self._u0.x.scatter_forward()
        self._u1.x.array[:] = self._uh.x.array
        self._u1.x.scatter_forward()
        return StepReport(
            picard=iterations, gamma=self._gamma_value, converged=converged)

    def destroy(self) -> None:
        """Destroys the PETSc vectors and matrices."""
        self._b.destroy()
        self._b_clean.destroy()
        self._system.destroy()
        self._seminorm.destroy()

    # ------------------------------------------------------------------
    # the nonlinear coefficient
    # ------------------------------------------------------------------
    def _fixed_seminorm(self) -> float:
        """The part of the theta-averaged seminorm that u0 and u1 fix.

        Returns:
            float -- The contribution of the two past time steps.
        """
        gamma = self._config.gamma
        return ((1. - 2.*gamma) * self._seminorm(self._u1)
                + gamma * self._seminorm(self._u0))

    def _compute_gamma(
        self,
        alpha: float,
        beta: float,
        fixed: float
    ) -> float:
        """Gamma = alpha + beta * theta-averaged ||grad u||^2.

        Arguments:
            alpha, beta -- The coefficients at the current time.
            fixed -- The contribution of u0 and u1, from
                `_fixed_seminorm`.

        Returns:
            float -- The nonlinear coefficient.
        """
        return alpha + beta * (
            self._config.gamma * self._seminorm(self._uh) + fixed)

    def _scaled_coefficient(self, gamma_value: float) -> float:
        """The coefficient the system matrix is built with.

        The scheme solves (M + theta tau^2 Gamma K) v = b.

        Arguments:
            gamma_value -- The nonlinear coefficient.

        Returns:
            float -- theta * tau^2 * Gamma.
        """
        return self._config.theta * self._config.tau**2 * gamma_value

    # ------------------------------------------------------------------
    # the starting values
    # ------------------------------------------------------------------
    def _initial_step(self) -> None:
        """Computes u_h^0 and u_h^1 from the Taylor start using the Ritz
        projection.
        """
        space = self._space
        mesh = self._domain.mesh
        v = ufl.TestFunction(space.function_space)
        dx = ufl.Measure('dx', domain=mesh)
        tau = self._config.tau

        # A zero prefactor collapses a form to ufl.Zero. Adding 0*v*dx
        # keeps the form linear in v.
        zero = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.))
        keep_rank1 = zero * v * dx

        projector = LinearSystem(
            space.assemble(space.stiffness_form, [self._bc], diag=1.),
            options_prefix='ritz_')

        u0_expression = self._data.u0.evaluation
        self._project(
            projector,
            ufl.dot(ufl.grad(u0_expression), ufl.grad(v)) * dx + keep_rank1,
            self._u0)

        seminorm_u0 = self._seminorm(self._u0)
        q0 = (self._data.alpha.value
              + self._data.beta.value * seminorm_u0)

        rhs = (
            tau * ufl.dot(ufl.grad(self._data.v0.evaluation),
                          ufl.grad(v)) * dx
            + tau * tau / 2. * ufl.dot(
                ufl.grad(self._data.source.evaluation), ufl.grad(v)) * dx
            + keep_rank1
        )
        if seminorm_u0 > 0.:
            # This method catches when u_0 = 0, which results in a
            # reduced ufl form as before.
            rhs = rhs + (
                ufl.dot(ufl.grad(u0_expression), ufl.grad(v)) * dx
                - tau * tau / 2. * q0 * ufl.dot(
                    ufl.grad(-ufl.div(ufl.grad(u0_expression))),
                    ufl.grad(v)) * dx
            )
        self._project(projector, rhs, self._u1)

        projector.destroy()

    def _project(
        self,
        projector: LinearSystem,
        rhs_form,
        result: dfx.fem.Function
    ) -> None:
        """Solves one Ritz projection into `result`.

        Arguments:
            projector -- The constrained stiffness system.
            rhs_form -- The linear form of the right-hand side.
            result -- Function to write the projection into.
        """
        vector = self._space.create_vector()
        with vector.localForm() as local:
            local.set(0.)
        dolfinx.fem.petsc.assemble_vector(vector, dfx.fem.form(rhs_form))
        dolfinx.fem.petsc.apply_lifting(
            vector, [self._space.stiffness_form], [[self._bc]])
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                           mode=PETSc.ScatterMode.REVERSE)
        dfx.fem.set_bc(vector, [self._bc])

        projector.solve(vector, result.x.petsc_vec)
        result.x.scatter_forward()
        vector.destroy()
