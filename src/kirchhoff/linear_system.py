"""Implements a linear system, ie, a matrix together with the Krylov
solver configured for it.
"""
from __future__ import annotations

from petsc4py import PETSc
import dolfinx as dfx
import dolfinx.fem.petsc

from .space_discretization import SpaceDiscretization


class LinearSystem:
    """A matrix with a Krylov solver attached.
    """

    def __init__(
        self,
        matrix: PETSc.Mat,
        *,
        options_prefix: str = '',
        ksp_type: str = 'cg',
        pc_type: str = 'hypre',
        rtol: float = 1e-10,
        atol: float = 1e-14,
        max_it: int = 1000,
        initial_guess_nonzero: bool = True,
        preconditioner_matrix: PETSc.Mat | None = None,
    ) -> None:
        self._matrix: PETSc.Mat = matrix
        self._total_iterations: int = 0

        ksp = PETSc.KSP().create(matrix.getComm())
        ksp.setOperators(matrix, preconditioner_matrix or matrix)
        ksp.setOptionsPrefix(options_prefix)
        ksp.setType(ksp_type)

        pc = ksp.getPC()
        pc.setType(pc_type)
        if pc_type == 'hypre':
            pc.setHYPREType('boomeramg')

        ksp.setTolerances(rtol=rtol, atol=atol, divtol=1e10, max_it=max_it)
        ksp.setInitialGuessNonzero(initial_guess_nonzero)
        ksp.setFromOptions()
        self._ksp: PETSc.KSP = ksp

    @property
    def matrix(self) -> PETSc.Mat:
        """Returns the system matrix.

        Returns:
            PETSc.Mat -- The matrix the solver operates on.
        """
        return self._matrix

    @property
    def solver(self) -> PETSc.KSP:
        """Returns the Krylov solver.

        Returns:
            PETSc.KSP -- The solver, already holding the matrix.
        """
        return self._ksp

    @property
    def total_iterations(self) -> int:
        """Summed Krylov iterations over every (time step) solve.

        Returns:
            int -- The running total.
        """
        return self._total_iterations

    def solve(self, rhs: PETSc.Vec, solution: PETSc.Vec) -> int:
        """Solves A x = b.

        Arguments:
            rhs -- Right hand side.
            solution -- Vector to write the solution into; its current
                contents are used as the initial guess.

        Raises:
            RuntimeError: If the Krylov solver did not converge.

        Returns:
            int -- Number of iterations taken.
        """
        self._ksp.solve(rhs, solution)
        reason = self._ksp.getConvergedReason()
        iterations = self._ksp.getIterationNumber()
        if reason < 0:
            raise RuntimeError(
                f'The linear solver failed with reason {reason} after '
                f'{iterations} iterations.')
        self._total_iterations += iterations
        return iterations

    def reuse_preconditioner(self, reuse: bool = True) -> None:
        """Keeps the current preconditioner when the matrix changes.

        Arguments:
            reuse -- Whether to keep the existing preconditioner.
        """
        self._ksp.getPC().setReusePreconditioner(reuse)

    def destroy(self) -> None:
        """Destroys the solver and the matrix."""
        self._ksp.destroy()
        self._matrix.destroy()


PRECONDITIONERS = ('amg', 'amg_reuse', 'mass', 'jacobi')


class WaveSystem(LinearSystem):
    """The operator M + c K with Dirichlet conditions, and its solver.

    The constrained mass and stiffness matrices are assembled once. Each
    time the coefficient changes, A is rewritten as a copy of M plus c
    times K. It shall be noted that the coefficient has to be spatially
    independent.
    """

    def __init__(
        self,
        space: SpaceDiscretization,
        bcs: list[dfx.fem.DirichletBC],
        *,
        coefficient: float = 0.,
        preconditioner: str = 'amg',
        **ksp_options
    ) -> None:
        """
        Arguments:
            space -- The space discretization supplying the forms.
            bcs -- The Dirichlet conditions of the system.
            coefficient -- Initial value of c.
            preconditioner -- One of PRECONDITIONERS.
            ksp_options -- Forwarded to LinearSystem; see there for the
                solver and preconditioner settings.

        Raises:
            ValueError: If the preconditioner is not recognized.
        """
        if preconditioner not in PRECONDITIONERS:
            raise ValueError(
                f'Unknown preconditioner {preconditioner!r}; '
                f'expected one of {PRECONDITIONERS}.')
        self._preconditioner = preconditioner
        self._preconditioner_ready = False
        self._space = space
        self._bcs = list(bcs)
        self._coefficient = 0.

        self._mass: PETSc.Mat = space.assemble(
            space.mass_form, self._bcs, diag=1.)
        self._stiffness: PETSc.Mat = space.assemble(
            space.stiffness_form, self._bcs, diag=0.)
        self._lift_mass: PETSc.Vec = space.create_vector()
        self._lift_stiffness: PETSc.Vec = space.create_vector()

        super().__init__(
            self._mass.copy(),
            preconditioner_matrix=(
                self._mass if preconditioner == 'mass' else None),
            pc_type='jacobi' if preconditioner == 'jacobi' else 'hypre',
            **ksp_options)

        self.update_lifting()
        self.set_coefficient(coefficient)

    @property
    def coefficient(self) -> float:
        """Returns the coefficient the matrix currently holds.

        Returns:
            float -- The coefficient c of A = M + c K.
        """
        return self._coefficient

    def set_coefficient(self, value: float) -> None:
        """Rewrites the system matrix as M + value * K.

        Arguments:
            value -- The new coefficient.
        """
        self._coefficient = float(value)
        self._mass.copy(self._matrix,
                        PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        self._matrix.axpy(self._coefficient, self._stiffness,
                          PETSc.Mat.Structure.SAME_NONZERO_PATTERN)

    def update_lifting(self) -> None:
        """Recomputes the Dirichlet lifting of each operator.

        Call once per time step, after the function carrying the
        boundary values has been interpolated at the new time.
        """
        for vector, form in ((self._lift_mass, self._space.mass_form),
                             (self._lift_stiffness,
                              self._space.stiffness_form)):
            with vector.localForm() as local:
                local.set(0.)
            dolfinx.fem.petsc.apply_lifting(vector, [form], [self._bcs])
            vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                               mode=PETSc.ScatterMode.REVERSE)

    def apply_bcs(self, rhs: PETSc.Vec) -> None:
        """Applies the lifting and the boundary values to a right-hand
        side.

        Arguments:
            rhs -- The assembled right-hand side, modified in place.
        """
        rhs.axpy(1., self._lift_mass)
        rhs.axpy(self._coefficient, self._lift_stiffness)
        dfx.fem.set_bc(rhs, self._bcs)

    def solve(self, rhs: PETSc.Vec, solution: PETSc.Vec) -> int:
        """Solves A x = b, freezing the hierarchy if asked to.

        Arguments:
            rhs -- Right hand side, with boundary conditions applied.
            solution -- Vector to write the solution into.

        Returns:
            int -- Number of iterations taken.
        """
        iterations = super().solve(rhs, solution)
        if (self._preconditioner == 'amg_reuse'
                and not self._preconditioner_ready):
            self.reuse_preconditioner(True)
            self._preconditioner_ready = True
        return iterations

    def destroy(self) -> None:
        """Destroys this object's PETSc objects, then the base's.
        """
        self._mass.destroy()
        self._stiffness.destroy()
        self._lift_mass.destroy()
        self._lift_stiffness.destroy()
        super().destroy()
