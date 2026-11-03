"""Implements the space discretization with the underlying function
space on a given mesh. It implements the assembly of the mass and
stiffness matrices subject to optional given boundary conditions.
"""
from petsc4py import PETSc
import dolfinx as dfx
import dolfinx.fem.petsc
import dolfinx.la.petsc
import ufl
from dolfinx.fem.forms import Form


class SpaceDiscretization:
    """Class implementing the space discretization. 
    Holds the function space and implements the mass and stiffness 
    forms.
    """

    def __init__(
        self,
        function_space: dfx.fem.FunctionSpace,
    ) -> None:
        self._function_space: dfx.fem.FunctionSpace = function_space

        u = ufl.TrialFunction(function_space)
        v = ufl.TestFunction(function_space)
        self._mass_form: Form = dfx.fem.form(u * v * ufl.dx)
        self._stiffness_form: Form = dfx.fem.form(
            ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)

    @property
    def function_space(self) -> dfx.fem.FunctionSpace:
        """Returns the underlying functions space.

        Returns:
            dfx.fem.FunctionSpace -- Underlying functions space.
        """
        return self._function_space

    @property
    def mass_form(self) -> Form:
        """Returns the mass form.

        Returns:
            Form -- Mass form.
        """
        return self._mass_form

    @property
    def stiffness_form(self) -> Form:
        """Returns the stiffness form.

        Returns:
            Form -- Stiffness form.
        """
        return self._stiffness_form

    def assemble(
        self,
        form: Form,
        bcs: list[dfx.fem.DirichletBC] | None = None,
        diag: float = 1.
    ) -> PETSc.Mat:
        """Implements the assemble routine for a specific form.

        Arguments:
            form -- Form to be compiled into a matrix

        Keyword Arguments:
            bcs -- Optional boundary conditions. (default: {None})
            diag -- Diagonal for the boundary conditions. (default: {1.})

        Returns:
            PETSc.Mat -- Matrix assembled from the form.
        """
        matrix = dolfinx.fem.petsc.assemble_matrix(
            form, bcs=[] if bcs is None else list(bcs), diag=diag)
        matrix.assemble()
        return matrix

    def create_vector(self) -> PETSc.Vec:
        """Returns a new ghosted vector laid out for this space.

        Suitable for assembling a right hand side into, and for
        apply_lifting, which writes into the ghost entries.

        Returns:
            PETSc.Vec -- A new vector, not zeroed.
        """
        return dolfinx.la.petsc.create_vector(
            [(self._function_space.dofmap.index_map,
             self._function_space.dofmap.index_map_bs)])
