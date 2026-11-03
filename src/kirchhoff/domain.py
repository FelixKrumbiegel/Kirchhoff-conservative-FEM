"""Implements the underlying computational domain: the mesh and the
boundary facets. It may also compute the boundary degrees of freedom for
a given function space defined on it.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import dolfinx as dfx
from mpi4py import MPI


class Domain:
    """Holds the underlying mesh and computes the cell connectivity and 
    boundary facets. Computes the boundary degrees of freedom for a given
    function space.
    """

    def __init__(
        self,
        mesh: dfx.mesh.Mesh
    ) -> None:
        self._mesh: dfx.mesh.Mesh = mesh

        tdim = mesh.topology.dim
        # to construct the boundary facets, the connectivity is required
        mesh.topology.create_connectivity(tdim - 1, tdim)
        self._boundary_facets: NDArray[np.int32] = dfx.mesh.exterior_facet_indices(
            mesh.topology)

        self._boundary_dofs: NDArray[np.int32] | None = None

    @property
    def mesh(self) -> dfx.mesh.Mesh:
        """Returns underlying mesh.

        Returns:
            dfx.mesh.Mesh -- Underlying mesh.
        """
        return self._mesh

    @property
    def comm(self) -> MPI.Intracomm:
        """Returns the MPI communicator associated with the mesh.

        Returns:
            MPI.Intracomm -- MPI communicator.
        """
        return self._mesh.comm

    @property
    def boundary_dofs(self) -> NDArray[np.int32]:
        """Returns the boundary DoFs.

        Returns:
            NDArray[np.int32] -- Boundary DoFs.
        """
        if self._boundary_dofs is None:
            raise RuntimeError('Boundary DoFs need to be set first.')
        return self._boundary_dofs

    def set_boundary_dofs(self, function_space: dfx.fem.FunctionSpace) -> None:
        """Sets the boundary DoFs of the Domain with respect to the 
        function space. Recallable with new function spaces.

        Arguments:
            function_space -- Function space of the simulation.
        """
        fdim = self.mesh.topology.dim - 1
        self._boundary_dofs = dfx.fem.locate_dofs_topological(
            function_space, fdim, self._boundary_facets
        )
