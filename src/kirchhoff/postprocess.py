"""Implements postprocessing functions such as error calculation.
"""

import numpy as np
import dolfinx as dfx
import ufl
from mpi4py import MPI


RELATIVE_FLOOR = 1e-6


def _relative(error: float, norm: float) -> float:
    """Returns nan when the reference is degenerate, otherwise 
    error/norm.

    Arguments:
        error -- The absolute error.
        norm -- The norm of the reference solution.

    Returns:
        float -- nan when the reference is degenerate, otherwise the
        relative error.
    """
    if norm <= RELATIVE_FLOOR * error or norm == 0.:
        return float('nan')
    return error / norm


def calculate_errors(
    mesh: dfx.mesh.Mesh,
    ux: dfx.fem.Function,
    uh: dfx.fem.Function,
) -> tuple[float, float, float, float]:
    """Calculates the L2 and H1 errors between and exact solution ux and
    a discrete solution uh.

    Arguments:
        mesh -- Underlying mesh.
        ux -- Interpolated exact solution.
        uh -- Discrete solution.
    """
    L2_norm = dfx.fem.form(
        ufl.inner(ux,
                  ux) * ufl.dx)
    norm_local = dfx.fem.assemble_scalar(L2_norm)
    norm_L2 = np.sqrt(mesh.comm.allreduce(norm_local, op=MPI.SUM))
    H1_norm = dfx.fem.form(ufl.inner(ufl.grad(ux),
                                     ufl.grad(ux))
                           * ufl.dx)
    norm_local = dfx.fem.assemble_scalar(H1_norm)
    norm_H1 = np.sqrt(mesh.comm.allreduce(norm_local, op=MPI.SUM))

    L2_error = dfx.fem.form(
        ufl.inner(uh - ux,
                  uh - ux) * ufl.dx)
    error_local = dfx.fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(mesh.comm.allreduce(error_local, op=MPI.SUM))

    H1_error = dfx.fem.form(ufl.inner(ufl.grad(uh - ux),
                                      ufl.grad(uh - ux))
                            * ufl.dx)
    error_local = dfx.fem.assemble_scalar(H1_error)
    error_H1 = np.sqrt(mesh.comm.allreduce(error_local, op=MPI.SUM))

    relative_L2 = _relative(error_L2, norm_L2)
    relative_H1 = _relative(error_H1, norm_H1)

    if mesh.comm.rank == 0:
        print(f"absolute Error L2 : {error_L2:.2e}")
        print(f"absolute Error H1 : {error_H1:.2e}")
        print(f"relative Error L2 : {relative_L2:.2e}")
        print(f"relative Error H1 : {relative_H1:.2e}")
    return error_L2, error_H1, error_L2/norm_L2, error_H1/norm_H1
