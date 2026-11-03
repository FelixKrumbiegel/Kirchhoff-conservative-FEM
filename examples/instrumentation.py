"""Timing for the study driver.

This lives next to the runner rather than in the package: the solver
should not know it is being measured. Nothing in `kirchhoff` imports it,
and deleting this file costs only the timing columns.

Why not PETSc's event log
-------------------------
PETSc counts KSPSolve and PCSetUp itself, which would need no wrapping
at all. But several DOLFINx distributions build PETSc with logging
compiled out, and then `getPerfInfo` returns zeros with no error, which
is exactly the failure that is hard to notice. `petsc_log_works` below
reports whether it is usable on this build; the timers here do not
depend on it either way.

What is measured
----------------
`SolveProbe` replaces two methods on `LinearSystem` and `WaveSystem` for
the duration of a `with` block and restores them afterwards. It is a
blunt instrument, and deliberately so: the alternative is threading
stopwatches through `step`, which would put measurement code in the
solver permanently for the sake of one study.

    solve_time         seconds inside LinearSystem.solve, which is the
                       Krylov iterations and, when the operator changed,
                       the preconditioner setup
    matrix_time        seconds inside WaveSystem.set_coefficient, the
                       copy and axpy that rebuild A
    iterations         Krylov iterations summed over every solve
    solves             how many solves happened

Separating the preconditioner setup from the Krylov work does not need a
profiler: run the same problem with --pc amg and --pc amg_reuse, and the
difference in solve_time is the setup cost, since nothing else differs.
That comparison is more trustworthy than a counter anyway.
"""
from __future__ import annotations

import time

from kirchhoff.linear_system import LinearSystem, WaveSystem


def petsc_log_works() -> bool:
    """Whether PETSc was built with its event log enabled.

    Returns:
        bool -- True if event timings are nonzero and therefore usable.
    """
    try:
        from petsc4py import PETSc  # pylint: disable=import-outside-toplevel
        PETSc.Log.begin()
        info = PETSc.Log.Event('KSPSolve').getPerfInfo()
        return bool(info) and 'time' in info
    except Exception:  # pylint: disable=broad-except
        return False


class SolveProbe:
    """Times the linear algebra of one run.

        with SolveProbe() as probe:
            ...
        probe.solve_time

    Wraps the methods on the classes, not on an instance, so the solver
    needs no hook and no argument. The originals are restored on exit,
    including when the block raises.
    """

    def __init__(self) -> None:
        self.solve_time: float = 0.
        self.matrix_time: float = 0.
        self.iterations: int = 0
        self.solves: int = 0
        self._solve = None
        self._set_coefficient = None

    def __enter__(self) -> SolveProbe:
        probe = self
        self._solve = LinearSystem.solve
        self._set_coefficient = WaveSystem.set_coefficient

        def timed_solve(system, rhs, solution):
            started = time.perf_counter()
            iterations = probe._solve(system, rhs, solution)
            probe.solve_time += time.perf_counter() - started
            probe.iterations += iterations
            probe.solves += 1
            return iterations

        def timed_set_coefficient(system, value):
            started = time.perf_counter()
            probe._set_coefficient(system, value)
            probe.matrix_time += time.perf_counter() - started

        LinearSystem.solve = timed_solve
        WaveSystem.set_coefficient = timed_set_coefficient
        return self

    def __exit__(self, *exception) -> None:
        LinearSystem.solve = self._solve
        WaveSystem.set_coefficient = self._set_coefficient

    def snapshot(self) -> tuple[float, int]:
        """Returns the running totals, for differencing per step.

        Returns:
            tuple -- Seconds in solve so far, and iterations so far.
        """
        return self.solve_time, self.iterations
