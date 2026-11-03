"""Runs Kirchhoff examples and records everything a study needs.

    python run_example.py --ex 41
    python run_example.py --ex 41 --nx 4 --nx-levels 5 --degrees 1,2,3
    python run_example.py --ex 41 --nx 64 --nt 512 --nt-levels 6 --degrees 3
    python run_example.py --ex 41 --nx 8 --nx-levels 4 --nt 128 --nt-levels 6
    python run_example.py --ex 5 --nx 64 --nt 4096 --pc mass --tag pc-mass

Sweeps
------
Each axis doubles independently:

    --nx-levels K   nx and ny double K-1 times
    --nt-levels L   ntime doubles L-1 times

The runs are the cross product, K*L of them, which is what a study of
the optimal tau for a given h needs. `--couple` zips the two axes
instead, giving K runs with tau proportional to h.

    space study     --nx-levels 5 --nt-levels 1   (nt fixed and high)
    time study      --nx-levels 1 --nt-levels 6   (nx fixed and fine)
    grid            --nx-levels 5 --nt-levels 6
    coupled         --nx-levels 5 --nt-levels 5 --couple

Output
------
Three kinds of file, all under --out-dir.

`summary.csv` is the master record: one row per run, appended across
invocations, carrying every parameter and every scalar metric. Every
question below is a filter on this one file. Delete it to start over.

`series_*.csv` is one file per run, one row per time step: the Picard
and Krylov counts, the nonlinear coefficient, the energy and the
amplitude. This is where a growing iteration count or a drifting energy
shows up.

`errors_*.csv` are the four pgfplots tables of the manuscript, written
only when the sweep has a single axis, since a table keyed by H is
meaningless over a grid.

`optimal_tau_ex*.csv` is written after a grid sweep: for each mesh and
degree, the coarsest time step whose error is within --optimal-tolerance
of the finest one. That is the point past which refining in time buys
nothing, and the ratio tau/h in that row is the quantity worth quoting.

What is tracked, and why
------------------------
errors            the four norms, as before
picard_mean/max   a max at the limit means the nonlinear iteration never
                  converged, and the errors are not the scheme's
krylov_mean       a proxy for the conditioning of A, and the thing the
                  preconditioner study moves
gamma_first/last  how far the nonlinearity travels over the run
energy_*          drift and spread, recorded only for the examples
                  that conserve energy, namely an unforced one with
                  autonomous coefficients. nan elsewhere, since the
                  number would carry no information there
solve_time        seconds inside the linear solves, measured by
                  wrapping them from the harness rather than from src
matrix_time       seconds rebuilding A from M and K
other_time        runtime minus the two: assembly and bookkeeping

To separate the preconditioner setup from the Krylov work, run the same
problem with --pc amg and --pc amg_reuse: nothing else differs, so the
gap in solve_time is the setup cost.
amplitude_max     catches a blow-up before it becomes a nan, which is
                  what a stability study needs
runtime           wall time of the time loop only
ranks             MPI ranks the run used; solve_time is rank 0 wall
                  time including its waits at collectives, so timings
                  are comparable only within one rank count
picard_tolerance  the nonlinear tolerance in force
ksp_rtol          the Krylov relative tolerance in force
status            ok, stalled, or diverged
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace, asdict, fields
from pathlib import Path

import numpy as np
import dolfinx as dfx
from mpi4py import MPI

from kirchhoff.domain import Domain
from kirchhoff.space_discretization import SpaceDiscretization
from kirchhoff.kirchhoff_solver import NonlinearKirchhoffThetaScheme
from kirchhoff.kirchhoff_data import KirchhoffConfig
from kirchhoff.functionals import (
    KirchhoffEnergy, gradient_seminorm, l2_norm_squared)
from kirchhoff.postprocess import calculate_errors

from examples import example, ALL
import instrumentation

PRECONDITIONERS = ('amg', 'amg_reuse', 'mass', 'jacobi')

# Past this the run is called diverged and abandoned. Every example is
# O(1) in amplitude, so this cannot fire on a healthy run.
#
# 1e6 was too permissive: a theta=0.2 run at tau/h=0.785 ended with an
# L2 error of 8.5e+04 and still reported `ok`, because the amplitude
# stayed under the limit at the final step. The EOC column then read
# 26.40, which is worse than a missing row.
AMPLITUDE_LIMIT = 1e2

# The energy is conserved only when the source vanishes and alpha and
# beta do not depend on t. Everywhere else the balance carries a work
# term, so the number would say nothing about the scheme and the two
# matrix vector products per step would be wasted.
CONSERVATIVE_EXAMPLES: tuple[int, ...] = (5,)


def conserves_energy(ex_nr: int) -> bool:
    """Whether the discrete energy of this example should be constant.

    This belongs on `Example` as a field, next to `autonomous`, once
    example 5 exists; it sits here so the package needs no change.

    Arguments:
        ex_nr -- Example number.

    Returns:
        bool -- Whether to record the energy.
    """
    return ex_nr in CONSERVATIVE_EXAMPLES


def default_final_time(ex_nr: int) -> float:
    """Final time for each example.

    Example 2 grows like exp(pi t) and is therefore posed on the short
    interval [0, 1/4]. Every other example runs to pi/2.

    Example 5 is the energy test and has no manufactured solution, so
    nothing fixes its horizon; pi/2 is roughly one oscillation. Pass
    --final-time to watch several.
    """
    if ex_nr // 10 == 2:
        return .25
    return math.pi / 2.


# ----------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Setup:
    """Everything a single solve needs. Mirrors the command line."""
    ex_nr: int
    nx: int
    ny: int
    ntime: int
    final_time: float
    degree: int
    theta: float
    eta: float
    gamma: float
    preconditioner: str = 'amg'
    picard_tolerance: float = 1e-10
    picard_max_iterations: int = 10
    ksp_rtol: float = 1e-10
    tag: str = ''


@dataclass(frozen=True, kw_only=True)
class Result:
    """One row of `summary.csv`.

    Every field is a scalar, so the whole study is one flat table that
    can be filtered on `tag`, `ex`, `preconditioner` or anything else.
    """
    tag: str
    ranks: int
    ex: int
    nx: int
    ny: int
    ntime: int
    degree: int
    h: float
    tau: float
    tau_over_h: float
    theta: float
    eta: float
    gamma: float
    preconditioner: str
    absolute_l2: float
    absolute_h1: float
    relative_l2: float
    relative_h1: float
    picard_mean: float
    picard_max: int
    picard_stalled: int
    krylov_mean: float
    krylov_total: int
    gamma_first: float
    gamma_last: float
    energy_first: float
    energy_last: float
    energy_drift: float
    energy_spread: float
    amplitude_max: float
    picard_tolerance: float
    ksp_rtol: float
    runtime: float
    solve_time: float
    matrix_time: float
    other_time: float
    status: str


@dataclass
class Series:
    """The per-step record of one run, written to its own file."""
    step: list[int]
    time: list[float]
    picard: list[int]
    krylov: list[int]
    gamma: list[float]
    energy: list[float]
    amplitude: list[float]

    @classmethod
    def empty(cls) -> Series:
        """Returns an empty series."""
        return cls([], [], [], [], [], [], [])

    def append(self, **values) -> None:
        """Appends one step."""
        for name, value in values.items():
            getattr(self, name).append(value)


def solve(setup: Setup) -> tuple[Result, Series]:
    """Runs one example to the final time.

    Arguments:
        setup -- The discretization and the example to run.

    Returns:
        tuple -- The summary row and the per-step series.
    """
    mesh: dfx.mesh.Mesh = dfx.mesh.create_unit_square(
        MPI.COMM_WORLD, setup.nx, setup.ny,
        cell_type=dfx.mesh.CellType.triangle)
    function_space: dfx.fem.FunctionSpace = dfx.fem.functionspace(
        mesh, ('Lagrange', setup.degree))

    ex = example(setup.ex_nr, final_time=setup.final_time)
    config = KirchhoffConfig(
        nx=setup.nx, ny=setup.ny,
        ntime=setup.ntime, final_time=setup.final_time,
        theta=setup.theta, eta=setup.eta, gamma=setup.gamma)

    domain = Domain(mesh)
    space = SpaceDiscretization(function_space)
    integrator = NonlinearKirchhoffThetaScheme(
        config, ex.data(mesh), domain, space,
        picard_tolerance=setup.picard_tolerance,
        picard_max_iterations=setup.picard_max_iterations,
        preconditioner=setup.preconditioner,
        ksp_rtol=setup.ksp_rtol)

    # Only built when the example conserves energy; it costs a mass
    # matrix and two matrix vector products per step.
    if conserves_energy(setup.ex_nr):
        stiffness = gradient_seminorm(space)
        mass = l2_norm_squared(space)
        energy = KirchhoffEnergy(
            space, config.tau, config.gamma, mass, stiffness)
    else:
        stiffness = mass = energy = None

    series = Series.empty()
    status = 'ok'
    started = time.perf_counter()
    with instrumentation.SolveProbe() as probe:
        for index in range(config.ntime - 1):
            before = probe.iterations
            try:
                report = integrator.step()
            except RuntimeError as failure:
                # A solver that gives up raises, and the traceback takes
                # the whole process with it: every step computed so far
                # is lost, which is exactly the record needed to explain
                # the failure. Keep the series, mark the run, and say
                # which step it was.
                if mesh.comm.rank == 0:
                    print(f'step {index + 1} of {config.ntime - 1} failed '
                          f'at t = {(index + 2) * config.tau:.6f}: '
                          f'{failure}')
                status = 'failed'
                break
            amplitude = mesh.comm.allreduce(
                float(np.max(np.abs(integrator.uh.x.array))), MPI.MAX)
            series.append(
                step=index + 1,
                time=(index + 2) * config.tau,
                picard=report.picard,
                # From the probe, so the solver needs no counter of its
                # own: the difference over one step is that step's cost.
                krylov=probe.iterations - before,
                gamma=report.gamma,
                energy=(float('nan') if energy is None
                        else integrator.energy(energy)),
                amplitude=amplitude)
            if not math.isfinite(amplitude) or amplitude > AMPLITUDE_LIMIT:
                status = 'diverged'
                break
    runtime = time.perf_counter() - started

    if status == 'ok':
        fine_space = dfx.fem.functionspace(
            mesh, ('Lagrange', setup.degree + 1))
        exact = ex.exact(mesh)
        exact.shift(setup.final_time)
        exact.set_fem_evaluation(fine_space)
        exact_fine = dfx.fem.Function(fine_space)
        exact_fine.interpolate(exact.fem_evaluation)
        errors = calculate_errors(mesh, exact_fine, integrator.uh)
    else:
        errors = (float('nan'),) * 4

    stalled = sum(1 for value in series.picard
                  if value >= setup.picard_max_iterations)
    if status == 'ok' and stalled:
        status = 'stalled'

    result = Result(
        tag=setup.tag, ranks=MPI.COMM_WORLD.size, ex=setup.ex_nr,
        nx=setup.nx, ny=setup.ny, ntime=setup.ntime, degree=setup.degree,
        h=1. / max(setup.nx, setup.ny), tau=config.tau,
        tau_over_h=config.tau * max(setup.nx, setup.ny),
        theta=setup.theta, eta=setup.eta, gamma=setup.gamma,
        preconditioner=setup.preconditioner,
        absolute_l2=errors[0], absolute_h1=errors[1],
        relative_l2=errors[2], relative_h1=errors[3],
        picard_mean=_mean(series.picard),
        picard_max=max(series.picard, default=0),
        picard_stalled=stalled,
        krylov_mean=_mean(series.krylov), krylov_total=sum(series.krylov),
        gamma_first=_first(series.gamma), gamma_last=_last(series.gamma),
        energy_first=_first(series.energy), energy_last=_last(series.energy),
        energy_drift=_drift(series.energy),
        energy_spread=_spread(series.energy),
        amplitude_max=max(series.amplitude, default=float('nan')),
        picard_tolerance=setup.picard_tolerance,
        ksp_rtol=setup.ksp_rtol,
        runtime=runtime,
        solve_time=probe.solve_time,
        matrix_time=probe.matrix_time,
        other_time=runtime - probe.solve_time - probe.matrix_time,
        status=status)

    integrator.destroy()
    # KirchhoffEnergy owns nothing: its two work vectors are Functions,
    # which clean up after themselves, and the functionals are borrowed.
    for owned in (mass, stiffness):
        if owned is not None:
            owned.destroy()
    return result, series


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else float('nan')


def _first(values: list) -> float:
    return float(values[0]) if values else float('nan')


def _last(values: list) -> float:
    return float(values[-1]) if values else float('nan')


def _drift(values: list) -> float:
    """Relative change of a series from first to last value."""
    if len(values) < 2 or values[0] == 0.:
        return float('nan')
    return (values[-1] - values[0]) / abs(values[0])


def _spread(values: list) -> float:
    """Relative peak to peak variation.

    Catches an oscillation that a first-to-last drift would miss, which
    is the shape the energy error takes at theta above 1/4.
    """
    if len(values) < 2 or values[0] == 0.:
        return float('nan')
    return (max(values) - min(values)) / abs(values[0])


# ----------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------
def sweep(
    setup: Setup,
    *,
    nx_levels: int,
    nt_levels: int,
    degrees: list[int],
    couple: bool
) -> list[Setup]:
    """Returns the setups of a study, coarsest first.

    Arguments:
        setup -- The coarsest level.
        nx_levels, nt_levels -- How many doublings of each axis.
        degrees -- Polynomial degrees to sweep.
        couple -- Zip the two axes instead of crossing them.

    Raises:
        ValueError: If coupling is asked for with unequal level counts.

    Returns:
        list[Setup] -- One setup per run.
    """
    meshes = [setup.nx * 2**k for k in range(nx_levels)]
    steps = [setup.ntime * 2**k for k in range(nt_levels)]

    if couple:
        if nx_levels != nt_levels:
            raise ValueError(
                '--couple needs --nx-levels and --nt-levels to agree.')
        pairs = list(zip(meshes, steps))
    else:
        pairs = [(nx, nt) for nx in meshes for nt in steps]

    ratio = setup.ny / setup.nx
    return [
        replace(setup, nx=nx, ny=int(nx * ratio), ntime=nt, degree=degree)
        for nx, nt in pairs
        for degree in degrees
    ]


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def _csv(value) -> str:
    """Formats one value for a CSV cell.

    numpy scalars leak in from the assembled norms, and repr() on those
    gives `np.float64(1e-06)` under numpy 2, which no CSV reader will
    take as a number. Casting to a Python float first keeps the full
    precision that repr() gives without the wrapper. Integers stay
    integers so that ntime does not become 4096.0.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return repr(float(value))


def append_summary(results: list[Result], directory: Path) -> Path:
    """Appends the runs to the master table, writing a header if new.

    Returns:
        Path -- The file written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'summary.csv'
    names = [field.name for field in fields(Result)]

    lines: list[str] = []
    if not path.exists():
        lines.append(','.join(names))
    for result in results:
        row = asdict(result)
        lines.append(','.join(_csv(row[name]) for name in names))

    with path.open('a', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path


def write_series(result: Result, series: Series, directory: Path) -> Path:
    """Writes the per-step record of one run.

    Returns:
        Path -- The file written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    label = result.tag or 'run'
    path = directory / (
        f'series_{label}_ex{result.ex}_nx{result.nx}'
        f'_nt{result.ntime}_p{result.degree}.csv')

    names = ['step', 'time', 'picard', 'krylov', 'gamma', 'energy',
             'amplitude']
    lines = [','.join(names)]
    for row in zip(*(getattr(series, name) for name in names)):
        lines.append(','.join(_csv(value) for value in row))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def write_error_tables(
    results: list[Result],
    degrees: list[int],
    directory: Path,
    ex_nr: int,
    by_time: bool,
    suffix: str = ''
) -> list[Path]:
    """Writes the four pgfplots tables of a one-dimensional sweep.

    The file name carries the axis, the quantity and the example
    number, which is not enough to separate two sweeps of the same
    example under different scheme parameters: a gamma = 0 run of
    example 41 would overwrite the gamma = 1/2 one. `suffix` is
    appended before the extension to keep them apart.

    Returns:
        list[Path] -- The files written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    axis = 'time' if by_time else 'space'
    key = 'tau' if by_time else 'H'
    labels = (['error'] if len(degrees) == 1
              else [f'p{degree}' for degree in degrees])

    def step_of(result: Result) -> float:
        return result.tau if by_time else result.h

    ordered = sorted({step_of(result) for result in results}, reverse=True)
    columns = (
        ('absolute_L2', 'absolute_l2'),
        ('absolute_H1', 'absolute_h1'),
        ('relative_L2', 'relative_l2'),
        ('relative_H1', 'relative_h1'),
    )

    written: list[Path] = []
    for label, attribute in columns:
        path = directory / f'errors_{axis}_{label}_ex{ex_nr}{suffix}.csv'
        lines = [f'{key},' + ','.join(labels)]
        for step in ordered:
            row = []
            for degree in degrees:
                match = [r for r in results
                         if step_of(r) == step and r.degree == degree]
                row.append(_csv(getattr(match[0], attribute))
                           if match else 'nan')
            lines.append(_csv(step) + ',' + ','.join(row))
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        written.append(path)
    return written


def write_optimal_tau(
    results: list[Result],
    directory: Path,
    ex_nr: int,
    tolerance: float
) -> Path | None:
    """For each mesh and degree, the coarsest tau that is good enough.

    The spatial error is O(h^{p+1}) and the temporal one O(tau^2), so
    below some tau the total error stops moving. This finds the first
    time step whose error is within `tolerance` of the finest one, which
    is the largest tau worth taking on that mesh. The tau/h of that row
    is the quantity to quote.

    Returns:
        Path | None -- The file written, or None if the sweep had only
        one time step per mesh.
    """
    keys = sorted({(r.nx, r.degree) for r in results})
    rows: list[str] = []
    for nx, degree in keys:
        group = sorted(
            (r for r in results
             if r.nx == nx and r.degree == degree
             and math.isfinite(r.absolute_l2)),
            key=lambda r: r.ntime)
        if len(group) < 2:
            continue
        finest = group[-1].absolute_l2
        chosen = next(
            (r for r in group
             if abs(r.absolute_l2 - finest) <= tolerance * abs(finest)),
            group[-1])
        rows.append(','.join(_csv(value) for value in (
            chosen.h, chosen.degree, chosen.ntime, chosen.tau,
            chosen.tau_over_h, chosen.absolute_l2, finest)))

    if not rows:
        return None

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'optimal_tau_ex{ex_nr}.csv'
    path.write_text(
        'H,p,ntime,tau,tau_over_h,error,error_finest\n'
        + '\n'.join(rows) + '\n', encoding='utf-8')
    return path


def _order(
    previous_error: float,
    current_error: float,
    previous_step: float,
    current_step: float
) -> float:
    """Experimental order of convergence between two levels.

    Uses the actual step ratio rather than assuming a halving, so it
    stays correct if a schedule other than doubling is added.
    """
    if min(previous_error, current_error) <= 0.:
        return float('nan')
    if previous_step <= 0. or current_step <= 0. or \
            previous_step == current_step:
        return float('nan')
    return (math.log(previous_error / current_error)
            / math.log(previous_step / current_step))


def _previous(group: list[Result], result: Result) -> Result | None:
    """The entry one refinement coarser than `result` in its group."""
    index = group.index(result)
    return group[index - 1] if index > 0 else None


def _orders(results: list[Result]) -> list[tuple[float, float]]:
    """Experimental orders in h and in tau, one pair per result.

    A grid sweep refines both axes, so an order is only meaningful
    against the run that differs in exactly one of them: the spatial
    order compares runs sharing a degree and a time step count, the
    temporal order runs sharing a degree and a mesh. Whichever axis is
    not being refined comes out as nan.
    """
    by_mesh: dict[tuple[int, int], list[Result]] = {}
    by_step: dict[tuple[int, int], list[Result]] = {}
    for result in results:
        by_mesh.setdefault((result.degree, result.ntime), []).append(result)
        by_step.setdefault((result.degree, result.nx), []).append(result)
    for group in by_mesh.values():
        group.sort(key=lambda r: r.nx)
    for group in by_step.values():
        group.sort(key=lambda r: r.ntime)

    orders: list[tuple[float, float]] = []
    for result in results:
        coarser_h = _previous(by_mesh[(result.degree, result.ntime)], result)
        coarser_t = _previous(by_step[(result.degree, result.nx)], result)
        orders.append((
            float('nan') if coarser_h is None else _order(
                coarser_h.absolute_l2, result.absolute_l2,
                coarser_h.h, result.h),
            float('nan') if coarser_t is None else _order(
                coarser_t.absolute_l2, result.absolute_l2,
                coarser_t.tau, result.tau)))
    return orders


def format_table(results: list[Result]) -> str:
    """Renders the runs of this invocation, with orders where they apply.

    Both order columns are on L2. EOCh is against the next coarser mesh
    at the same time step, EOCt against the next coarser time step on
    the same mesh, so a single-axis sweep leaves the other column empty.
    """
    orders = _orders(results)
    header = (f'{"nx":>5} {"ntime":>8} {"p":>2} {"tau/h":>9} '
              f'{"L2":>12} {"EOCh":>5} {"EOCt":>5} {"H1":>12} '
              f'{"picard":>7} {"krylov":>7} '
              f'{"dE/E":>10} {"time/s":>8} {"solve/s":>8} '
              f'{"mat/s":>7}  status')
    lines = [header, '-' * len(header)]
    for result, (order_h, order_t) in zip(results, orders):
        lines.append(
            f'{result.nx:>5} {result.ntime:>8} {result.degree:>2} '
            f'{result.tau_over_h:>9.3e} '
            f'{result.absolute_l2:>12.4e} {order_h:>5.2f} {order_t:>5.2f} '
            f'{result.absolute_h1:>12.4e} '
            f'{result.picard_mean:>7.2f} {result.krylov_mean:>7.1f} '
            f'{result.energy_drift:>10.2e} {result.runtime:>8.1f} '
            f'{result.solve_time:>8.1f} {result.matrix_time:>7.1f}  '
            f'{result.status}')
    return '\n'.join(lines)


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def _degree_list(text: str) -> list[int]:
    """Parses '1,2,3' into [1, 2, 3]."""
    try:
        degrees = [int(part) for part in text.split(',') if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f'not a comma separated list of integers: {text!r}') from error
    if not degrees or any(degree < 1 for degree in degrees):
        raise argparse.ArgumentTypeError('degrees must be positive integers')
    return degrees


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--ex', type=int, required=True, choices=ALL,
                        metavar='N', help=f'example number, one of {ALL}')
    parser.add_argument('--tag', default='',
                        help='label for these runs in summary.csv')

    mesh = parser.add_argument_group('discretization')
    mesh.add_argument('--nx', type=int, default=4,
                      help='cells in x on the coarsest level')
    mesh.add_argument('--ny', type=int, default=None,
                      help='cells in y; defaults to --nx')
    mesh.add_argument('--nt', type=int, default=1024*16,
                      help='time steps on the coarsest level')
    mesh.add_argument('--nx-levels', type=int, default=1,
                      help='how many doublings of the mesh')
    mesh.add_argument('--nt-levels', type=int, default=1,
                      help='how many doublings of the time step count')
    mesh.add_argument('--couple', action='store_true',
                      help='zip the two axes instead of crossing them')
    mesh.add_argument('--degrees', type=_degree_list, default=[1],
                      metavar='P[,P...]',
                      help='polynomial degrees to sweep, e.g. 1,2,3')
    mesh.add_argument('--final-time', type=float, default=None,
                      help='end of the time interval; defaults to the '
                           'horizon the example is posed on')

    scheme = parser.add_argument_group('scheme')
    scheme.add_argument('--theta', type=float, default=.5,
                        help='theta of the Laplace term')
    scheme.add_argument('--eta', type=float, default=0.,
                        help='theta of the right-hand side')
    scheme.add_argument('--gamma', type=float, default=.5,
                        help='theta of the nonlinearity')
    scheme.add_argument('--pc', default='amg', choices=PRECONDITIONERS,
                        help='preconditioner for the linear solves')
    scheme.add_argument('--picard-tolerance', type=float, default=1e-10,
                        help='tolerance of the nonlinear iteration')
    scheme.add_argument('--picard-max', type=int, default=10,
                        help='cap on the nonlinear iteration')
    scheme.add_argument('--ksp-rtol', type=float, default=1e-10,
                        help='relative tolerance of the linear solves; '
                             'the absolute one stays at the 1e-14 of '
                             'LinearSystem, which can bind instead once '
                             'this goes below about 1e-12')

    output = parser.add_argument_group('output')
    output.add_argument('--out-dir', type=Path, default=None,
                        help='where to write; defaults to <this file>/results')
    output.add_argument('--no-series', action='store_true',
                        help='skip the per-step files, which a large '
                             'sweep produces a lot of')
    output.add_argument('--tables', action='store_true',
                        help='also write the pgfplots error tables; only '
                             'meaningful for a single-axis sweep')
    output.add_argument('--table-suffix', default='',
                        help='appended to the error table file names, to '
                             'keep two sweeps of the same example under '
                             'different scheme parameters apart')
    output.add_argument('--optimal-tolerance', type=float, default=.05,
                        help='relative slack for the optimal tau search')

    args = parser.parse_args(argv)
    if args.ny is None:
        args.ny = args.nx
    if args.final_time is None:
        args.final_time = default_final_time(args.ex)
    if min(args.nx_levels, args.nt_levels) < 1:
        parser.error('level counts must be at least 1')
    if args.couple and args.nx_levels != args.nt_levels:
        parser.error('--couple needs --nx-levels and --nt-levels to agree')
    if args.tables and args.nx_levels > 1 and args.nt_levels > 1 \
            and not args.couple:
        parser.error('--tables needs a single-axis sweep')
    if args.out_dir is None:
        args.out_dir = Path(__file__).resolve().parent / 'results'
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        int -- Process exit status, nonzero if any run did not finish
        cleanly, so a driver script can tell.
    """
    args = parse_args(argv)
    base = Setup(
        ex_nr=args.ex, nx=args.nx, ny=args.ny, ntime=args.nt,
        final_time=args.final_time, degree=args.degrees[0],
        theta=args.theta, eta=args.eta, gamma=args.gamma,
        preconditioner=args.pc,
        picard_tolerance=args.picard_tolerance,
        picard_max_iterations=args.picard_max,
        ksp_rtol=args.ksp_rtol,
        tag=args.tag)

    setups = sweep(base, nx_levels=args.nx_levels, nt_levels=args.nt_levels,
                   degrees=args.degrees, couple=args.couple)

    results: list[Result] = []
    for setup in setups:
        result, series = solve(setup)
        results.append(result)
        if MPI.COMM_WORLD.rank == 0 and not args.no_series:
            write_series(result, series, args.out_dir)

    if MPI.COMM_WORLD.rank == 0:
        ex = example(args.ex, final_time=args.final_time)
        print(f'\nexample {ex.name}, T = {args.final_time:.6f}, '
              f'theta = ({args.theta}, {args.eta}, {args.gamma}), '
              f'pc = {args.pc}, ranks = {MPI.COMM_WORLD.size}, '
              f'picard = {args.picard_tolerance:g}, '
              f'ksp = {args.ksp_rtol:g}, tag = {args.tag or "-"}')
        print(format_table(results))

        append_summary(results, args.out_dir)
        if args.tables:
            write_error_tables(
                results, args.degrees, args.out_dir, args.ex,
                by_time=args.nx_levels == 1 and args.nt_levels > 1,
                suffix=args.table_suffix)
        if args.nt_levels > 1 and not args.couple:
            write_optimal_tau(results, args.out_dir, args.ex,
                              args.optimal_tolerance)
        print(f'\nwritten to {args.out_dir}')

    return 1 if any(r.status != 'ok' for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
