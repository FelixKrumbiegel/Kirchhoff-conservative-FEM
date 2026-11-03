#!/usr/bin/env python3
"""Turns the raw run output into the two energy files the figures use.

The manuscript plots

    energy_drift_gamma*.csv    t, drift     relative energy against time
    energy_spread_theta*.csv   tau, spread  peak-to-peak against tau

Neither is written by run_example.py, which emits a per-step series and
a per-run summary row. Without this step the figures cannot be redrawn
from the shipped data, which is the one thing a release has to allow.

    python extract_energy.py results_paper [points]

Reads series_conservation-g*.csv and summary.csv from that directory and
writes the four files beside them.

The drift files are thinned to `points` rows, 2000 by default. A series
at 81920 steps is 8 MB, and a pgfplots line with that many points makes
the LaTeX run crawl and the PDF enormous for nothing: at 300 dpi and
10 cm width there are 1200 pixels across. The series file stays whole
beside it, as the raw record; only the plot input is thinned.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def _log_indices(count: int, points: int) -> list[int]:
    """Row indices spaced evenly on a log scale, not a linear one.

    The drift figure has a log x axis, and the series starts at t = tau
    and runs to T, four decades. Even spacing would put four fifths of
    the points in the last decade; this spreads them across all of it.
    The first and last rows are always kept, so the curve starts and
    ends where the run did.

    Arguments:
        count -- How many rows the series has.
        points -- Roughly how many to keep.

    Returns:
        list -- Strictly increasing indices into the series.
    """
    if count <= points or points < 2:
        return list(range(count))
    span = math.log(count)
    chosen = {0, count - 1}
    for k in range(points):
        chosen.add(min(count - 1,
                       int(round(math.exp(span * k / (points - 1)))) - 1))
    return sorted(i for i in chosen if i >= 0)


def drift_series(path: Path, points: int) -> list[tuple[float, float]]:
    """Relative energy against time, from one per-step series file.

    Thinned logarithmically in t, since that is the axis the figure
    uses. See `_log_indices`.

    Arguments:
        path -- A series_*.csv written by run_example.py.
        points -- Roughly how many rows to keep.

    Returns:
        list -- (t, (E(t) - E(0)) / |E(0)|).
    """
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    first = float(rows[0]['energy'])
    # The magnitude, not the signed value. The figure has a log y axis,
    # and the drift is negative for most of the run, so pgfplots would
    # silently discard every point. The sign carries no information
    # here anyway: what is being shown is how far the energy moves.
    #
    # Row 0 is skipped because it is the reference and its drift is
    # exactly zero, which a log axis also cannot take.
    picked = [i for i in _log_indices(len(rows), points) if i > 0]
    out = []
    for i in picked:
        value = abs(float(rows[i]['energy']) - first) / abs(first)
        if value > 0.:
            out.append((float(rows[i]['time']), value))
    return out


# The tau refinement is tagged `convergence-t<theta>` by run_paper.sh
# and `energy-t<theta>` by run_all.sh. The thetas themselves are not
# hard-coded: adding one to either script is enough and this needs no
# edit.
CONVERGENCE_TAGS = ('convergence-t', 'energy-t')


def _label(theta: str) -> str:
    """`0.5` -> `05`, for the file name."""
    return theta.replace('.', '').rstrip('0') or '0'


def _theta_of(tag: str) -> str:
    """The theta a refinement tag carries."""
    for prefix in CONVERGENCE_TAGS:
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return tag


def tau_refinement_tags(summary: Path) -> list[str]:
    """The tags of every tau refinement in the file.

    Arguments:
        summary -- The summary.csv of the run directory.

    Returns:
        list -- Full tags, e.g. `convergence-t0.5`.
    """
    with summary.open(newline='', encoding='utf-8') as handle:
        tags = {r['tag'] for r in csv.DictReader(handle)
                if r['tag'].startswith(CONVERGENCE_TAGS)}
    return sorted(tags)


def spread_by_tau(
    summary: Path,
    tag_prefix: str
) -> list[tuple[float, float]]:
    """Peak-to-peak energy variation against tau, from summary.csv.

    The spread rather than the drift, because over many periods a drift
    that reverses averages away while the spread does not.

    Arguments:
        summary -- The summary.csv of the run directory.
        tag_prefix -- Only rows whose tag starts with this are used.

    Returns:
        list -- (tau, spread), coarsest tau first.
    """
    with summary.open(newline='', encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle)
                if r['tag'].startswith(tag_prefix)]
    # run_all.sh sweeps the mesh as well, so a tau can appear three
    # times. The drift is independent of nx to three digits, but three
    # rows at one tau would still draw as a zigzag, so the finest mesh
    # wins and the rest are dropped.
    best: dict[float, tuple[int, float]] = {}
    for r in rows:
        if r['energy_spread'] in ('', 'nan'):
            continue
        tau, nx = float(r['tau']), int(r['nx'])
        if tau not in best or nx > best[tau][0]:
            best[tau] = (nx, float(r['energy_spread']))
    return sorted(((tau, value) for tau, (_, value) in best.items()),
                  reverse=True)


def write(path: Path, header: tuple[str, str],
          rows: list[tuple[float, float]]) -> None:
    """Writes a two-column CSV at full precision."""
    lines = [','.join(header)]
    lines += [f'{a!r},{b!r}' for a, b in rows]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'  {path.name:<34} {len(rows)} rows')


def main(directory: Path, points: int = 2000) -> int:
    """Entry point.

    Returns:
        int -- Process exit status.
    """
    if not directory.is_dir():
        print(f'no such directory: {directory}', file=sys.stderr)
        return 1

    print('drift against time, one file per gamma:')
    for series in sorted(directory.glob('series_conservation-g*.csv')):
        gamma = series.name.split('-g')[1].split('_')[0].replace('.', '')
        write(directory / f'energy_drift_gamma{gamma or "0"}.csv',
              ('t', 'drift'), drift_series(series, points))

    summary = directory / 'summary.csv'
    if not summary.exists():
        print(f'\nno summary.csv in {directory}; skipping the spread files',
              file=sys.stderr)
        return 1

    print('\nspread against tau, one file per theta:')
    for tag in tau_refinement_tags(summary):
        rows = spread_by_tau(summary, tag)
        if rows:
            label = _label(_theta_of(tag))
            write(directory / f'energy_spread_theta{label}.csv',
                  ('tau', 'spread'), rows)
    return 0


if __name__ == '__main__':
    sys.exit(main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results_paper'),
        int(sys.argv[2]) if len(sys.argv) > 2 else 2000))
