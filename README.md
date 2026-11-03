# FEM solver using FEniCSx for a second-order Kirchhoff-type PDE using Picard iterations

This code accompanies the two papers on conservative discretizations of a
two-dimensional Kirchhoff-type equation. It is a proof of concept and is not
optimized in any way.

The equation is

$$u_{tt} - \Gamma(t)\,\Delta u = f, \qquad
  \Gamma(t) = \alpha(t) + \beta(t)\,\|\nabla u\|^2,$$

on the unit square with homogeneous Dirichlet data, discretized with
continuous Lagrange elements in space and a three-layer $\theta$-scheme in
time, the nonlinear coefficient resolved by a Picard iteration.

## Installation

For the installation we refer to [miniforge](https://github.com/conda-forge/miniforge).

    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

## Setting up the environment

Create and activate an environment:

    conda create -n kirchhoff-env
    conda activate kirchhoff-env

    conda install fenics-dolfinx
    conda install pip

    pip install -e . --no-deps --no-build-isolation

`--no-deps` matters: without it pip replaces the conda-forge `numpy`,
`mpi4py` and `petsc4py` with PyPI wheels built against a different MPI and
PETSc, and the environment stops working.

## Usage

One script runs everything, `examples/run_example.py`. A single run:

    python examples/run_example.py --ex 41

A convergence study is the same call with refinement levels; the mesh and the
time step refine independently, and the runs are their cross product.

    # spatial: nx doubles four times, nt fixed
    python examples/run_example.py --ex 41 --nx 4 --nx-levels 5 --nt 4096 \
        --degrees 1,2,3 --tables

    # temporal: nt doubles three times, mesh fixed
    python examples/run_example.py --ex 41 --nx 64 --nt 512 --nt-levels 4 \
        --degrees 3 --tables

`--help` lists every option. The defaults of `--theta`, `--eta`, `--gamma`,
`--pc` and the two tolerances are the values used throughout the papers; see
*Parameters* for why.

To reproduce the figures and tables, use the driver rather than the individual
commands:

    ./examples/run_paper.sh              every run of the papers
    ./examples/run_paper.sh --list       group names
    ./examples/run_paper.sh conservation one group
    DRY=1 ./examples/run_paper.sh        print the commands, run nothing
    NP=4 ./examples/run_paper.sh         on four MPI ranks

`examples/run_all.sh` is the larger exploratory campaign behind those choices:
the tolerance study, the stability threshold in $\theta$, the preconditioner
comparison, and the sweeps in $\eta$ and $\gamma$. It is not needed to
reproduce the papers, only to see how the parameters were settled.

## The examples

Every example is a manufactured solution of the form
$u(t,x,y) = g(t)\,\varphi(x,y)$ with
$\varphi = \sin(\lambda\pi x)\sin(\lambda\pi y)$, an eigenfunction of the
Dirichlet Laplacian on the unit square. The source is then

$$f = \bigl[\,g''(t) + \mu\,\Gamma(t)\,g(t)\,\bigr]\varphi,
  \qquad \mu = 2(\lambda\pi)^2 ,$$

written out explicitly in `examples/examples.py` rather than derived, and
checked numerically against the stated solution by the PDE residual.

In `EX_NR` the last digit is $\lambda$, except in the 3xx family where the
middle digit is $\lambda$ and the last is $\lambda_1$.

### Example 1 --- the autonomous reference

`11`, `12`, `13` for $\lambda = 1,2,3$. Constant $\alpha = \beta = 1$ and
$g(t) = \sin t$ on $[0, \pi/2]$. The baseline against which the others are
read: everything else changes exactly one thing relative to it.

`1` is accepted as well and is the same as `11`. It exists so that `--ex` has
a default; the scripts use `11`, so that no table is produced twice.

### Example 2 --- nonautonomous, exponential growth

`21`, `22`, `23`. $\alpha(t) = 1 + 4t$, $\beta(t) = 2 - 4t$ and
$g(t) = e^{\pi t}$. Tests that the time-dependent coefficients are carried
correctly through each step while the solution grows by two orders of
magnitude. The horizon is $[0, 1/4]$, not $\pi/2$, because $e^{\pi t}$ would
otherwise reach 138; it is set automatically and needs no flag.

### Example 3 --- a pulse, with a second frequency

All nine of `311`, `321`, `331`, `313`, `323`, `333`, `315`, `325`, `335`,
for $\lambda = 1,2,3$ and $\lambda_1 = 1,3,5$. Autonomous, with
$g(t) = 1 + \cos(\omega t)$ and $\omega = \lambda_1\pi/(2T)$, so a run always
covers half a pulse whatever the horizon. $\lambda_1$ varies the temporal
frequency independently of the spatial one, which the other examples cannot
do: there the time scale is fixed and only $\lambda$ moves.

This family runs at four times the time steps of the others. At
$\lambda_1 = 5$ the pulse is the fastest in the set, and at the usual `nt` its
temporal error still sits above the spatial one.

The spatial study runs all nine, since $\lambda$ and $\lambda_1$ both matter
there; the temporal study runs `311`, `313`, `315`, which hold $\lambda$ fixed
and vary only the temporal frequency.

### Example 4 --- nonautonomous, everything else held fixed

`41`, `42`, `43`. $\alpha(t) = 1 + t^2$, $\beta(t) = 2 - \sin t$, with the
exact solution, domain and horizon of Example 1. Since only the coefficients
differ, any deviation from the Example 1 curves is attributable to the time
dependence alone. Example 2 changes the solution as well and cannot isolate
it this way.

### Example 5 --- energy conservation

`5`. Unforced and autonomous: $f = 0$, $\alpha = \beta = 1$, with
$u(0) = 0$ and $u_t(0) = \varphi$. This is the only case in which the discrete
energy is conserved, and the only one for which the runner records it.

It has **no exact solution**. Since $\varphi$ is an eigenfunction the problem
reduces to $g'' + \mu(1 + Gg^2)g = 0$, an undamped Duffing equation, which has
no elementary closed form. A placeholder stands in its place, so the error
columns of these runs carry no information and the relative ones come back as
`nan` by design. What is measured is `energy_drift` and `energy_spread`.

Two different experiments use it, and they answer different questions:

 - **conservation**, at fixed $\tau$ over ten periods. Does the energy stay
   put, or does it drift away slowly? Only a long horizon separates a bounded
   oscillation from an accumulating one. Run across
   $\gamma \in \{1/2, 1/4, 0\}$, because $\gamma$ enters the conserved
   functional explicitly.
 - **convergence**, refining $\tau$ over one period. Away from $\theta = 1/2$
   the energy is conserved only in the limit, and the defect follows
   $\mathrm{d}E/E \sim C\,(1/2 - \theta)\,\tau^2$.

## Parameters

Every default was chosen by a measurement.

| | value | why |
|---|---|---|
| `--theta` | `0.5` | the value the energy identity is proved for, and the only one at which the discrete energy is conserved: a drift of `-4e-8` against `-4.6e-5` at $\theta = 1/4$. Also unconditionally stable, where $\theta = 0.2$ diverges at $\tau/h = 0.785$ |
| `--eta` | `0.25` | thirteen times more accurate than $\eta = 0$ or $1/2$ at the same discretization, symmetric about $1/4$, where the leading term of the source mean cancels |
| `--gamma` | `0.5` | free choice: errors agree to five digits for every $\gamma \in [0, 1/2]$ and the order is unchanged. Only the Picard count moves. At $\gamma = 0$ the scheme is linearized and the loop converges in one pass |
| `--pc` | `jacobi` | identical errors to AMG at a quarter of the runtime. $A = M + cK$ is mass dominated at these $\tau/h$, so a diagonal suffices, and unlike AMG its iteration count does not depend on the rank count |
| `--picard-tolerance` | `1e-10` | the error floor is `n_steps * ksp_rtol` and does not involve the Picard tolerance at all |
| `--ksp-rtol` | `1e-12` | `1e-14` costs 50% more Krylov iterations to move a floor already two decades below the finest spatial error, and below `1e-12` the absolute tolerance of `1e-14` binds instead |

Two consequences worth knowing when reading the tables.

The spatial studies run at `nt = 4096`, not 16384. The solver floor is
proportional to the number of steps, so a quarter of the steps is a quarter
of the floor, `4.1e-9` against `1.6e-8`, while the temporal error at
`nt = 4096` is still a factor of 22 below the spatial error at
`nx = 64, p = 3`.

At `p = 3` the last row of each spatial table is temporal, not spatial. For
Example 4 the value at `nx = 64` is `1.5070e-08`, identical to five digits
with the temporal error at `nt = 4096` from the time study. Three clean
orders precede it; the fourth row documents where the time discretization
takes over.

## Output

Everything lands in `examples/results_paper/`:

| file | content |
|---|---|
| `summary.csv` | one row per run, every parameter and every metric |
| `errors_{space,time}_*_ex*.csv` | the pgfplots tables, keyed by `H` or `tau` |
| `series_*.csv` | one row per time step, for the energy runs |
| `energy_drift_*.csv`, `energy_spread_*.csv` | the energy figures |

The last two are written by `examples/extract_energy.py`, which `run_paper.sh`
calls at the end of the conservation group. It thins the series to about 2000
points: a pgfplots line with 80000 of them makes the LaTeX run crawl for
nothing. The full series stays beside it as the raw record.

## Time discretization only

Two scripts verify the conservation identity without a finite element
space. They are not interchangeable.

    python examples/run_energy_conservation.py

Conservation and temporal order for both symmetric three-layer schemes, in
exact spatial arithmetic. Because the manufactured solutions are Dirichlet
eigenfunctions, the schemes reduce to scalar three-layer recursions, so
$\Gamma$ is resolved to machine precision and the drift comes out near
`1e-15`. This is also the only experiment covering the nonautonomous cases
with conserved quantities, which Example 5 cannot: it is autonomous by
construction.

    python examples/run_fully_discrete_conservation.py

The fully discrete identity itself, for $\theta = \gamma = 1/2$, constant
coefficients and $f = 0$: that the exact solution of the discrete scheme
preserves $E_h^n$ step by step, with no time-stepping error of its own in the
way.

Neither replaces the FEM runs of Example 5, which test the same identity in a
real finite element space with inexact solves, where the drift is limited to
about `1e-8` by the Picard and Krylov tolerances. A defect of size `1e-11`
would be invisible there and plain in these two. Conversely the spectral
subspace is the degenerate case in which the discrete Laplacian is diagonal
and there is no quadrature error, so Example 5 is what shows the identity
survives a genuine conforming subspace.

Both need no FEM and depend only on the Python standard library.

## Tests

    pytest examples/test_kirchhoff_examples.py

The source terms are written out by hand rather than derived, so nothing in
`examples.py` guarantees that $f$ matches the solution it was built for. The
tests check it numerically, by assembling the PDE residual with $u_{tt}$ taken
by central differences, and also check the initial data and the boundary
trace. A consistent example lands near `1e-6`; an inconsistent one is `O(1)`.
