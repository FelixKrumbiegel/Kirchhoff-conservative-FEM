"""Energy-conservation and temporal-order experiment for the two symmetric
three-layer schemes, run in exact spatial arithmetic.

For the homogeneous Kirchhoff problem with constant coefficients on the square
Omega = (0, ell)^2,

    u_tt - (alpha + beta * |grad u|^2) * Laplace(u) = 0,

the eigenfunction phi(x, y) = sin(lam*pi*x/ell) * sin(lam*pi*y/ell) satisfies
-Laplace(phi) = mu*phi with mu = 2*(lam*pi/ell)^2.  Both three-layer schemes map
the span of phi into itself, so if the initial data lie in that span the whole
discrete evolution does, and every scheme reduces *exactly* to a scalar
three-layer recursion for the coefficient c_k in u_k = c_k * phi.

This makes the experiment free of any spatial discretization error: what is
measured is the time discretization alone.  That is what the schemes' energy
conservation is a statement about, so no finite element computation is needed
here -- and none is used, so this script depends only on the standard library.

E1 is the invariant of the locally linear scheme, E2 that of the nonlinear
one; E* is the second-order approximation of the continuous energy E, and K
denotes kinetic energy (the manuscript writes T for the final time only).

Usage:
    python examples/run_energy_conservation.py [--outdir ../figures]
"""

from __future__ import annotations

import argparse
import math
import os


# ----------------------------------------------------------------------
#  Problem data
# ----------------------------------------------------------------------

class Problem:
    """The scalar reduction of the homogeneous Kirchhoff problem."""

    def __init__(self, alpha: float = 1., beta: float = 1., lam: int = 1,
                 ell: float = 1., a: float = 1., b: float = 0.) -> None:
        self.alpha, self.beta, self.lam, self.ell = alpha, beta, lam, ell
        self.a, self.b = a, b
        k = lam * math.pi / ell
        self.mu = 2. * k * k
        self.M = (ell / 2.) ** 2
        self.G = self.mu * self.M

    # --- energies -----------------------------------------------------
    def V(self, c: float) -> float:
        return .5 * c * c * self.G

    def E_exact(self, c: float, cp: float) -> float:
        v = self.V(c)
        return .5 * cp * cp * self.M + self.alpha * v + self.beta * v * v

    def accel(self, c: float) -> float:
        """c'' = -mu * (alpha + beta * G * c^2) * c."""
        return -self.mu * (self.alpha + self.beta * self.G * c * c) * c

    # --- start-up values, eq. (2.4)-(2.5) of the manuscript -----------
    def start(self, tau: float) -> tuple[float, float]:
        q0 = self.alpha + self.beta * self.G * self.a * self.a
        # phi_2 = f_0 + q_0*Laplace(phi_0)
        phi2 = -q0 * self.mu * self.a
        c0 = self.a
        c1 = self.a + tau * self.b + .5 * tau * tau * phi2
        return c0, c1


# ----------------------------------------------------------------------
#  The two schemes, in their scalar form
# ----------------------------------------------------------------------

def step_linear(p: Problem, tau: float, c_prev: float, c_cur: float) -> float:
    """Locally linear scheme: c_{k+1} = 2 c_k / (1 + A_k) - c_{k-1}."""
    q = p.alpha + p.beta * p.G * c_cur * c_cur
    A = .5 * p.mu * tau * tau * q
    return 2. * c_cur / (1. + A) - c_prev


def step_nonlinear(p: Problem, tau: float, c_prev: float, c_cur: float,
                   tol: float = 1e-15, max_it: int = 50) -> float:
    """Genuinely nonlinear scheme, solved by Newton's method in c_{k+1}."""
    h = .5 * p.mu * tau * tau
    bg = .5 * p.beta * p.G
    c2 = c_prev * c_prev

    def F(z: float) -> float:
        return (z - 2. * c_cur + c_prev
                + h * (p.alpha + bg * (z * z + c2)) * (z + c_prev))

    def dF(z: float) -> float:
        return 1. + h * (2. * bg * z * (z + c_prev)
                         + p.alpha + bg * (z * z + c2))

    z = step_linear(p, tau, c_prev, c_cur)
    for _ in range(max_it):
        fz = F(z)
        dz = fz / dF(z)
        z -= dz
        if abs(dz) <= tol * max(1., abs(z)):
            break
    return z


def step_fixedpoint(p: Problem, tau: float, c_prev: float, c_cur: float,
                    tol: float = 1e-14, max_it: int = 200):
    """The analysed fixed-point iteration, in its scalar form.

    This is the iteration whose geometric convergence is the subject of the
    one-step solvability theorem, *not* a Newton solve.  Writing
    v = (u_{k+1} + u_{k-1}) / 2 turns the nonlinear scheme into T_k v = g_k
    with g_k = (tau^2 / 2) f_k + u_k, and the iteration freezes the nonlinear
    coefficient at the previous iterate:

        v^{(m+1)} = g_k / ( 1 + (tau^2 / 2) mu [ alpha
                     + (beta G / 2) ( (2 v^{(m)} - c_prev)^2 + c_prev^2 ) ] ),

    started from v^{(0)} = (c_cur + c_prev) / 2, which is the initial iterate
    fixed in the analysis.  Returns (c_next, iterations, contraction).

    `contraction` is |v^(2) - v^(1)| / |v^(1) - v^(0)|, i.e. the ratio at the
    second iteration.  It is taken there rather than at the last step because
    once the increments approach round-off the measured ratio is dominated by
    cancellation and says nothing about the iteration.
    """
    g = c_cur                                # homogeneous problem: f_k = 0
    v = .5 * (c_cur + c_prev)
    d_prev = None
    contraction = float('nan')
    for it in range(1, max_it + 1):
        u_m = 2. * v - c_prev                # u_{k+1}^{(m)}
        q = p.alpha + .5 * p.beta * p.G * (u_m * u_m + c_prev * c_prev)
        v_new = g / (1. + .5 * tau * tau * p.mu * q)
        d = abs(v_new - v)
        if it == 2 and d_prev is not None and d_prev > 1e-300:
            contraction = d / d_prev
        v, d_prev = v_new, d
        if d <= tol * max(1., abs(v)):
            return 2. * v - c_prev, it, contraction
    return 2. * v - c_prev, max_it, contraction


SCHEMES = {'linear': step_linear, 'nonlinear': step_nonlinear}


def run(p: Problem, tau: float, n: int, scheme: str) -> list[float]:
    """Returns [c_0, ..., c_n]."""
    step = SCHEMES[scheme]
    c0, c1 = p.start(tau)
    c = [c0, c1]
    for _ in range(1, n):
        c.append(step(p, tau, c[-2], c[-1]))
    return c


# ----------------------------------------------------------------------
#  Reference solution: classical RK4 on (c, c') at a far finer step
# ----------------------------------------------------------------------

def reference(p: Problem, t_end: float, n_ref: int) -> tuple[float, float]:
    """Integrates to t_end and returns (c, c') there."""
    h = t_end / n_ref
    c, v = p.a, p.b
    for _ in range(n_ref):
        k1c, k1v = v, p.accel(c)
        k2c, k2v = v + .5 * h * k1v, p.accel(c + .5 * h * k1c)
        k3c, k3v = v + .5 * h * k2v, p.accel(c + .5 * h * k2c)
        k4c, k4v = v + h * k3v, p.accel(c + h * k3c)
        c += h / 6. * (k1c + 2. * k2c + 2. * k3c + k4c)
        v += h / 6. * (k1v + 2. * k2v + 2. * k3v + k4v)
    return c, v


# ----------------------------------------------------------------------
#  Experiments
# ----------------------------------------------------------------------

def energy_drift(p: Problem, t_end: float, n: int, outdir: str) -> None:
    """Invariants over all time levels, at a fixed tau, for both schemes."""
    tau = t_end / n
    rows: list[tuple[float, float, float]] = []
    for scheme, key in (('linear', 'E1'), ('nonlinear', 'E2')):
        c = run(p, tau, n, scheme)
        vals = []
        for k in range(1, len(c)):
            Vkm, Vk = p.V(c[k - 1]), p.V(c[k])
            Tk = .5 * ((c[k] - c[k - 1]) / tau) ** 2 * p.M
            if key == 'E1':
                e = Tk + .5 * p.alpha * (Vkm + Vk) + p.beta * Vkm * Vk
            else:
                e = Tk + .5 * p.alpha * (Vkm + Vk) \
                    + .5 * p.beta * (Vkm * Vkm + Vk * Vk)
            vals.append(e)
        e1 = vals[0]
        rows.append((scheme, key, [(k + 1) * tau for k in range(len(vals))],
                     [abs(v - e1) / abs(e1) for v in vals]))

    path = os.path.join(outdir, 'energy_drift.csv')
    with open(path, 'w') as fh:
        fh.write('t,drift_E1,drift_E2\n')
        ts, d1 = rows[0][2], rows[0][3]
        d2 = rows[1][3]
        stride = max(1, len(ts) // 2000)
        for i in range(0, len(ts), stride):
            if d1[i] > 0. and d2[i] > 0.:
                fh.write(f'{ts[i]},{d1[i]},{d2[i]}\n')
    print(f'  wrote {path}  (max drift: E1 {max(rows[0][3]):.3e}, '
          f'E2 {max(rows[1][3]):.3e})')


def energy_approximation(p: Problem, t_end: float, ns: list[int],
                         outdir: str) -> None:
    """max_k |E(t_k) - E*_k| against tau, for the nonlinear scheme."""
    path = os.path.join(outdir, 'energy_approximation.csv')
    E0 = p.E_exact(p.a, p.b)
    with open(path, 'w') as fh:
        fh.write('tau,err\n')
        prev = None
        for n in ns:
            tau = t_end / n
            c = run(p, tau, n, 'nonlinear')
            worst = 0.
            for k in range(1, len(c) - 1):
                Tst = .5 * ((c[k + 1] - c[k - 1]) / (2. * tau)) ** 2 * p.M
                Vk = p.V(c[k])
                worst = max(worst, abs(E0 - (Tst + p.alpha * Vk
                                             + p.beta * Vk * Vk)))
            rate = '' if prev is None else \
                f'   rate {math.log(prev / worst) / math.log(2.):.2f}'
            print(f'  tau={tau:.3e}  max|E-E*|={worst:.6e}{rate}')
            prev = worst
            fh.write(f'{tau},{worst}\n')
    print(f'  wrote {path}')


def temporal_order(p: Problem, t_end: float, ns: list[int],
                   outdir: str, n_ref: int) -> None:
    """Error at the final time against tau, for both schemes."""
    c_ref, _ = reference(p, t_end, n_ref)
    print(f'  reference c(T) = {c_ref:.15e}  (RK4, {n_ref} steps)')
    path = os.path.join(outdir, 'temporal_order.csv')
    with open(path, 'w') as fh:
        fh.write('tau,err_linear,err_nonlinear\n')
        prev = {}
        for n in ns:
            tau = t_end / n
            errs = {}
            for scheme in ('linear', 'nonlinear'):
                c = run(p, tau, n, scheme)
                errs[scheme] = abs(c[-1] - c_ref) / abs(c_ref)
            msg = f'  tau={tau:.3e}'
            for scheme in ('linear', 'nonlinear'):
                r = '' if scheme not in prev else \
                    f' (rate {math.log(prev[scheme] / errs[scheme]) / math.log(2.):.2f})'
                msg += f'  {scheme}: {errs[scheme]:.4e}{r}'
            print(msg)
            prev = errs
            fh.write(f'{tau},{errs["linear"]},{errs["nonlinear"]}\n')
    print(f'  wrote {path}')


# ----------------------------------------------------------------------
#  Nonautonomous experiment
#
#  With alpha(t), beta(t) time dependent and a manufactured solution
#  u(x, y, t) = g(t) * phi(x, y), the source term is f = F(t) * phi with
#
#      F(t) = g''(t) + mu * (alpha(t) + beta(t) * G * g(t)^2) * g(t),
#
#  so the exact coefficient is g itself and no reference integrator is
#  needed.  The reduction to a scalar recursion is exactly as above.
# ----------------------------------------------------------------------

class Nonautonomous:
    """A manufactured nonautonomous test case, reduced to one scalar mode."""

    def __init__(self, name, alpha, beta, g, dg, ddg, t_end,
                 lam: int = 1, ell: float = 1.) -> None:
        self.name = name
        self.alpha, self.beta = alpha, beta
        self.g, self.dg, self.ddg = g, dg, ddg
        self.t_end = t_end
        k = lam * math.pi / ell
        self.mu = 2. * k * k
        self.M = (ell / 2.) ** 2
        self.G = self.mu * self.M

    def F(self, t: float) -> float:
        gt = self.g(t)
        return self.ddg(t) + self.mu * (self.alpha(t)
                                        + self.beta(t) * self.G * gt * gt) * gt

    def start(self, tau: float) -> tuple[float, float]:
        c0 = self.g(0.)
        q0 = self.alpha(0.) + self.beta(0.) * self.G * c0 * c0
        phi2 = self.F(0.) - q0 * self.mu * c0
        return c0, c0 + tau * self.dg(0.) + .5 * tau * tau * phi2

    def step_linear(self, tau: float, t_k: float,
                    c_prev: float, c_cur: float) -> float:
        q = self.alpha(t_k) + self.beta(t_k) * self.G * c_cur * c_cur
        A = .5 * self.mu * tau * tau * q
        return (2. * c_cur + tau * tau * self.F(t_k)) / (1. + A) - c_prev

    def step_nonlinear(self, tau: float, t_k: float,
                       c_prev: float, c_cur: float,
                       tol: float = 1e-15, max_it: int = 50) -> float:
        h = .5 * self.mu * tau * tau
        a_k, b_k = self.alpha(t_k), self.beta(t_k)
        bg = .5 * b_k * self.G
        c2 = c_prev * c_prev
        rhs = tau * tau * self.F(t_k)

        def Ff(z: float) -> float:
            return (z - 2. * c_cur + c_prev
                    + h * (a_k + bg * (z * z + c2)) * (z + c_prev) - rhs)

        def dF(z: float) -> float:
            return 1. + h * (2. * bg * z * (z + c_prev)
                             + a_k + bg * (z * z + c2))

        z = self.step_linear(tau, t_k, c_prev, c_cur)
        for _ in range(max_it):
            dz = Ff(z) / dF(z)
            z -= dz
            if abs(dz) <= tol * max(1., abs(z)):
                break
        return z

    def run(self, tau: float, n: int, scheme: str) -> float:
        """Returns the approximation of g(t_end)."""
        c_prev, c_cur = self.start(tau)
        step = self.step_linear if scheme == 'linear' else self.step_nonlinear
        for k in range(1, n):
            c_prev, c_cur = c_cur, step(tau, k * tau, c_prev, c_cur)
        return c_cur


def nonautonomous_order(cases: list[Nonautonomous], ns: list[int],
                        outdir: str) -> None:
    for case in cases:
        print(f'  --- {case.name} (T = {case.t_end:.6f}) ---')
        exact = case.g(case.t_end)
        path = os.path.join(outdir, f'nonautonomous_{case.name}.csv')
        with open(path, 'w') as fh:
            fh.write('tau,err_linear,err_nonlinear\n')
            prev: dict[str, float] = {}
            for n in ns:
                tau = case.t_end / n
                errs = {s: abs(case.run(tau, n, s) - exact) / abs(exact)
                        for s in ('linear', 'nonlinear')}
                msg = f'    tau={tau:.3e}'
                for s in ('linear', 'nonlinear'):
                    r = '' if s not in prev else \
                        f' (rate {math.log(prev[s] / errs[s]) / math.log(2.):.2f})'
                    msg += f'  {s}: {errs[s]:.4e}{r}'
                print(msg)
                prev = errs
                fh.write(f'{tau},{errs["linear"]},{errs["nonlinear"]}\n')
        print(f'    wrote {path}')


def fixed_point_study(p: Problem, t_end: float, ns: list[int],
                      outdir: str) -> None:
    """First-step contraction ratio and iteration count of the analysed
    fixed-point iteration, against the time step size.
    """
    path = os.path.join(outdir, 'fixed_point.csv')
    with open(path, 'w') as fh:
        fh.write('tau,q_first_step_max,iter_max\n')
        for n in ns:
            tau = t_end / n
            c_prev, c_cur = p.start(tau)
            q_max, it_max = 0., 0
            for _ in range(1, n):
                c_next, it, q = step_fixedpoint(p, tau, c_prev, c_cur)
                if q == q:                            # skip NaN
                    q_max = max(q_max, q)
                it_max = max(it_max, it)
                c_prev, c_cur = c_cur, c_next
            print(f'  tau={tau:.3e}   q_max={q_max:.4e}   '
                  f'q_max/tau={q_max / tau:.3f}   max iterations={it_max}')
            fh.write(f'{tau},{q_max},{it_max}\n')
    print(f'  wrote {path}')


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.normpath(os.path.join(here, '..', '..', 'figures'))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default=default_out)
    ap.add_argument('--alpha', type=float, default=1.)
    ap.add_argument('--beta', type=float, default=1.)
    ap.add_argument('--lam', type=int, default=1)
    ap.add_argument('--final-time', type=float, default=10.)
    ap.add_argument('--drift-final-time', type=float, default=100.,
                    help='longer horizon for the invariant-drift test')
    ap.add_argument('--n-drift', type=int, default=200000)
    ap.add_argument('--n-ref', type=int, default=4000000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    p = Problem(alpha=args.alpha, beta=args.beta, lam=args.lam, a=1., b=0.)
    print(f'mu = {p.mu:.6f},  M = {p.M},  G = {p.G:.6f},  '
          f'E(0) = {p.E_exact(p.a, p.b):.12f}')

    print(f'\n[1/5] energy drift of the discrete invariants '
          f'(T = {args.drift_final_time:g})')
    energy_drift(p, args.drift_final_time, args.n_drift, args.outdir)

    ns = [2 ** j for j in range(10, 17)]
    print('\n[2/5] approximation of the continuous energy by E*')
    energy_approximation(p, args.final_time, ns, args.outdir)

    print('\n[3/5] temporal order at the final time')
    temporal_order(p, args.final_time, ns, args.outdir, args.n_ref)

    cases = [
        Nonautonomous(
            'A',
            alpha=lambda t: 1. + t * t,
            beta=lambda t: 2. - math.sin(t),
            g=math.sin,
            dg=math.cos,
            ddg=lambda t: -math.sin(t),
            t_end=math.pi / 2.),
        Nonautonomous(
            'B',
            alpha=lambda t: 1. + 4. * t,
            beta=lambda t: 2. - 4. * t,
            g=lambda t: math.exp(math.pi * t),
            dg=lambda t: math.pi * math.exp(math.pi * t),
            ddg=lambda t: math.pi ** 2 * math.exp(math.pi * t),
            t_end=.25),
    ]
    print('\n[4/5] temporal order with time-dependent alpha and beta')
    nonautonomous_order(cases, [2 ** j for j in range(8, 15)], args.outdir)

    print('\n[5/5] geometric convergence of the analysed fixed-point iteration')
    fixed_point_study(p, args.final_time, [2 ** j for j in range(7, 14)],
                      args.outdir)


if __name__ == '__main__':
    main()
