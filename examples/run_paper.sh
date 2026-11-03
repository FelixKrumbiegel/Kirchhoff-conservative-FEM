#!/usr/bin/env bash
#
# The production runs: every example at the settled parameters, and the
# energy runs. No parameter studies; those were done once and their
# answers are baked into the values below.
#
#   ./run_paper.sh            all of it
#   ./run_paper.sh conservation   one group
#   ./run_paper.sh --list     group names
#   DRY=1 ./run_paper.sh      print the commands, run nothing
#   NP=4 ./run_paper.sh       on four ranks
#
# Output goes to results_paper/, separate from the exploratory
# results/, so manuscript data cannot be mixed with a trial run.
#
# ----------------------------------------------------------------------
# The parameters, and the measurement behind each
# ----------------------------------------------------------------------
# theta = 1/2   the value the energy identity is proved for, and the
#               only one at which the discrete energy is conserved:
#               -4e-8 against -4.6e-5 at theta = 1/4. Also
#               unconditionally stable, where theta = 0.2 diverges at
#               tau/h = 0.785.
#
# eta = 1/4     thirteen times more accurate than eta = 0 or 1/2 at the
#               same discretization, 3.06e-7 against 4.05e-6 and
#               3.44e-6, symmetric about 1/4 and order 2 throughout.
#               The leading term of the source mean cancels there.
#
# gamma = 1/2   free choice: errors agree within 15% and EOCt is 2.00
#               for every gamma in [0.1, 0.5]. Only the Picard count
#               moves, 3.95 to 3.99.
#
# picard 1e-10  the error floor is n_steps * ksp_rtol and does not
# ksp    1e-12  involve the Picard tolerance: (1e-10, 1e-12) and
#               (1e-12, 1e-12) agree to five digits at every level.
#               1e-14 costs 50% more Krylov iterations to move a floor
#               already two decades below the finest spatial error.
#
# pc = jacobi   same errors as amg at a quarter of the runtime, and the
#               only preconditioner whose iteration count does not
#               depend on the rank count. A = M + cK is mass dominated
#               at these tau/h, so a diagonal suffices.
#
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/results_paper"
NP="${NP:-1}"
MPIRUN="${MPIRUN:-mpirun}"

# One thread per rank: the work is sparse matvecs, which are bandwidth
# bound, so threads inside a rank only contend for the same memory and
# make the timings unrepeatable.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ "${NP}" -gt 1 ]]; then
    RUN="${MPIRUN} -n ${NP} python ${HERE}/run_example.py"
else
    RUN="python ${HERE}/run_example.py"
fi

SCHEME="--theta 0.5 --eta 0.25 --gamma 0.5"
SOLVER="--pc jacobi --picard-tolerance 1e-10 --ksp-rtol 1e-12"
COMMON="${SCHEME} ${SOLVER} --out-dir ${OUT}"

LOG="${OUT}/run_paper_np${NP}.log"
mkdir -p "${OUT}"

run() {
    local label="$1"; shift
    if [[ "${DRY:-0}" == "1" ]]; then printf '%s\n' "$*"; return 0; fi
    printf '\n\033[1m=== %s ===\033[0m\n' "${label}"
    printf '\n=== %s ===\n%s\n' "${label}" "$*" >> "${LOG}"
    local started=${SECONDS}
    if "$@" >> "${LOG}" 2>&1; then
        printf '    ok (%ss)\n' "$((SECONDS - started))"
    else
        printf '    FAILED after %ss, see %s\n' \
            "$((SECONDS - started))" "${LOG}"
    fi
}

# The energy figures are drawn from files extract_energy.py derives
# from the run output: the drift from the conservation group's series,
# the spread from the convergence group's rows in summary.csv. Either
# group can be run on its own, so both call this; it writes whatever it
# finds and says nothing about the rest.
write_energy_plots() {
    [[ "${DRY:-0}" == "1" ]] && return 0
    if python "${HERE}/extract_energy.py" "${OUT}" >> "${LOG}" 2>&1; then
        printf '    energy plot files written\n'
    else
        printf '    extract_energy.py failed, see %s\n' "${LOG}"
    fi
}

# ----------------------------------------------------------------------
# Spatial convergence, every example
# ----------------------------------------------------------------------
# EOCh reads p+1 in L2 and p in H1. Writes errors_space_*_ex*.csv,
# keyed by H with one column per degree.
#
# nt = 4096: the floor is proportional to n_steps, so a quarter of
# 16384 is a quarter of the floor, 4.1e-9 against 1.6e-8, while the
# temporal error at nt=4096 is 6.4e-8, a factor of 22 below the 1.4e-6
# spatial error at nx=64, p=3. Cheaper and more accurate.
#
# The 3xx family gets four times the steps: its pulse is the fastest in
# the set, omega = lam1 pi / (2 T) with lam1 up to 5, and at nt=4096 its
# temporal error still sits above the spatial one, which showed up as
# EOCh 1.63 where the others gave 3.
group_space() {
    # `1` and `11` are the same example; 11 is used so that `1`
    # stays free as the default of --ex without a duplicate table.
    for ex in 11 12 13 21 22 23 41 42 43; do
        run "space ex${ex}" ${RUN} --ex "${ex}" --tag space \
            --nx 4 --nx-levels 5 --nt 4096 --degrees 1,2,3 \
            ${COMMON} --tables --no-series
    done
    for ex in 311 321 331 313 323 333 315 325 335; do
        run "space ex${ex}" ${RUN} --ex "${ex}" --tag space \
            --nx 4 --nx-levels 5 --nt 16384 --degrees 1,2,3 \
            ${COMMON} --tables --no-series
    done
}

# ----------------------------------------------------------------------
# Temporal convergence, one example per family
# ----------------------------------------------------------------------
# EOCt reads 2 in both norms. nx=64 and p=3 put the spatial error at
# 1.4e-6, below the temporal error at every level; four levels stop at
# nt=4096, before the 4.1e-9 Krylov floor.
group_time() {
    # One per family, except the 3xx family, where lam1 is the
    # temporal frequency: 311, 313, 315 vary it at fixed lam, so a
    # temporal study has three points of comparison rather than one.
    for ex in 11 21 311 313 315 41; do
        run "time ex${ex}" ${RUN} --ex "${ex}" --tag time \
            --nx 64 --nt 512 --nt-levels 4 --degrees 3 \
            ${COMMON} --tables --no-series
    done
}

# ----------------------------------------------------------------------
# Energy conservation: theta = 1/2, three values of gamma
# ----------------------------------------------------------------------
# A statement at fixed tau: the discrete energy is constant for every
# tau, not merely in the limit. So this needs the time series and a long
# horizon, and no refinement at all.
#
# A hundred periods at tau = 6.2831853/8192 = 7.67e-4, which is the
# third of the four levels the convergence group runs, so the two
# figures share a time step and the conservation claim can be read
# against a convergence row at the same tau. It is not the finest
# convergence level: a hundred periods at that tau would be 1.6 million
# steps for no extra information.
#
# gamma enters the conserved functional explicitly, through the
# (beta/4) gamma (S_n - S_{n-1})^2 term, so the identity is worth
# showing across it rather than at one value. gamma = 0 is the
# degenerate end: the nonlinearity is then evaluated at u^n alone, the
# correction term vanishes, and the scheme is linearized, with
# picard_mean reading exactly 2. That the identity survives it is the
# strongest form of the claim.
#
# Plot (E(t) - E(0)) / E(0) from the series files, so all three start at
# zero despite sitting at slightly different levels.
#
# Example 5 has no exact solution, only a placeholder, so the error
# columns of these rows are meaningless and the relative ones come back
# as nan by design. The series file is the entire output here; nothing
# in summary.csv from this group belongs in the manuscript except
# energy_drift and energy_spread.
#
# gamma = 0 needs `assert config.gamma != 0.` removed from
# NonlinearKirchhoffThetaScheme.__init__.
#
# picard-max is raised because 1e-12 needs more passes than the default
# cap of 10 allows; without it a step hits the cap, the run is marked
# stalled, and the group reports a failure.
group_conservation() {
    local tight="--pc jacobi --picard-tolerance 1e-12 --ksp-rtol 1e-13 \
                 --picard-max 30"
    for value in 0.5 0.25 0.; do
        run "conservation ex5 gamma=${value}" ${RUN} --ex 5 \
            --tag "conservation-g${value}" \
            --nx 32 --nt 819200 --degrees 2 \
            --theta 0.5 --eta 0.25 --gamma "${value}" ${tight} \
            --out-dir "${OUT}" --final-time 628.31853
    done

    write_energy_plots
}

# ----------------------------------------------------------------------
# Energy convergence: theta = 1/4, four time steps
# ----------------------------------------------------------------------
# A statement about tau: away from theta = 1/2 the energy is conserved
# only in the limit, and the defect follows
#
#     dE/E  ~  C (1/2 - theta) tau^2
#
# The tau^2 is the four levels below; the (1/2 - theta) factor is the
# ratio -4.59e-5 / -3.67e-5 = 1.25 between theta = 1/4 and 3/10, which
# matches 0.25 / 0.20 exactly.
#
# One gamma suffices: at theta = 1/4 the drift was -4.59e-5, -4.6e-5
# and -4.6e-5 for gamma of 0.1, 0.25 and 0.5, so the defect does not
# depend on it.
#
# One period rather than a hundred: the rate is already clean there,
# -4.59e-5, -1.15e-5, -2.85e-6, falling by exactly four per halving,
# and a long horizon costs a hundred times as much to show the same
# exponent. No series, the drift column is the whole result.
group_convergence() {
    run "convergence ex5 theta=0.25" ${RUN} --ex 5 \
        --tag convergence-t0.25 \
        --nx 32 --nt 2048 --nt-levels 4 --degrees 2 \
        --theta 0.25 --eta 0.25 --gamma 0.5 ${SOLVER} \
        --out-dir "${OUT}" --final-time 6.2831853 --no-series

    # The same refinement at theta = 1/2, as the baseline the other
    # converges towards: here the drift should stay at the solver noise
    # level and show no tau dependence at all. That contrast, a rate of
    # 2 against no rate, is what separates a consistency defect from
    # numerical noise.
    run "convergence ex5 theta=0.5" ${RUN} --ex 5 \
        --tag convergence-t0.5 \
        --nx 32 --nt 2048 --nt-levels 4 --degrees 2 \
        --theta 0.5 --eta 0.25 --gamma 0.5 ${SOLVER} \
        --out-dir "${OUT}" --final-time 6.2831853 --no-series

    write_energy_plots
}

# ----------------------------------------------------------------------
# The linearized scheme: gamma = 0
# ----------------------------------------------------------------------
# One example run both ways, so the linearization can be read directly
# against the same rows at gamma = 1/2 in the space and time groups.
#
# At gamma = 0 the nonlinear coefficient is evaluated at u^n alone, so
# Gamma stops depending on the iterate and the Picard loop converges in
# a single pass: picard_mean should read exactly 2, one update and one
# confirmation. If it reads more, the linearization is not what it
# claims to be.
#
# Example 41 because it is nonautonomous: alpha and beta move over the
# run, so an explicit treatment of the nonlinearity is actually being
# tested rather than sitting on a constant.
#
# Needs `assert config.gamma != 0.` removed from
# NonlinearKirchhoffThetaScheme.__init__.
group_linearized() {
    run "linearized ex41 space" ${RUN} --ex 41 --tag linearized-space \
        --nx 4 --nx-levels 5 --nt 4096 --degrees 1,2,3 \
        --theta 0.5 --eta 0.25 --gamma 0. ${SOLVER} \
        --out-dir "${OUT}" --tables --table-suffix _g0 --no-series

    run "linearized ex41 time" ${RUN} --ex 41 --tag linearized-time \
        --nx 64 --nt 512 --nt-levels 4 --degrees 3 \
        --theta 0.5 --eta 0.25 --gamma 0. ${SOLVER} \
        --out-dir "${OUT}" --tables --table-suffix _g0 --no-series
}

# Not GROUPS: bash owns that name for the current user's supplementary
# group ids, so assigning to it is silently ignored.
PAPER_GROUPS=(space time conservation convergence linearized)

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${PAPER_GROUPS[@]}"
    exit 0
fi

[[ "${DRY:-0}" == "1" ]] || : > "${LOG}"
for name in "${@:-${PAPER_GROUPS[@]}}"; do
    "group_${name}"
done

if [[ "${DRY:-0}" != "1" ]]; then
    printf '\n\033[1mdone.\033[0m %s\n' "${OUT}/summary.csv"
fi
