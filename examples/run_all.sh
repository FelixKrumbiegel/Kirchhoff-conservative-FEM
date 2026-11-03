#!/usr/bin/env bash
#
# Runs every study. Each invocation appends to results/summary.csv with
# its own --tag, so the whole campaign is one table to filter:
#
#   awk -F, 'NR==1 || $1=="pc-mass"' results/summary.csv
#
#   ./run_all.sh                 everything, full size
#   QUICK=1 ./run_all.sh         a smoke test, minutes not hours
#   DRY=1 ./run_all.sh           print the commands, run nothing
#   NP=8 ./run_all.sh            run each case on 8 ranks
#   ./run_all.sh preconditioner  one section
#   ./run_all.sh --skip tolerance   everything except that one
#   ./run_all.sh --list          section names
#
# NP selects the number of MPI ranks per run; the default of 1 launches
# python directly, without a launcher. MPIRUN overrides the launcher for
# a scheduler, MPIRUN='srun' or MPIRUN='mpirun --oversubscribe'.
#
# The runs are sequential, one MPI job at a time. Only rank 0 prints and
# writes, so summary.csv gets one row per run however many ranks there
# are, and the numbers do not depend on NP.
#
# Stability runs are expected to fail, so `set -e` is not used; the exit
# status of each run is recorded instead. run_example.py returns nonzero
# when any run in it diverged or stalled.
#
# What dominates the cost: every Picard iteration changes the system
# matrix, so hypre rebuilds its hierarchy, which costs several times a
# solve. A run is therefore roughly
#
#     steps * picard_iterations * (amg setup + cg solve)
#
# and the setup term is the big one. That is what the preconditioner
# section measures, and if --pc amg_reuse or --pc mass turns out to win
# clearly, it is worth making one of them the default for every other
# section too.
#
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/results"

NP="${NP:-1}"
MPIRUN="${MPIRUN:-mpirun}"

# One thread per rank. Without this, every rank starts as many
# BLAS/OpenMP threads as there are cores, so NP ranks oversubscribe the
# machine NP-fold. The work here is sparse matrix-vector products and
# vector updates, which are memory bandwidth bound: threads inside a
# rank contend for the same bandwidth rather than adding throughput, and
# the contention makes solve_time unrepeatable. Exported, so they reach
# the ranks through mpirun.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# One log per rank count and profile, so a second campaign at another NP
# does not overwrite the first. summary.csv needs no such treatment: it
# is appended to and carries a ranks column.
LOG="${OUT}/run_all_np${NP}$([[ "${QUICK:-0}" == "1" ]] && printf '_quick').log"

# Preconditioner for every section except the one that compares them.
# jacobi, on the evidence of that comparison: same errors, a quarter of
# the runtime, because A = M + cK is mass dominated at these tau/h and a
# diagonal is enough. Set PC=amg to go back.
PC="${PC:-jacobi}"

# Set from the tolerance section. The error floor is n_steps * ksp_rtol
# and nothing else: at nt=16384 that predicts 1.6e-8 and gives 1.69e-8,
# and the same errors come out at picard 1e-10 as at 1e-12, because the
# nonlinear iteration converges far below its tolerance in four passes.
#
# ksp 1e-14 is not worth it: it raises krylov_mean by half to move a
# floor already two decades below the finest spatial error, and below
# about 1e-12 the absolute tolerance of 1e-14 inside LinearSystem binds
# instead, which is why EOCt still degrades there.
PICARD_TOL="${PICARD_TOL:-1e-10}"
KSP_RTOL="${KSP_RTOL:-1e-12}"
TOLERANCES_ARG="--picard-tolerance ${PICARD_TOL} --ksp-rtol ${KSP_RTOL}"
if [[ "${NP}" -gt 1 ]]; then
    RUN="${MPIRUN} -n ${NP} python ${HERE}/run_example.py"
else
    RUN="python ${HERE}/run_example.py"
fi

# ----------------------------------------------------------------------
# Profiles. QUICK shrinks every section, not just some of them: the
# point of a smoke test is to exercise each code path once.
# ----------------------------------------------------------------------
# Which examples each section covers. The same in both profiles: a
# smoke test that skips examples is not testing the examples. QUICK
# shrinks the discretizations instead.
# Example 1 is reachable as `1` and as `11`; both are lambda = 1 and
# run identically. The runs below use 11 throughout, so that `1` stays
# free as the default of --ex without producing a duplicate table.
SPACE_EXAMPLES="11 12 13 21 22 23 311 321 331 313 323 333 315 325 335 41 42 43"
# One per family, except the 3xx family, where lam1 is the temporal
# frequency: 311, 313, 315 vary it at fixed lam, so a temporal study
# has three points of comparison rather than one.
TIME_EXAMPLES="11 21 311 313 315 41"
GRID_EXAMPLES="11 41"

if [[ "${QUICK:-0}" == "1" ]]; then
    # convergence, space
    SPACE_NX=4;   SPACE_LEVELS=3;  SPACE_NT=512;    SPACE_DEGREES=1,2
    SPACE_NT_FAST=4
    # convergence, time
    TIME_NX=16;   TIME_NT=128;     TIME_LEVELS=4;   TIME_DEGREE=2
    # grid
    GRID_NX=8;    GRID_NX_LEVELS=2
    GRID_NT=128;  GRID_NT_LEVELS=3; GRID_DEGREES=1
    # energy
    ENERGY_THETAS="0.25 0.5"
    ENERGY_NX="16";  ENERGY_NT=512; ENERGY_NT_LEVELS=2
    ENERGY_DEGREE=2; ENERGY_TIME=3.1415927
    # conservation: two periods at the energy section's finest tau
    CONS_NX=16;  CONS_NT=16384;  CONS_TIME=12.5663706
    # stability. nx=32 so that the coarsest tau/h reaches 0.785, which
    # is where theta = 0.2 actually diverges; at nx=16 the sweep tops
    # out at 0.393 and the section shows nothing.
    STABILITY_THETAS="0.2 0.25 0.5"
    STAB_NX=32;   STAB_NT=64;      STAB_LEVELS=4
    # preconditioner
    PC_NX="16 32";  PC_NT=512;     PC_DEGREE=2
    # tolerance
    TOL_NX=16;   TOL_NT=128;   TOL_LEVELS=4;   TOL_DEGREE=2
    # gamma and eta
    GAMMA_NX=16; GAMMA_NT=512; GAMMA_LEVELS=2; GAMMA_DEGREE=2
    GAMMA_ORDER_NX=32; GAMMA_ORDER_DEGREE=3
    ETA_NX=16;   ETA_NT=128;   ETA_LEVELS=3;   ETA_DEGREE=2
else
    # nt=4096 rather than 16384: the floor is proportional to n_steps,
    # so a quarter of the steps is a quarter of the floor, 4e-9 instead
    # of 1.6e-8, while the temporal error at 4096 is 6.4e-8, far below
    # the 1.4e-6 spatial error at nx=64, p=3. Cheaper and more accurate.
    SPACE_NX=4;   SPACE_LEVELS=5;  SPACE_NT=4096;   SPACE_DEGREES=1,2,3
    SPACE_NT_FAST=4

    # Four levels, not six: EOCt reads 2.00, 2.00, 1.99 through nt=4096
    # and then 1.58 and 0.33 as the floor takes over. Four clean points
    # beat six with two wrong ones, and it drops the two most expensive
    # runs in the campaign.
    TIME_NX=64;   TIME_NT=512;     TIME_LEVELS=4;   TIME_DEGREE=3

    GRID_NX=8;    GRID_NX_LEVELS=4
    GRID_NT=128;  GRID_NT_LEVELS=6; GRID_DEGREES=1,2

    ENERGY_THETAS="0.25 0.3 0.5"
    ENERGY_NX="16 32 64"; ENERGY_NT=2048; ENERGY_NT_LEVELS=3
    ENERGY_DEGREE=2;      ENERGY_TIME=6.2831853
    # conservation: ten periods at tau = 6.2831853/8192 = 7.67e-4, the
    # finest step the energy section reaches, so the two sit on the same
    # tau and only the horizon differs. A hundred periods showed nothing
    # the ten do not, at ten times the cost.
    CONS_NX=32;  CONS_NT=81920;  CONS_TIME=62.831853
    # A hundred periods at the tau the energy section uses for one.

    STABILITY_THETAS="0.1 0.2 0.24 0.25 0.26 0.5"
    STAB_NX=32;   STAB_NT=64;      STAB_LEVELS=5

    PC_NX="32 64";  PC_NT=2048;    PC_DEGREE=2

    TOL_NX=64;   TOL_NT=512;    TOL_LEVELS=6;   TOL_DEGREE=3
    GAMMA_NX=32; GAMMA_NT=1024; GAMMA_LEVELS=3; GAMMA_DEGREE=2
    # The order half of the gamma section needs a mesh fine enough that
    # the temporal error is visible: at nx=32, p=2 every gamma sat on
    # the spatial floor of 8.59e-6 and EOCt read 0, which shows that
    # gamma does not change the error but cannot show that it preserves
    # the order.
    GAMMA_ORDER_NX=64; GAMMA_ORDER_DEGREE=3
    # Four levels, not five. eta = 1/4 is a factor of 13 better than
    # the others, so it reaches the ksp floor two levels sooner: at
    # nt=8192 its error was 6.7e-9 against a floor of 8192*1e-12 =
    # 8.2e-9, and EOCt went 1.52 then -0.21. The other two are still
    # clean there, but a table where one column stops converging first
    # invites the wrong reading.
    ETA_NX=64;   ETA_NT=512;    ETA_LEVELS=4;   ETA_DEGREE=3
fi

# Values swept by the parameter sections. gamma and eta are both
# restricted to [0, 1/2] by the configuration.
# Pairs of "picard:ksp" tolerances. The middle two differ only in the
# Krylov tolerance, which is what separates a floor set by the nonlinear
# iteration from one set by the linear solves.
TOLERANCE_PAIRS="1e-10:1e-10 1e-12:1e-10 1e-12:1e-12 1e-14:1e-14"
GAMMA_VALUES="0.1 0.25 0.5"
# gamma enters the conserved functional explicitly, so the
# conservation section sweeps it; 0 is the degenerate end, where
# the correction term vanishes and the scheme is linearized.
CONSERVATION_GAMMAS="0.5 0.25 0."
ETA_VALUES="0. 0.25 0.5"

# The coarsest mesh of the space study is SPACE_NX squared, times two
# triangles. Splitting that over many ranks leaves a few cells each,
# where the partitioning overhead dominates and a rank can end up with
# none at all.
COARSEST_CELLS=$((2 * SPACE_NX * SPACE_NX))
if [[ "${NP}" -gt 1 && $((COARSEST_CELLS / NP)) -lt 16 ]]; then
    printf 'note: the coarsest level has %s cells over %s ranks.\n' \
        "${COARSEST_CELLS}" "${NP}"
    printf '      raise SPACE_NX or lower NP for those levels to mean much.\n'
fi

mkdir -p "${OUT}"

run() {
    # run <label> <args...>
    local label="$1"; shift
    if [[ "${DRY:-0}" == "1" ]]; then
        printf '%s\n' "$*"
        return 0
    fi
    printf '\n\033[1m=== %s ===\033[0m\n' "${label}"
    printf '\n=== %s ===\n%s\n' "${label}" "$*" >> "${LOG}"
    local started=${SECONDS}
    if "$@" >> "${LOG}" 2>&1; then
        printf '    ok (%ss)\n' "$((SECONDS - started))"
    else
        printf '    finished with failures after %ss (see %s)\n' \
            "$((SECONDS - started))" "${LOG}"
    fi
}

hint() {
    # hint <reason> <args...> : a run that is deliberately not made,
    # printed so the command is still to hand.
    local reason="$1"; shift
    printf '\n\033[1m=== not run ===\033[0m %s\n' "${reason}"
    printf '    to try it anyway:\n      %s\n' "$*"
}

section_convergence() {
    # Spatial order: the time step is fixed and small enough that the
    # temporal error stays below the spatial one. --tables writes the
    # pgfplots files of the manuscript.
    #
    # The 3xx family carries the fastest pulse, omega = lam1 pi / (2 T)
    # with lam1 up to 5, and its temporal error still sits above the
    # spatial one at the nt the other families need: example 315 read
    # EOCh 1.63 at p=2 where 311 and 313 gave 3. A spatial study has to
    # take nt from the fastest example, not the average, so that family
    # gets SPACE_NT_FAST times as many steps.
    for ex in ${SPACE_EXAMPLES}; do
        nt=${SPACE_NT}
        if [[ ${ex} -ge 300 && ${ex} -lt 400 ]]; then
            nt=$((SPACE_NT * SPACE_NT_FAST))
        fi
        run "space ex${ex} (nt=${nt})" ${RUN} --ex "${ex}" --tag space \
            --nx ${SPACE_NX} --nx-levels ${SPACE_LEVELS} --nt "${nt}" \
            --degrees ${SPACE_DEGREES} --pc ${PC} ${TOLERANCES_ARG} \
            --tables --no-series
    done

    # Temporal order: the mesh is fine and the degree high, so the
    # spatial error sits below the temporal one. This is the most
    # expensive section, since it is the finest mesh and the highest
    # degree carried over the most steps.
    for ex in ${TIME_EXAMPLES}; do
        run "time ex${ex}" ${RUN} --ex "${ex}" --tag time \
            --nx ${TIME_NX} --nt ${TIME_NT} --nt-levels ${TIME_LEVELS} \
            --degrees ${TIME_DEGREE} --pc ${PC} ${TOLERANCES_ARG} \
            --tables --no-series
    done
}

section_grid() {
    # The full h by tau cross product, from which optimal_tau_ex*.csv is
    # derived: for each mesh, the largest tau whose error still matches
    # the tau to zero limit.
    for ex in ${GRID_EXAMPLES}; do
        run "grid ex${ex}" ${RUN} --ex "${ex}" --tag grid \
            --nx ${GRID_NX} --nx-levels ${GRID_NX_LEVELS} \
            --nt ${GRID_NT} --nt-levels ${GRID_NT_LEVELS} \
            --degrees ${GRID_DEGREES} --pc ${PC} ${TOLERANCES_ARG} --no-series
    done
}

# The energy figures are drawn from files extract_energy.py derives from
# the run output: the drift from the conservation section's series, the
# spread from the energy section's rows in summary.csv. Either section
# can be run on its own, so both call this; it writes whatever it finds.
write_energy_plots() {
    [[ "${DRY:-0}" == "1" ]] && return 0
    if python "${HERE}/extract_energy.py" "${OUT}" >> "${LOG}" 2>&1; then
        printf '    energy plot files written\n'
    else
        printf '    extract_energy.py failed, see %s\n' "${LOG}"
    fi
}

section_energy() {
    # Example 5 is the unforced autonomous case, the only one where the
    # energy is conserved and the only one the runner records it for.
    # Series files carry energy against time; the summary carries the
    # drift and the peak to peak spread.
    #
    # This is the tau refinement at a short horizon: it answers whether
    # the defect falls like tau^2, not whether the energy stays put over
    # many periods. The conservation section answers that one.
    #
    # theta = 1/2 is the value the energy identity is proved for, and
    # the data agrees: the drift is -4e-8 there against -4.6e-5 at
    # theta = 1/4, and the latter falls by exactly four per halving of
    # tau, so it is an O(tau^2) defect rather than noise. Across theta
    # it follows dE/E ~ C (1/2 - theta) tau^2: the ratio -4.59e-5 /
    # -3.67e-5 = 1.25 matches 0.25 / 0.20 exactly.
    #
    # The mesh sweep is here to show the drift is purely temporal: it
    # came out identical to three digits at nx = 16, 32 and 64.
    #
    # Example 5 has no exact solution, only a placeholder, so the error
    # columns of these rows are meaningless and the relative ones come
    # back as nan by design.
    for theta in ${ENERGY_THETAS}; do
        for nx in ${ENERGY_NX}; do
            run "energy ex5 theta=${theta} nx=${nx}" \
                ${RUN} --ex 5 --tag "energy-t${theta}" \
                --nx "${nx}" --nt ${ENERGY_NT} \
                --nt-levels ${ENERGY_NT_LEVELS} \
                --degrees ${ENERGY_DEGREE} --theta "${theta}" \
                --pc ${PC} ${TOLERANCES_ARG} \
                --final-time ${ENERGY_TIME}
        done
    done
}

section_stability() {
    # theta >= 1/4 is the unconditional threshold. Below it the scheme
    # is only conditionally stable, so the runs are expected to diverge
    # once tau/h is large enough; the driver records that as a status
    # rather than a crash.
    for theta in ${STABILITY_THETAS}; do
        run "stability theta=${theta}" ${RUN} --ex 11 \
            --tag "stab-t${theta}" --theta "${theta}" \
            --nx ${STAB_NX} --nt ${STAB_NT} --nt-levels ${STAB_LEVELS} \
            --degrees 1 --pc ${PC} ${TOLERANCES_ARG} --no-series
    done

    # The two degenerate parameter values are not runnable, for
    # different reasons, so they are printed rather than attempted.
    #
    # theta = 0 is not a bug in the solver. The scheme solves for
    #
    #     v = theta u^{n+1} + (1 - 2 theta) u^n + theta u^{n-1}
    #
    # and recovers u^{n+1} = (v - (1-2 theta) u^n - theta u^{n-1})/theta.
    # That change of variable is only invertible for theta nonzero: at
    # theta = 0 the equation gives v = u^n and says nothing about
    # u^{n+1}. An explicit scheme has to be written as one, solving for
    # u^{n+1} against the mass matrix directly, rather than reached by
    # passing zero to this one.
    hint "theta = 0: the change of variable is not invertible there" \
        ${RUN} --ex 11 --tag degenerate-theta0 --theta 0. \
        --nx ${STAB_NX} --nt ${STAB_NT} --degrees 1 --no-series

    # gamma = 0 does run. The nonlinear coefficient is then evaluated at
    # u^n alone, so Gamma stops depending on the iterate and the Picard
    # loop converges in one pass: a linearized scheme rather than a
    # broken one. Worth having next to the others, since picard_mean in
    # summary.csv should read 1 for these rows and the errors show what
    # the linearization costs.
    #
    # This needs the `assert config.gamma != 0.` removed from
    # NonlinearKirchhoffThetaScheme.__init__.
    run "linearized gamma=0" ${RUN} --ex 11 --tag linearized-gamma0 \
        --gamma 0. --nx ${STAB_NX} --nt ${STAB_NT} \
        --nt-levels ${STAB_LEVELS} --degrees 1 --pc ${PC} \
        ${TOLERANCES_ARG} --no-series
}

section_preconditioner() {
    # Same problem four ways. Compare solve_time and krylov_mean in
    # summary.csv. The gap between amg and amg_reuse is the setup cost,
    # since nothing else differs between those two runs.
    for pc in amg amg_reuse mass jacobi; do
        for nx in ${PC_NX}; do
            run "pc=${pc} nx=${nx}" ${RUN} --ex 41 --tag "pc-${pc}" \
                --nx "${nx}" --nt ${PC_NT} --degrees ${PC_DEGREE} \
                --pc "${pc}" ${TOLERANCES_ARG} --no-series
        done
    done

    # The mass preconditioner should lose ground as tau/h grows, since
    # that is when A stops being mass dominated.
    for pc in amg mass; do
        run "pc=${pc} coarse time" ${RUN} --ex 41 --tag "pc-${pc}-coarse" \
            --nx 64 --nt 128 --nt-levels 4 --degrees ${PC_DEGREE} \
            --pc "${pc}" ${TOLERANCES_ARG} --no-series
    done
}

section_tolerance() {
    # The temporal study showed the L2 error at example 1 going
    #
    #     nt   512     1024     2048     4096     8192    16384
    #     L2   4.1e-6  1.0e-6   2.5e-7   1.3e-7   1.9e-7  2.2e-6
    #
    # order 2 for two refinements, then growth. A spatial floor would
    # plateau; growth means accumulation. Each step admits an error of
    # about the Picard tolerance and the steps add up, so the floor sits
    # near n_steps * tolerance: 16384 * 1e-10 is 1.6e-6, which is the
    # number observed.
    #
    # Running the same refinement at three tolerances settles it. If the
    # floor drops with the tolerance, the diagnosis holds and the
    # tolerance to quote is whichever keeps n_steps * tol below the
    # tau^2 being measured. If it stops dropping at 1e-12, the next
    # suspect is the Krylov rtol, which defaults to 1e-10 in
    # LinearSystem and is not yet reachable from the command line.
    #
    # picard-max is raised because a tighter tolerance needs more
    # iterations; a run that still hits the cap is reported as stalled,
    # which is itself the answer for that tolerance.
    #
    # Read absolute_l2 across the four tags at each ntime. Rows two and
    # three differ only in the Krylov tolerance, so if the floor moves
    # between them it is the linear solves; if it moves between rows one
    # and two it is the nonlinear iteration. The absolute Krylov
    # tolerance stays at 1e-14 throughout and may bind at the last
    # pair.
    for pair in ${TOLERANCE_PAIRS}; do
        picard="${pair%%:*}"
        ksp="${pair##*:}"
        run "tolerance picard=${picard} ksp=${ksp}" ${RUN} --ex 11 \
            --tag "tol-p${picard}-k${ksp}" \
            --nx ${TOL_NX} --nt ${TOL_NT} --nt-levels ${TOL_LEVELS} \
            --degrees ${TOL_DEGREE} \
            --picard-tolerance "${picard}" --ksp-rtol "${ksp}" \
            --picard-max 30 --pc ${PC} --no-series
    done
}

section_gamma() {
    # The energy half of this section moved to `conservation`, which
    # sweeps gamma at a long horizon and adds gamma = 0. What is left is
    # the order: gamma should not change it, only how many Picard
    # iterations reaching it costs.
    #
    # Compare EOCt and picard_mean across the rows. Example 41 is
    # nonautonomous, so the nonlinearity is doing something over the run.
    for value in ${GAMMA_VALUES}; do
        run "gamma=${value} order" ${RUN} --ex 41 \
            --tag "gamma-order-${value}" \
            --nx ${GAMMA_ORDER_NX} --nt ${GAMMA_NT} \
            --nt-levels ${GAMMA_LEVELS} --degrees ${GAMMA_ORDER_DEGREE} \
            --gamma "${value}" --pc ${PC} ${TOLERANCES_ARG} --no-series
    done
}

section_eta() {
    # eta weights the three point mean of the source. That mean is
    # second order for any eta in [0, 1/2], so the temporal order should
    # not move; what changes is the constant, since eta = 1/4 cancels
    # the leading term of the mean against the scheme's own.
    #
    # Only meaningful on a forced example: example 5 has no source, so
    # eta does nothing there.
    for value in ${ETA_VALUES}; do
        run "eta=${value}" ${RUN} --ex 11 --tag "eta-${value}" \
            --nx ${ETA_NX} --nt ${ETA_NT} --nt-levels ${ETA_LEVELS} \
            --degrees ${ETA_DEGREE} --eta "${value}" --pc ${PC} \
            ${TOLERANCES_ARG} --no-series
    done

    write_energy_plots
}

# tolerance leads, because its answer sets the --picard-tolerance the
# rest of the campaign should use, and finding that out afterwards means
# running everything twice.
section_conservation() {
    # Energy at fixed tau over many periods. The energy section refines
    # tau at one period and shows the defect falls like tau^2; only a
    # long horizon distinguishes a bounded oscillation from a drift that
    # accumulates, and that distinction is the claim.
    #
    # Ten periods, not a hundred: a secular drift is already ten times
    # larger there than over one period, well clear of the noise, and a
    # hundred costs ten times as much to show the same thing.
    #
    # Read energy_spread rather than energy_drift: over many periods a
    # drift that reverses averages away, while the spread does not.
    #
    # gamma enters the conserved functional explicitly, through the
    # (beta/4) gamma (S_n - S_{n-1})^2 term, so it is swept rather than
    # fixed. gamma = 0 is the degenerate end: the correction term
    # vanishes, the nonlinearity is evaluated at u^n alone and the
    # scheme is linearized, with picard_mean reading exactly 2. That the
    # identity survives it is the strongest form of the claim, and it
    # needs `assert config.gamma != 0.` removed from the solver.
    #
    # picard-max is raised because the tighter tolerance needs more
    # passes than the default cap of 10 allows; without it a step hits
    # the cap, the run is marked stalled, and the section reports a
    # failure.
    local tight="--pc ${PC} --picard-tolerance 1e-12 --ksp-rtol 1e-13 \
                 --picard-max 30"
    for value in ${CONSERVATION_GAMMAS}; do
        run "conservation ex5 gamma=${value}" ${RUN} --ex 5 \
            --tag "conservation-g${value}" \
            --nx ${CONS_NX} --nt ${CONS_NT} --degrees 2 \
            --theta 0.5 --gamma "${value}" ${tight} \
            --final-time ${CONS_TIME}
    done

    # The same horizon at theta = 1/4, where the defect is O(tau^2) per
    # step and should accumulate visibly. Without it the theta = 1/2
    # line has nothing to be flat against.
    run "conservation ex5 theta=0.25" ${RUN} --ex 5 \
        --tag conservation-t0.25 \
        --nx ${CONS_NX} --nt ${CONS_NT} --degrees 2 \
        --theta 0.25 --gamma 0.5 ${tight} --final-time ${CONS_TIME}

    write_energy_plots
}

section_linearized() {
    # One example at gamma = 0, run both ways, so the linearization can
    # be read directly against the same rows at gamma = 1/2 in the
    # convergence section.
    #
    # Gamma then stops depending on the iterate and the Picard loop
    # converges in a single pass: picard_mean should read exactly 2, one
    # update and one confirmation. More than that means the
    # linearization is not what it claims to be.
    #
    # Example 41 because it is nonautonomous: alpha and beta move over
    # the run, so an explicit treatment of the nonlinearity is actually
    # being tested rather than sitting on a constant.
    #
    # Needs `assert config.gamma != 0.` removed from the solver.
    run "linearized ex41 space" ${RUN} --ex 41 --tag linearized-space \
        --nx ${SPACE_NX} --nx-levels ${SPACE_LEVELS} --nt ${SPACE_NT} \
        --degrees ${SPACE_DEGREES} --gamma 0. --pc ${PC} \
        ${TOLERANCES_ARG} --tables --table-suffix _g0 --no-series

    run "linearized ex41 time" ${RUN} --ex 41 --tag linearized-time \
        --nx ${TIME_NX} --nt ${TIME_NT} --nt-levels ${TIME_LEVELS} \
        --degrees ${TIME_DEGREE} --gamma 0. --pc ${PC} \
        ${TOLERANCES_ARG} --tables --table-suffix _g0 --no-series
}

SECTIONS=(tolerance convergence grid energy conservation linearized
          stability preconditioner gamma eta)

if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${SECTIONS[@]}"
    exit 0
fi

# --skip a,b runs everything except those sections. Useful once the
# tolerance sweep has been read and its answer written into the
# defaults: there is no point paying for it again.
SKIP=""
if [[ "${1:-}" == "--skip" ]]; then
    SKIP=",${2//[[:space:]]/},"
    shift 2
fi

REQUESTED=("${@:-${SECTIONS[@]}}")
CHOSEN=()
for name in "${REQUESTED[@]}"; do
    if [[ -n "${SKIP}" && "${SKIP}" == *",${name},"* ]]; then
        printf 'skipping %s\n' "${name}"
        continue
    fi
    CHOSEN+=("${name}")
done

[[ "${DRY:-0}" == "1" ]] || : > "${LOG}"
for name in "${CHOSEN[@]}"; do
    "section_${name}"
done

if [[ "${DRY:-0}" != "1" ]]; then
    printf '\n\033[1mdone.\033[0m summary: %s\n' "${OUT}/summary.csv"
    printf 'log: %s\n' "${LOG}"
fi
