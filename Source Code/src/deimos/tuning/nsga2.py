"""Generic real-coded NSGA-II (Deb et al. 2002). No ADCS-specific knowledge
lives here -- see tuning/objectives.py for the PD/PID-gain fitness function
this drives.

Beyond textbook NSGA-II this adds four things an overnight run needs:

  parallel evaluation   every individual in a generation is scored
                        independently, so the generation is embarrassingly
                        parallel and the speedup is close to linear in cores
  hypervolume tracking  a single monotone quality number for the whole front
                        (see tuning/hypervolume.py for why per-objective
                        bests are the wrong convergence signal)
  plateau stopping      stop when hypervolume stops growing, rather than
                        burning the remaining budget on generations that
                        buy nothing
  a wall-clock deadline stop cleanly at the next generation boundary once
                        the budget is spent, so a fixed overnight window
                        produces a finished run rather than a truncated one
"""

import json
import os
import pickle
import time
import warnings
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import numpy as np

from deimos.tuning.hypervolume import StallDetector, hypervolume_fraction


def _dominates(a, b):
    """a dominates b: a <= b in every objective, a < b in at least one (minimization)."""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(objectives):
    """
    objectives: (N, M) array, M objectives, all minimized.
    Returns list of fronts (lists of indices); front[0] is Pareto-optimal
    within this population.
    """
    N = objectives.shape[0]
    dominates = [[] for _ in range(N)]
    dominated_count = np.zeros(N, dtype=int)
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominates[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(front_objectives):
    """front_objectives: (n, M) for ONE front. Higher = more isolated."""
    n, M = front_objectives.shape
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for m in range(M):
        order = np.argsort(front_objectives[:, m])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        f_min, f_max = front_objectives[order[0], m], front_objectives[order[-1], m]
        if f_max == f_min:
            continue
        for i in range(1, n - 1):
            distance[order[i]] += (
                (front_objectives[order[i + 1], m] - front_objectives[order[i - 1], m])
                / (f_max - f_min)
            )
    return distance


class NSGA2:
    """
    Real-coded NSGA-II (Deb et al. 2002). Minimizes every objective
    `evaluate` returns. Genes live in [0,1]^D internally -- `decode`
    maps that to the real search space (log-scale, physical bounds,
    whatever), keeping the GA operators scale-free.

    n_workers: number of processes used to evaluate one generation. Genes are
        decoded in THIS process and only the decoded values plus `evaluate`
        cross the process boundary, so `decode` may be any callable but
        `evaluate` must be picklable when n_workers > 1 (which is why
        objectives.Evaluator is a class and not a closure). n_workers=1 keeps
        everything in-process and imposes no picklability requirement at all,
        so the unit tests can go on using lambdas.

    hv_reference: fixed (M,) worst-case corner for the hypervolume indicator.
        None disables hypervolume tracking (and therefore plateau stopping).
        It must be a constant of the problem, never derived from the current
        population -- see tuning/hypervolume.py.

    stall_patience / stall_tol / stall_min_generations: plateau stopping. None
        patience disables it and the run goes the full n_generations.

    eval_timeout: [s] None (default) preserves the original behaviour --
        `evaluate` is trusted to return in bounded time, as every existing
        Evaluator does (it catches its own exceptions and returns worst-case
        rather than raising). Set this when that trust has been violated:
        one candidate on one run took ~1800s against a ~3s baseline (no
        exception, no crash -- the simulation loop itself just ran very
        slowly, plausibly a subnormal-float slowdown from a near-degenerate
        gain vector) and blocked the ENTIRE generation behind it, since
        `ProcessPoolExecutor` results are collected in submission order --
        burning the whole wall-clock deadline on one stuck candidate instead
        of the population `pop_size - 1` candidates that finished in
        seconds. With eval_timeout set, a candidate that exceeds it is
        scored as worst-case (`evaluate.worst`, the same fallback the
        Evaluator itself uses for a non-finite result) instead of blocking
        -- so `evaluate` MUST expose a `.worst` attribute when this is used
        (checked at construction, not deferred to the first timeout).
        Requires n_workers > 1: there is no way to bound a same-process
        call without threads, and a hung candidate would then just hang the
        one process doing everything.
    """

    def __init__(self, n_genes, evaluate, decode, pop_size=40,
                 crossover_prob=0.9, mutation_prob=None,
                 eta_c=15.0, eta_m=20.0, seed=None, verbose_eval=True,
                 store_full_population=True, n_workers=1, hv_reference=None,
                 stall_patience=None, stall_tol=1e-4, stall_min_generations=40,
                 eval_timeout=None):
        self.n_genes = n_genes
        self.evaluate = evaluate
        self.decode = decode
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob or (1.0 / n_genes)
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.verbose_eval = verbose_eval
        self.n_workers = max(1, int(n_workers))
        self.hv_reference = (None if hv_reference is None
                             else np.asarray(hv_reference, dtype=float))
        self.eval_timeout = None if eval_timeout is None else float(eval_timeout)
        if self.eval_timeout is not None:
            if self.n_workers <= 1:
                raise ValueError(
                    "eval_timeout requires n_workers > 1 -- a hung candidate "
                    "in-process has no way to be bounded without blocking "
                    "the only worker there is.")
            if not hasattr(evaluate, "worst"):
                raise ValueError(
                    "eval_timeout is set but `evaluate` has no `.worst` "
                    "attribute to fall back to on a timeout -- "
                    "objectives.Evaluator provides one; a bare callable "
                    "(e.g. a test lambda) does not.")
        self.n_timeouts = 0
        self.stall = (None if stall_patience is None
                      else StallDetector(patience=stall_patience, tol=stall_tol,
                                         min_generations=stall_min_generations))
        # store_full_population: keep every individual's genes+objectives per
        # generation in history, not just the Pareto front. At typical sizes
        # here (pop<=100, genes<=9) this is a few hundred KB total even over
        # 300 generations, so it's on by default -- the alternative is losing
        # the ability to see anything but the front after the run ends, which
        # is exactly the kind of "wish I'd kept it" gap that's expensive to
        # discover after the fact (means rerunning the whole GA).
        self.store_full_population = store_full_population
        # Per-generation record, filled by run(). Generic: just the objective
        # values, no knowledge of what they mean -- viz/tuning.py labels them.
        self.history = []
        self.n_evaluations = 0
        self.stop_reason = None
        self._t_start = None
        self._pool = None
        # An evaluator may advertise extra per-individual quantities to record
        # alongside the objectives (settling time, control effort, ...). Read
        # once here rather than per generation so a plain callable costs
        # nothing and the contract is fixed for the whole run.
        self.diagnostic_labels = getattr(evaluate, "diagnostic_labels", None)

    # ---------------- bookkeeping ----------------

    def _record(self, generation, pop, obj, diag=None):
        front = fast_non_dominated_sort(obj)[0]
        hv = (float("nan") if self.hv_reference is None
              else hypervolume_fraction(obj[front], self.hv_reference))
        entry = {
            "generation": generation,
            "front_size": len(front),
            "best_per_objective": obj[front].min(axis=0).copy(),
            "median_per_objective": np.median(obj, axis=0).copy(),
            "front_objectives": obj[front].copy(),
            "front_genes": pop[front].copy(),
            "hypervolume": hv,
            "elapsed_s": (float("nan") if self._t_start is None
                          else time.perf_counter() - self._t_start),
            "n_evaluations": self.n_evaluations,
        }
        if diag is not None:
            # Engineering numbers (settling time, effort, ...) for every
            # individual, not just the three the search sorts on. nanmin/
            # nanmedian because a candidate that never settled carries NaN
            # there by design and must not poison the whole generation's
            # summary.
            entry["diagnostic_labels"] = tuple(self.diagnostic_labels)
            entry["front_diagnostics"] = diag[front].copy()
            # An all-NaN column is the NORMAL early-run state for settling
            # time: nothing has settled yet. numpy warns about the all-NaN
            # slice; here that warning is noise, and the NaN it returns is
            # exactly the right answer ("no individual settled").
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # Best over the whole POPULATION, not just the front. These
                # are reported quantities, not objectives, so "the best
                # settling time anything achieved this generation" is the
                # useful number -- and it must be consistent with n_settled
                # below, which is also population-wide. Taking the front-only
                # minimum produced the genuinely confusing state of
                # "settled = 1" next to "best settling time = did not settle",
                # which happens whenever the individual that settled is
                # dominated on the three search objectives.
                entry["best_per_diagnostic"] = np.nanmin(diag, axis=0)
                entry["front_best_per_diagnostic"] = np.nanmin(diag[front], axis=0)
                entry["median_per_diagnostic"] = np.nanmedian(diag, axis=0)
                # How many individuals actually settled. A rising count is the
                # single most legible sign the search is working, and it is
                # invisible in objective space because ITAE never ties.
                entry["n_settled"] = int(np.sum(np.isfinite(diag[:, 0])))
            if self.store_full_population:
                entry["population_diagnostics"] = diag.copy()
        if self.store_full_population:
            # genes, not just decoded gains: decode() needs controller_type
            # context that NSGA2 doesn't have, so raw [0,1]^N genes are what
            # gets stored -- run objectives.make_decode(ctype) on these later.
            entry["population_genes"] = pop.copy()
            entry["population_objectives"] = obj.copy()
        if self.stall is not None:
            self.stall.update(hv)
            entry["hv_relative_gain"] = self.stall.relative_gain
        self.history.append(entry)
        return entry

    # ---------------- evaluation ----------------

    def _evaluate_population(self, genes, label="eval"):
        """-> (objectives (N,M), diagnostics (N,D) or None).

        Diagnostics are returned only when `evaluate` advertises them via a
        `diagnostic_labels` attribute, so a plain callable (a lambda in a
        unit test, say) keeps the original objectives-only contract and needs
        no changes.
        """
        decoded = [self.decode(g) for g in genes]
        n = len(decoded)
        t0 = time.perf_counter()

        if self.n_workers > 1 and self._pool is not None and self.eval_timeout is None:
            results = []
            # chunksize 1: evaluations are seconds long and can vary by an
            # order of magnitude between a well-conditioned candidate and one
            # that limit-cycles, so handing out work one at a time keeps the
            # pool balanced. The dispatch overhead is microseconds against a
            # multi-second task.
            for i, r in enumerate(self._pool.map(self.evaluate, decoded, chunksize=1)):
                results.append(r)
                self.n_evaluations += 1
                if self.verbose_eval:
                    self._progress(label, i, n, t0)

        elif self.n_workers > 1 and self._pool is not None:
            # Same contract as the branch above, but bounded: `pool.map`
            # yields results in submission order, so one candidate stuck
            # past eval_timeout would otherwise block every result after it
            # too, not just its own. Submitting individually and collecting
            # with a per-future timeout stops the wait -- not the stuck
            # worker process itself, ProcessPoolExecutor has no clean way to
            # kill a task that is already running -- which is exactly why
            # the pool gets rebuilt below once the generation is done, to
            # hand back a full-strength pool for the next one.
            futures = [self._pool.submit(self.evaluate, d) for d in decoded]
            results = [None] * n
            timed_out = False
            worst_diag = (None if self.diagnostic_labels is None
                          else np.full(len(self.diagnostic_labels), np.nan))
            fallback = (self.evaluate.worst if self.diagnostic_labels is None
                        else (self.evaluate.worst, worst_diag))
            for i, fut in enumerate(futures):
                try:
                    results[i] = fut.result(timeout=self.eval_timeout)
                except FutureTimeoutError:
                    results[i] = fallback
                    self.n_timeouts += 1
                    timed_out = True
                    print(f"\n  WARNING: candidate {i+1}/{n} exceeded "
                          f"eval_timeout={self.eval_timeout:.0f}s -- scored "
                          f"worst-case, not blocking the rest of the "
                          f"generation ({self.n_timeouts} timeout(s) so far)",
                          flush=True)
                self.n_evaluations += 1
                if self.verbose_eval:
                    self._progress(label, i, n, t0)

            if timed_out:
                # The worker that was still running its stuck task when we
                # stopped waiting is not reclaimed by anything short of a
                # fresh pool -- shutdown() lets current tasks finish (or, in
                # practice, keep running) rather than killing them, but does
                # stop new work going to them, so tearing down and rebuilding
                # is what actually restores full parallelism for gen+1.
                from concurrent.futures import ProcessPoolExecutor
                self._pool.shutdown(wait=False, cancel_futures=True)
                self._pool = ProcessPoolExecutor(max_workers=self.n_workers)
                if self.verbose_eval:
                    print("  rebuilt the worker pool after a timeout", flush=True)
        else:
            results = []
            for i, d in enumerate(decoded):
                results.append(self.evaluate(d))
                self.n_evaluations += 1
                if self.verbose_eval:
                    self._progress(label, i, n, t0)

        if self.verbose_eval:
            print(flush=True)

        if self.diagnostic_labels is None:
            return np.array(results, dtype=float), None
        obj = np.array([r[0] for r in results], dtype=float)
        diag = np.array([r[1] for r in results], dtype=float)
        return obj, diag

    def _progress(self, label, i, n, t0):
        elapsed = time.perf_counter() - t0
        avg = elapsed / (i + 1)
        print(f"\r  {label}: {i+1}/{n}  elapsed={elapsed:6.1f}s  "
              f"eta={avg * (n - i - 1):6.1f}s", end="", flush=True)

    # ---------------- genetic operators ----------------

    def _tournament(self, rank, crowd):
        i, j = self.rng.integers(0, len(rank), size=2)
        if rank[i] < rank[j]:
            return i
        if rank[j] < rank[i]:
            return j
        return i if crowd[i] > crowd[j] else j

    def _sbx(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() > self.crossover_prob:
            return c1, c2
        for k in range(self.n_genes):
            if self.rng.random() > 0.5 or abs(p1[k] - p2[k]) < 1e-14:
                continue
            u = self.rng.random()
            beta = (2 * u) ** (1 / (self.eta_c + 1)) if u <= 0.5 else \
                   (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
            c1[k] = 0.5 * ((1 + beta) * p1[k] + (1 - beta) * p2[k])
            c2[k] = 0.5 * ((1 - beta) * p1[k] + (1 + beta) * p2[k])
        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

    def _mutate(self, ind):
        ind = ind.copy()
        for k in range(self.n_genes):
            if self.rng.random() > self.mutation_prob:
                continue
            u = self.rng.random()
            delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1 if u < 0.5 else \
                    1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
            ind[k] = np.clip(ind[k] + delta, 0, 1)
        return ind

    # ---------------- persistence ----------------

    @staticmethod
    def _replace_with_retry(tmp, path, attempts=12, delay=0.25):
        """os.replace(), retried -- because on Windows it is not safe against
        a concurrent READER.

        POSIX lets you rename over a file another process has open. Windows
        does not: if the monitor or the logbook happens to have the
        checkpoint open at that instant, os.replace raises
        PermissionError WinError 5 and, unguarded, that propagates up and
        kills the whole seed. This actually happened -- an 8-hour run lost
        its third seed to a millisecond-wide race with its own read-only
        dashboard, which is an absurd way to lose a night of compute.

        The window is tiny (the reader holds the file only as long as
        pickle.load takes), so a short retry loop closes it completely. If
        every attempt fails we warn and carry on rather than raise: a missed
        checkpoint costs one generation of crash-insurance, whereas killing
        the run costs the entire seed. The two are not remotely equivalent,
        so this must never be fatal.
        """
        for i in range(attempts):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                if i == attempts - 1:
                    break
                time.sleep(delay)
        warnings.warn(
            f"could not update {path} after {attempts} attempts (file locked "
            f"by another process?). The run continues; this generation's "
            f"checkpoint was skipped.",
            RuntimeWarning, stacklevel=2)
        return False

    def _checkpoint(self, path, pop, obj, diag=None):
        """Overwrite path with the current population/objectives/history.
        Called after every generation so a killed kernel loses at most one
        generation's work instead of the whole run. Cheap relative to a
        single generation's simulate() calls (milliseconds vs. minutes), so
        this is always safe to leave on."""
        if path is None:
            return
        tmp = str(path) + ".tmp"
        # Recreate the parent dir if it is gone. Not the expected case --
        # run_seed() creates it once up front -- but two overnight_tune.py
        # invocations sharing an --out directory is a real failure mode: the
        # second one's startup archiving step (which renames the whole
        # directory before writing) can sweep a still-running first
        # instance's seed folder out from under it mid-checkpoint. Losing an
        # entire seed to that is worse than the cost of a mkdir every
        # generation, and a checkpoint written into a directory that
        # reappeared moments ago is still a checkpoint worth having.
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            pickle.dump({"pop": pop, "obj": obj, "diag": diag,
                         "history": self.history,
                         "diagnostic_labels": self.diagnostic_labels,
                         "hv_reference": self.hv_reference,
                         "seed": self.seed, "stop_reason": self.stop_reason}, f)
        # atomic-ish: never leaves a half-written file at `path`
        self._replace_with_retry(tmp, path)

    def _status(self, path, generation, n_generations, deadline):
        """A few hundred bytes of JSON, rewritten every generation.

        Separate from the checkpoint pickle on purpose: this is the file you
        want to `cat` from another terminal, poll from a phone, or read from a
        script, without unpickling megabytes of population history or needing
        deimos importable to do it.
        """
        if path is None:
            return
        rec = self.history[-1]
        elapsed = rec["elapsed_s"]
        per_gen = elapsed / max(generation, 1)
        remaining_gens = max(n_generations - generation, 0)
        eta = per_gen * remaining_gens
        if deadline is not None:
            eta = min(eta, max(deadline - time.time(), 0.0))
        payload = {
            "generation": int(generation),
            "n_generations": int(n_generations),
            "front_size": int(rec["front_size"]),
            "hypervolume": float(rec["hypervolume"]),
            "hv_relative_gain": float(rec.get("hv_relative_gain", float("nan"))),
            "best_per_objective": [float(v) for v in rec["best_per_objective"]],
            "n_evaluations": int(self.n_evaluations),
            "n_settled": int(rec.get("n_settled", -1)),
            "best_per_diagnostic": (
                None if "best_per_diagnostic" not in rec
                else {lab: float(v) for lab, v
                      in zip(rec["diagnostic_labels"], rec["best_per_diagnostic"])}),
            "elapsed_s": float(elapsed),
            "seconds_per_generation": float(per_gen),
            "eta_s": float(eta),
            "deadline_in_s": (None if deadline is None
                              else float(max(deadline - time.time(), 0.0))),
            "stop_reason": self.stop_reason,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = str(path) + ".tmp"
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)  # see _checkpoint
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        self._replace_with_retry(tmp, path)

    # ---------------- main loop ----------------

    def run(self, n_generations, verbose=True, seed_individuals=None,
            checkpoint_path=None, status_path=None, deadline=None):
        """
        seed_individuals: (k, n_genes) array of known-good genomes to place in
            the initial population instead of k random ones (a "warm start").
            NSGA-II is elitist -- parents compete with offspring every
            generation -- so a seeded individual can only be displaced by
            something that dominates it. Seeding the design you already have
            therefore makes "the search returned something worse than what I
            started with" structurally impossible.

        checkpoint_path: (pop, obj, history) pickled here after every
            generation, overwriting the previous checkpoint each time. Load it
            with `pickle.load(open(path, "rb"))`.

        status_path: small JSON summary written every generation.

        deadline: absolute time.time() past which the run stops at the next
            generation boundary. Checked BEFORE starting a generation and
            again after, using the measured time of the last generation, so
            the run does not start a generation it cannot finish in budget.
        """
        self.history = []
        self.n_evaluations = 0
        self.stop_reason = None
        self._t_start = time.perf_counter()
        self._pool = None

        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        if status_path is not None:
            Path(status_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.n_workers > 1:
                from concurrent.futures import ProcessPoolExecutor
                # One pool for the whole run, not one per generation: process
                # startup on Windows means a fresh interpreter plus a numpy
                # import per worker, which is seconds -- paying that 300 times
                # would be a real fraction of the budget.
                self._pool = ProcessPoolExecutor(max_workers=self.n_workers)
                if verbose:
                    print(f"parallel evaluation: {self.n_workers} worker processes",
                          flush=True)

            pop = self.rng.random((self.pop_size, self.n_genes))

            if seed_individuals is not None:
                seeds = np.clip(np.atleast_2d(np.asarray(seed_individuals, dtype=float)),
                                0.0, 1.0)
                if seeds.shape[1] != self.n_genes:
                    raise ValueError(
                        f"seed_individuals must have {self.n_genes} genes, "
                        f"got {seeds.shape[1]}")
                k = min(len(seeds), self.pop_size)
                pop[:k] = seeds[:k]

            obj, diag = self._evaluate_population(pop, label="init pop")
            # generation 0 = the random initial population, so a convergence
            # plot shows what the search actually bought over random sampling.
            self._record(0, pop, obj, diag)
            self._checkpoint(checkpoint_path, pop, obj, diag)
            self._status(status_path, 0, n_generations, deadline)

            for gen in range(n_generations):
                if deadline is not None and self._out_of_time(deadline):
                    self.stop_reason = (
                        f"wall-clock deadline reached after generation {gen} "
                        f"({self.history[-1]['elapsed_s']:.0f} s elapsed)")
                    break

                fronts = fast_non_dominated_sort(obj)
                rank = np.empty(len(pop), dtype=int)
                crowd = np.empty(len(pop))
                for r, front in enumerate(fronts):
                    rank[front] = r
                    crowd[front] = crowding_distance(obj[front])

                offspring = []
                while len(offspring) < self.pop_size:
                    i, j = self._tournament(rank, crowd), self._tournament(rank, crowd)
                    c1, c2 = self._sbx(pop[i], pop[j])
                    offspring.append(self._mutate(c1))
                    offspring.append(self._mutate(c2))
                offspring = np.array(offspring[:self.pop_size])
                offspring_obj, offspring_diag = self._evaluate_population(
                    offspring, label=f"gen {gen+1}/{n_generations}")

                combined = np.vstack([pop, offspring])
                combined_obj = np.vstack([obj, offspring_obj])
                combined_diag = (None if diag is None
                                 else np.vstack([diag, offspring_diag]))
                fronts = fast_non_dominated_sort(combined_obj)

                next_idx = []
                for front in fronts:
                    if len(next_idx) + len(front) <= self.pop_size:
                        next_idx.extend(front)
                    else:
                        cd = crowding_distance(combined_obj[front])
                        order = np.argsort(-cd)
                        remaining = self.pop_size - len(next_idx)
                        next_idx.extend([front[k] for k in order[:remaining]])
                        break

                pop, obj = combined[next_idx], combined_obj[next_idx]
                diag = None if combined_diag is None else combined_diag[next_idx]
                rec = self._record(gen + 1, pop, obj, diag)
                self._checkpoint(checkpoint_path, pop, obj, diag)
                self._status(status_path, gen + 1, n_generations, deadline)

                if verbose:
                    hv_txt = ("" if not np.isfinite(rec["hypervolume"])
                              else f"  HV={rec['hypervolume']:.6f}")
                    # Settling time is the number a human reads first, so it
                    # goes on the console line even though it steers nothing.
                    settle_txt = ""
                    if "best_per_diagnostic" in rec:
                        bs = rec["best_per_diagnostic"][0]
                        settle_txt = (
                            f"  settled={rec['n_settled']:2d}/{len(pop)}"
                            + ("" if not np.isfinite(bs) else f" best={bs:.1f}s"))
                    print(f"gen {gen+1:3d}/{n_generations}  "
                          f"front size={rec['front_size']:3d}{hv_txt}{settle_txt}  "
                          f"best per-obj={np.round(rec['best_per_objective'], 4)}",
                          flush=True)

                if self.stall is not None and self.stall.stalled():
                    self.stop_reason = self.stall.reason()
                    if verbose:
                        print(f"stopping early -- {self.stop_reason}", flush=True)
                    break
            else:
                self.stop_reason = f"completed all {n_generations} generations"

            if self.stop_reason is None:
                self.stop_reason = f"stopped after generation {len(self.history) - 1}"

            self._checkpoint(checkpoint_path, pop, obj, diag)
            self._status(status_path, len(self.history) - 1, n_generations, deadline)

        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None

        fronts = fast_non_dominated_sort(obj)
        pareto = fronts[0]
        # Kept as an attribute rather than widening the return tuple, so every
        # existing `pop, obj = ga.run(...)` caller is untouched.
        self.final_diagnostics = None if diag is None else diag[pareto]
        return pop[pareto], obj[pareto]

    def _out_of_time(self, deadline):
        """True if there is not enough budget left to finish another
        generation. Uses the measured duration of the last generation rather
        than just `now > deadline`, so the run stops instead of starting a
        generation it will have to abandon."""
        now = time.time()
        if now >= deadline:
            return True
        if len(self.history) < 2:
            return False
        last = self.history[-1]["elapsed_s"] - self.history[-2]["elapsed_s"]
        return (now + last) > deadline
