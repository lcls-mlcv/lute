#!/usr/bin/env python
"""Detector distance and beam centre from a calibrant powder run (LCLS-II, psana2).

An independent cross-check on BayFAI: one script goes from xtc2 to a distance,
figures and a summary.

    source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
    export PYTHONPATH=/sdf/group/lcls/ds/tools/LCLSGeom/src:$PYTHONPATH   # for the .poni
    mpirun -n 12 python powder_distance.py -e <experiment> -r <run> --expected <lo_mm> <hi_mm>

Steps
  1. powder: mean and max of <det>.raw.calib over N events (MPI), plus pixel
     coordinates, mask and photon energy, saved to <out>/<exp>_r<run>_powder.h5.
     Skip with --powder FILE. This also accepts a LUTE/SmallData file that has
     Sums/<det>_calib(_max) and UserDataCfg/<det>/{x,y,ix,iy,mask}.
  2. fit: per-panel high-pass; the brightest pixels are taken as ring candidates.
     Fit D (sample -> detector plane), beam centre (cx, cy) and two small tilts
     by maximizing agreement of each pixel's 2θ with the calibrant cones.
     Coarse D x centre grid search, then Nelder-Mead at decreasing ring widths.
  3. check: refit on two disjoint halves of the pixels (repeatability), score vs
     D (uniqueness), per-ring residuals.
  4. report: figures, SUMMARY.md, geometry.json and r<run>_<det>.poni in --out.
     The poni is in the LCLSGeom template frame (what smalldata azav_lclsgeom
     and BayFAI use), so it can go straight into getAzIntLCLSGeomParams.poni_file.

Only rank 0 fits. After the powder step the other ranks wait idle and rank 0
takes over their cores for the (numba, multithreaded) fit.
"""
import argparse, json, os, sys, time
import numpy as np

HC = 12398.42  # eV·Å
T_START = time.time()


def nthreads():
    """Threads for the fit (PD_THREADS overrides): the cores this process may use, at most 32."""
    return int(os.environ.get("PD_THREADS", min(32, len(os.sched_getaffinity(0)))))


def claim_job_cores(comm):
    """After the powder step only rank 0 works, so give it every core the job's ranks were pinned to.

    Under Slurm/mpirun each rank is bound to one core, which would leave rank 0's
    threads on a single core. Collective: every rank must call it.
    """
    cores = comm.gather(os.sched_getaffinity(0), root=0)
    if comm.Get_rank() == 0:
        try:
            os.sched_setaffinity(0, set().union(*cores))
        except OSError:
            pass
        _IDLE.append(comm)


_IDLE = []  # set on rank 0 once the other ranks are parked in idle_until_done


def release_idle_ranks():
    if _IDLE:
        _IDLE.pop().Ibarrier().Wait()


def idle_until_done(comm):
    """Non-root ranks: wait for rank 0 without spinning (MPI_Finalize can busy-poll the cores rank 0 now uses)."""
    req = comm.Ibarrier()
    while not req.Test():
        time.sleep(0.2)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--exp", required=True, help="experiment, e.g. mfx100000000")
    ap.add_argument("-r", "--run", type=int, required=True)
    ap.add_argument("-d", "--det", default="jungfrau", help="area detector name (default jungfrau)")
    ap.add_argument("-c", "--calibrant", default="LaB6", help="pyFAI calibrant name (LaB6, CeO2, Si, AgBh, ...)")
    ap.add_argument("-n", "--events", type=int, default=1500, help="events for the powder (default 1500)")
    ap.add_argument("-o", "--out", help="output dir (default ./<exp>_r<run>_<det>)")
    ap.add_argument("--powder", help="reuse an existing powder/SmallData h5 instead of reading xtc2")
    ap.add_argument("--ebeam", default="ebeamh", help="psana detector with ebeamPhotonEnergy (default ebeamh)")
    ap.add_argument("--wavelength-A", type=float, help="override wavelength (Å)")
    ap.add_argument("--energy-eV", type=float, help="override photon energy (eV)")
    ap.add_argument("--dist-range", type=float, nargs=2, default=[40, 600], metavar=("MIN", "MAX"),
                    help="distance search range in mm (default 40 600)")
    ap.add_argument("--expected", type=float, nargs=2, metavar=("LO", "HI"),
                    help="expected distance band in mm (shaded in plots, flagged in summary)")
    ap.add_argument("--centre-search", type=float, default=6.0,
                    help="± beam-centre search about the psana origin, mm (default 6)")
    ap.add_argument("--beamstop", type=float, default=25.0,
                    help="ignore pixels closer than this to the psana origin, mm (default 25)")
    ap.add_argument("--max-rings", type=int, default=15, help="calibrant rings used (default 15)")
    ap.add_argument("--ring-frac", type=float, default=2.5,
                    help="top %% of high-passed pixels treated as ring candidates (default 2.5)")
    ap.add_argument("--no-tilt", action="store_true", help="fit D and centre only")
    ap.add_argument("--no-figures", action="store_true", help="skip the figures (speed tests only)")
    ap.add_argument("--datasource-kw", default="",
                    help="extra DataSource kwargs, e.g. 'dir=/path/xtc' (comma-separated k=v)")
    ap.add_argument("--elog-summary", nargs="?", const="Detector_Distance", metavar="NAME",
                    help="publish report.html to the eLog Summaries tab under NAME "
                         "(default Detector_Distance) and post key numbers to the run's workflow status")
    return ap.parse_args()


# --------------------------------------------------------------------------- step 1
def make_powder(a, path):
    from mpi4py import MPI
    from psana import DataSource
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    T = [("start", T_START), ("imports", time.time())]
    kw = dict(kv.split("=", 1) for kv in a.datasource_kw.split(",") if kv)
    dets = [a.det] + ([a.ebeam] if a.ebeam else [])
    ds = DataSource(exp=a.exp, run=a.run, max_events=a.events, detectors=dets, **kw)
    run = next(ds.runs())
    T.append(("run start (calib constants)", time.time()))
    det = run.Detector(a.det)
    try:
        eb = run.Detector(a.ebeam) if a.ebeam else None
    except Exception:
        eb = None

    pmax = psum = None
    en, n, t0 = [], 0, time.time()
    for evt in run.events():
        c = det.raw.calib(evt)
        if c is None:
            continue
        if pmax is None:
            pmax, psum = c.astype(np.float32), c.astype(np.float32)  # float32 tested: same fit, see NOTES.md
        else:
            np.maximum(pmax, c, out=pmax)
            psum += c
        if eb is not None:
            e = eb.raw.ebeamPhotonEnergy(evt)
            if e is not None:
                en.append(float(e))
        n += 1
        if n % 200 == 0:
            print(f"  rank {rank}: {n} events, {time.time()-t0:.0f} s", flush=True)

    shapes = comm.allgather(None if pmax is None else pmax.shape)
    T.append(("event loop", time.time()))
    shape = next((s for s in shapes if s is not None), None)
    if shape is None:
        if rank == 0:
            sys.exit(f"No {a.det} calib data in run {a.run}")
        sys.exit(0)
    lmax = pmax if pmax is not None else np.full(shape, -np.inf, np.float32)
    lsum = psum if psum is not None else np.zeros(shape, np.float32)
    gmax = np.empty(shape, np.float32) if rank == 0 else None
    gsum = np.empty(shape, np.float32) if rank == 0 else None
    comm.Reduce(lmax, gmax, op=MPI.MAX, root=0)
    comm.Reduce(lsum, gsum, op=MPI.SUM, root=0)
    ntot = comm.reduce(n, root=0)
    ens = comm.gather(en, root=0)
    T.append(("reduce", time.time()))
    if rank == 0:  # copies: these can be views into psana shared memory, freed below
        ix, iy = (np.array(v) for v in det.raw._pixel_coord_indexes())
        x, y, z = (np.array(v) for v in det.raw._pixel_coords())
        mask = det.raw._mask()
        mask = None if mask is None else np.array(mask)
    # psana frees its MPI shared-memory windows (a collective call) when the DataSource is
    # garbage-collected, so all ranks drop it here, together, before the others go idle.
    import gc
    del det, eb, run, ds
    gc.collect()
    comm.Barrier()
    claim_job_cores(comm)
    if rank != 0:
        idle_until_done(comm)
        return False

    import h5py
    ens = np.concatenate([np.asarray(e) for e in ens if len(e)]) if any(len(e) for e in ens) else np.array([])
    with h5py.File(path, "w") as f:
        f[f"Sums/{a.det}_calib"] = gsum / max(ntot, 1)
        f[f"Sums/{a.det}_calib_max"] = gmax
        g = f.create_group(f"UserDataCfg/{a.det}")
        g["ix"], g["iy"], g["x"], g["y"], g["z"] = ix, iy, x, y, z
        g["mask"] = (mask if mask is not None else np.ones(shape)).astype(np.uint8)
        if len(ens):
            f["ebeamh/ebeamPhotonEnergy"] = ens
        f.attrs.update(exp=a.exp, run=a.run, det=a.det, events=ntot)
    T.append(("write h5", time.time()))
    print(f"[powder] {ntot} events -> {path}", flush=True)
    return [f"[time] powder {name:<28s} {t - T[i][1]:6.1f} s" for i, (name, t) in enumerate(T[1:])]


def load_powder(path, det):
    import h5py
    with h5py.File(path, "r") as f:
        key = next((k for k in (f"Sums/{det}_calib", f"Sums/{det}_calib_max") if k in f), None)
        if key is None:
            sys.exit(f"{path}: no Sums/{det}_calib or _calib_max")
        img = f[key][()].astype(float)
        g = f[f"UserDataCfg/{det}"]
        x, y = g["x"][()] * 1e-6, g["y"][()] * 1e-6  # µm -> m
        ix, iy = g["ix"][()].astype(int), g["iy"][()].astype(int)
        mask = g["mask"][()].astype(bool) if "mask" in g else np.ones(img.shape, bool)
        energy = None
        for k in ("ebeamh/ebeamPhotonEnergy", "ebeam/ebeamPhotonEnergy"):
            if k in f:
                e = f[k][()]
                e = e[np.isfinite(e) & (e > 100)]
                if e.size:
                    energy = float(np.median(e))
                break
        nev = int(f.attrs.get("events", 0)) or None
    return dict(img=img, x=x, y=y, ix=ix, iy=iy, mask=mask & np.isfinite(img) & (img != 0),
                energy=energy, key=key, events=nev)


# --------------------------------------------------------------------------- step 2
def ring_dspacings(name, nmax):
    from pyFAI.calibrant import get_calibrant
    d = np.array(sorted(get_calibrant(name).get_dSpacing(), reverse=True))
    keep = [d[0]]
    for v in d[1:]:
        if abs(v - keep[-1]) / keep[-1] > 2e-4:  # merge coincident reflections
            keep.append(v)
    return np.array(keep[:nmax])


def ring_candidates(P, ring_frac, beamstop):
    from scipy.ndimage import median_filter
    img, m = P["img"], P["mask"]
    from concurrent.futures import ThreadPoolExecutor
    hp = np.zeros_like(img)

    def panel(p):  # scipy releases the GIL, so panels run in parallel threads
        if m[p].any():
            a = np.where(m[p], img[p], np.median(img[p][m[p]]))
            hp[p] = a - median_filter(a, 15)
    with ThreadPoolExecutor(min(img.shape[0], nthreads())) as ex:
        list(ex.map(panel, range(img.shape[0])))
    ok = m & (np.hypot(P["x"], P["y"]) > beamstop * 1e-3)
    v = hp[ok]
    sel = v > np.percentile(v, 100 - ring_frac)
    X, Y = P["x"][ok][sel], P["y"][ok][sel]
    W = np.clip(v[sel], 0, np.percentile(v[sel], 99))  # cap single-crystal spots
    return hp, X, Y, W / W.mean()


def two_theta(q, X, Y):
    D, cx, cy, tx, ty = q
    px, py = X - cx, Y - cy
    return np.arctan2(np.hypot(px * np.cos(tx), py * np.cos(ty)), D + px * np.sin(tx) + py * np.sin(ty))


def nearest(t, tth):
    """Distance from each 2θ to the nearest calibrant cone (tth sorted ascending)."""
    i = np.clip(np.searchsorted(tth, t), 1, len(tth) - 1)
    return np.minimum(np.abs(t - tth[i - 1]), np.abs(t - tth[i]))


def _score_np(q, X, Y, W, tth, sigma):
    dt = nearest(two_theta(q, X, Y), tth)
    return np.sum(W * np.exp(-0.5 * (dt / sigma) ** 2)) / len(W)


score = _score_np
score_many = lambda Q, X, Y, W, tth, sigma: np.array([_score_np(q, X, Y, W, tth, sigma) for q in Q])


def build_score():
    """Swap in the fused numba kernel (same arithmetic as _score_np, multithreaded).

    Called after claim_job_cores: numba sizes its thread pool from the CPU affinity at import.
    """
    global score, score_many
    try:
        import numba
    except ImportError:
        return
    numba.set_num_threads(min(nthreads(), numba.config.NUMBA_NUM_THREADS))

    @numba.njit(cache=True)
    def _nearest1(t, tth):
        lo, hi = 0, tth.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if tth[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        i = min(max(lo, 1), tth.shape[0] - 1)
        return min(abs(t - tth[i - 1]), abs(t - tth[i]))

    @numba.njit(parallel=True, cache=True)
    def _score_nb(q, X, Y, W, tth, sigma):
        D, cx, cy, tx, ty = q[0], q[1], q[2], q[3], q[4]
        ctx, cty, stx, sty = np.cos(tx), np.cos(ty), np.sin(tx), np.sin(ty)
        g = np.empty(X.shape[0])
        for k in numba.prange(X.shape[0]):
            px, py = X[k] - cx, Y[k] - cy
            d = _nearest1(np.arctan2(np.hypot(px * ctx, py * cty), D + px * stx + py * sty), tth)
            g[k] = W[k] * np.exp(-0.5 * (d / sigma) ** 2)
        return g.sum() / X.shape[0]

    @numba.njit(parallel=True, cache=True)
    def _score_many_nb(Q, X, Y, W, tth, sigma):  # one geometry per thread, for grid scans
        out = np.empty(Q.shape[0])
        for j in numba.prange(Q.shape[0]):
            D, cx, cy, tx, ty = Q[j, 0], Q[j, 1], Q[j, 2], Q[j, 3], Q[j, 4]
            ctx, cty, stx, sty = np.cos(tx), np.cos(ty), np.sin(tx), np.sin(ty)
            acc = 0.0
            for k in range(X.shape[0]):
                px, py = X[k] - cx, Y[k] - cy
                d = _nearest1(np.arctan2(np.hypot(px * ctx, py * cty), D + px * stx + py * sty), tth)
                acc += W[k] * np.exp(-0.5 * (d / sigma) ** 2)
            out[j] = acc / X.shape[0]
        return out

    score = lambda q, X, Y, W, tth, sigma: _score_nb(np.asarray(q, np.float64), X, Y, W, tth, float(sigma))
    score_many = lambda Q, X, Y, W, tth, sigma: _score_many_nb(np.asarray(Q, np.float64), X, Y, W, tth, float(sigma))


def fit_geometry(X, Y, W, tth, a, log, tick=lambda name: None):
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), min(len(X), 80000), replace=False)
    Xs, Ys, Ws = X[sub], Y[sub], W[sub]
    lo, hi = a.dist_range[0] * 1e-3, a.dist_range[1] * 1e-3
    c = a.centre_search * 1e-3

    # Coarse: D on a log grid, centre on a 1 mm grid, wide rings.
    Ds = np.exp(np.arange(np.log(lo), np.log(hi), 0.004))
    s0 = score_many(np.c_[Ds, np.zeros((len(Ds), 4))], Xs, Ys, Ws, tth, 2e-3)
    starts = []
    for i in np.argsort(s0)[::-1]:
        if all(abs(Ds[i] / s - 1) > 0.03 for s in starts):
            starts.append(Ds[i])
        if len(starts) == 6:
            break
    cands = []
    grid = np.arange(-c, c + 1e-9, 1e-3)
    for D in starts:  # same grid and order as before, so ties resolve the same way
        Q = np.array([[D * f, cx, cy, 0, 0] for f in (0.985, 1.0, 1.015) for cx in grid for cy in grid])
        sc = score_many(Q, Xs, Ys, Ws, tth, 1.5e-3)
        cands.append(max(zip(sc, Q[:, 0], Q[:, 1], Q[:, 2])))
    tick("fit coarse")
    log(f"[fit] coarse starts (D mm, cx, cy mm, score): " +
        "; ".join(f"{d*1e3:.1f} ({x*1e3:+.0f},{y*1e3:+.0f}) {s:.3f}" for s, d, x, y in sorted(cands, reverse=True)))

    scale = np.array([1e-3, 1e-4, 1e-4, 1e-3, 1e-3])
    free = slice(0, 3) if a.no_tilt else slice(0, 5)

    def refine(q, Xr, Yr, Wr):
        for sigma in (1.5e-3, 6e-4, 3e-4):
            def f(u):
                qq = q.copy(); qq[free] = u * scale[free]
                return -score(qq, Xr, Yr, Wr, tth, sigma)
            r = minimize(f, q[free] / scale[free], method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-3, fatol=1e-9))
            q = q.copy(); q[free] = r.x * scale[free]
        return q

    fits = []
    for s, D, cx, cy in sorted(cands, reverse=True)[:4]:
        q = refine(np.array([D, cx, cy, 0.0, 0.0]), Xs, Ys, Ws)
        fits.append((score(q, Xs, Ys, Ws, tth, 3e-4), q))
        log(f"[fit] start {D*1e3:6.1f} mm -> D={q[0]*1e3:7.2f} mm  centre ({q[1]*1e3:+.3f}, {q[2]*1e3:+.3f}) mm  "
            f"tilt ({np.degrees(q[3]):+.3f}, {np.degrees(q[4]):+.3f})°  score {fits[-1][0]:.4f}")
    tick("fit starts")
    fits.sort(key=lambda t: -t[0])
    q0 = fits[0][1]
    runner_up = fits[1][0] / fits[0][0] if len(fits) > 1 and fits[0][0] > 0 else 0.0

    # Final refit on all candidate pixels, then on two disjoint halves.
    big = rng.choice(len(X), min(len(X), 400000), replace=False)
    q = refine(q0, X[big], Y[big], W[big])
    tick("fit final")
    perm = rng.permutation(len(X))
    halves = [perm[: len(X) // 2][:300000], perm[len(X) // 2:][:300000]]
    qh = np.array([refine(q, X[h], Y[h], W[h]) for h in halves])
    tick("fit halves")
    return q, qh, runner_up, fits


# --------------------------------------------------------------------------- step 4
INK, MUTED, GRID, RING = "#1f2328", "#6e7781", "#d0d7de", "#d1242f"


def _style(ax):
    ax.grid(color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figures(P, hp, q, tth, dd, wl, a, tag, out, X, Y, W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    files = {}
    exp_band = a.expected

    # Ring overlay on the assembled image (circles; tilts are small-angle).
    img, m, ix, iy, x, y = P["img"], P["mask"], P["ix"], P["iy"], P["x"], P["y"]
    sh = (iy.max() + 1, ix.max() + 1)
    full = np.full(sh, np.nan); full[iy[m], ix[m]] = img[m]
    bs = 3  # the axes are ~1000 px wide at 150 dpi; matplotlib would resample to nearest anyway
    small = full[::bs, ::bs]
    hh, ww = small.shape[0] * bs, small.shape[1] * bs
    k = np.flatnonzero(m.ravel())[::997]
    sx = np.polyfit(x.ravel()[k], ix.ravel()[k], 1)
    sy = np.polyfit(y.ravel()[k], iy.ravel()[k], 1)
    c_ix, c_iy = np.polyval(sx, q[1]), np.polyval(sy, q[2])
    fig, ax = plt.subplots(figsize=(8.4, 8), dpi=150)
    vals = small[np.isfinite(small) & (small > 0)]
    lo, hi = np.percentile(vals, [3, 99.5])
    im = ax.imshow(np.where(small > 0, small, np.nan), cmap="Greys", norm=LogNorm(lo, hi),
                   origin="lower", interpolation="nearest", extent=(0, ww, 0, hh))
    t = np.linspace(0, 2 * np.pi, 1440)
    for i, tt in enumerate(tth):
        r = q[0] * np.tan(tt)
        if r * abs(sx[0]) > 1.5 * max(sh):
            break
        ax.plot(c_ix + r * abs(sx[0]) * np.cos(t), c_iy + r * abs(sy[0]) * np.sin(t), color=RING, lw=0.5, alpha=0.75)
        if i < 6:
            ang = np.radians(28 + 9 * i)
            ax.annotate(f"{dd[i]:.3f} Å", (c_ix + r * abs(sx[0]) * np.cos(ang), c_iy + r * abs(sy[0]) * np.sin(ang)),
                        color=RING, fontsize=7, xytext=(3, 3), textcoords="offset points",
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
    ax.plot(c_ix, c_iy, "+", color=RING, ms=10, mew=1.2)
    ax.set_xlim(0, sh[1]); ax.set_ylim(0, sh[0])
    ax.set_title(f"{tag} {a.calibrant}: predicted rings for D = {q[0]*1e3:.2f} mm\n"
                 f"beam centre ({q[1]*1e3:+.2f}, {q[2]*1e3:+.2f}) mm psana frame, λ = {wl:.4f} Å",
                 fontsize=9, color=INK)
    ax.set_xlabel("image x (pixel)", color=MUTED); ax.set_ylabel("image y (pixel)", color=MUTED)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label(P["key"].split("/")[-1] + " (keV)", color=MUTED)
    fig.tight_layout(); files["rings"] = os.path.join(out, f"{tag}_rings.png"); fig.savefig(files["rings"]); plt.close(fig)

    # 1D profile vs 2θ about the fitted geometry.
    tt_all = np.degrees(two_theta(q, x[m], y[m])); h_all = hp[m]
    b = np.arange(max(1.0, tt_all.min()), tt_all.max(), 0.02)
    ib = np.floor((tt_all - b[0]) / 0.02).astype(np.int64); ok = (ib >= 0) & (ib < len(b) - 1)
    s = np.bincount(ib[ok], h_all[ok], len(b) - 1); n = np.bincount(ib[ok], None, len(b) - 1)
    bc = 0.5 * (b[1:] + b[:-1])
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    ax.plot(bc, np.where(n > 200, s / np.maximum(n, 1), np.nan), color=INK, lw=0.9)
    for i, tdeg in enumerate(np.degrees(tth)):
        if tdeg > bc[-1]:
            break
        ax.axvline(tdeg, color=RING, lw=0.7, ls="--", alpha=0.7)
        ax.text(tdeg, 0.98, f"{dd[i]:.3f}", transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=7, color=RING)
    ax.set_xlabel("2θ (deg)", color=MUTED); ax.set_ylabel("background-subtracted intensity", color=MUTED)
    ax.set_title(f"{tag} radial profile at D = {q[0]*1e3:.2f} mm; dashed = {a.calibrant} (d in Å)", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout(); files["profile"] = os.path.join(out, f"{tag}_profile.png"); fig.savefig(files["profile"]); plt.close(fig)

    # Score vs D (other parameters fixed at the fit).
    rng = np.random.default_rng(2)
    sub = rng.choice(len(X), min(len(X), 150000), replace=False)
    lo_d, hi_d = q[0] * 0.8, q[0] * 1.25
    if exp_band:
        lo_d, hi_d = min(lo_d, exp_band[0] * 1e-3 * 0.95), max(hi_d, exp_band[1] * 1e-3 * 1.05)
    Ds = np.linspace(lo_d, hi_d, 500)
    sc = score_many(np.c_[Ds, np.tile(q[1:], (len(Ds), 1))], X[sub], Y[sub], W[sub], tth, 3e-4)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.plot(Ds * 1e3, sc, color=INK, lw=1.2)
    if exp_band:
        ax.axvspan(*exp_band, color=GRID, alpha=0.5, lw=0)
        ax.text(np.mean(exp_band), 0.97, f"expected {exp_band[0]:g}–{exp_band[1]:g} mm",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=MUTED)
    ax.axvline(q[0] * 1e3, color=RING, ls="--", lw=1)
    ax.text(q[0] * 1e3, 0.85, f"  {q[0]*1e3:.2f} mm", transform=ax.get_xaxis_transform(), color=RING, fontsize=8)
    ax.set_xlabel("distance D (mm), centre and tilts fixed at best fit", color=MUTED)
    ax.set_ylabel("ring agreement (higher = better)", color=MUTED)
    ax.set_title(f"{tag} {a.calibrant}: fit agreement vs detector distance", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout(); files["score"] = os.path.join(out, f"{tag}_score_vs_distance.png"); fig.savefig(files["score"]); plt.close(fig)
    return files


def poni_frame(P, det):
    """LCLSGeom detector and the psana -> pyFAI (poni1, poni2) affine maps.

    Independent of the fit, so main() runs it in a background thread while fitting.
    Returns (fd, A1, A2) or a message saying why no poni can be written.
    """
    try:
        from LCLSGeom.manager import get_geometry
        from LCLSGeom.converter import PsanaToPyFAI
    except ImportError as e:
        return f"skipped ({e}); add LCLSGeom to PYTHONPATH to write a poni"
    fd = PsanaToPyFAI.convert(get_geometry(det), det)
    c = fd.get_pixel_corners().mean(axis=2).reshape(-1, 3)  # z, poni1 axis, poni2 axis
    xy = np.c_[P["x"].reshape(-1), P["y"].reshape(-1), np.ones(c.shape[0])]
    if xy.shape[0] != c.shape[0]:
        return f"skipped: LCLSGeom detector has {c.shape[0]} pixels, psana has {xy.shape[0]}"
    k = np.flatnonzero(P["mask"].reshape(-1))[::97]
    A1, *_ = np.linalg.lstsq(xy[k], c[k, 1], rcond=None)
    A2, *_ = np.linalg.lstsq(xy[k], c[k, 2], rcond=None)
    rms = max(np.std(xy[k] @ A1 - c[k, 1]), np.std(xy[k] @ A2 - c[k, 2]))
    if rms > 1e-6:
        return f"skipped: psana -> pyFAI frame is not a rigid mapping (rms {rms*1e6:.1f} µm)"
    return fd, A1, A2


def write_poni(frame, q, wl, path, log):
    """pyFAI poni in the LCLSGeom template frame, as smalldata azav_lclsgeom expects.

    The psana -> pyFAI axis mapping is fitted from the pixel coordinates, so it does
    not assume a particular detector orientation. Tilts are dropped (rot = 0): the
    small-angle tilt model is not a pyFAI rotation, and the fitted tilts are ~0.05°.
    Needs LCLSGeom: export PYTHONPATH=/sdf/group/lcls/ds/tools/LCLSGeom/src:$PYTHONPATH
    """
    if isinstance(frame, str):
        log(f"[poni] {frame}")
        return None
    fd, A1, A2 = frame
    p1, p2 = np.r_[q[1], q[2], 1] @ A1, np.r_[q[1], q[2], 1] @ A2
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    ai = AzimuthalIntegrator(detector=fd, dist=q[0], poni1=p1, poni2=p2, rot1=0, rot2=0, rot3=0,
                             wavelength=wl * 1e-10)
    if os.path.exists(path):
        os.remove(path)  # pyFAI appends to an existing poni
    ai.save(path)
    log(f"[poni] {path}: dist {q[0]:.6f} m, poni1 {p1:+.6f} m, poni2 {p2:+.6f} m (LCLSGeom frame), λ {wl:.5f} Å")
    return path


def ring_table(q, X, Y, tth, dd):
    t = two_theta(q, X, Y)
    i = np.clip(np.searchsorted(tth, t), 1, len(tth) - 1)
    j = np.where(np.abs(t - tth[i - 1]) < np.abs(t - tth[i]), i - 1, i)
    dt = t - tth[j]
    rows = []
    for r in range(len(tth)):
        s = (j == r) & (np.abs(dt) < np.radians(0.05))
        if s.sum() > 200:
            rows.append((dd[r], np.degrees(tth[r]), int(s.sum()), np.degrees(np.mean(np.abs(dt[s])))))
    return rows


# --------------------------------------------------------------------------- main
def main():
    a = parse_args()
    tag = f"{a.exp}_r{a.run:04d}_{a.det}"
    out = os.path.abspath(a.out or tag)
    os.makedirs(out, exist_ok=True)

    powder, powder_times = a.powder, []
    if powder is None:
        powder = os.path.join(out, f"{tag}_powder.h5")
        powder_times = make_powder(a, powder)
        if not powder_times:
            return  # non-root ranks
    else:
        try:
            from mpi4py import MPI
            if MPI.COMM_WORLD.Get_size() > 1:
                claim_job_cores(MPI.COMM_WORLD)
                if MPI.COMM_WORLD.Get_rank() != 0:
                    idle_until_done(MPI.COMM_WORLD)
                    return
        except ImportError:
            pass
    build_score()

    logf = open(os.path.join(out, "fit.log"), "w")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    for s in powder_times:
        log(s)
    t0 = time.time()
    tlast = [t0]
    def tick(name):
        t = time.time(); log(f"[time] {name:<16s} {t - tlast[0]:6.1f} s"); tlast[0] = t
    P = load_powder(powder, a.det)
    from concurrent.futures import ThreadPoolExecutor
    poni_job = ThreadPoolExecutor(1).submit(poni_frame, P, a.det)  # overlaps with the fit
    tick("load")
    if a.wavelength_A:
        wl, wl_src = a.wavelength_A, "--wavelength-A"
    elif a.energy_eV:
        wl, wl_src = HC / a.energy_eV, "--energy-eV"
    elif P["energy"]:
        wl, wl_src = HC / P["energy"], f"median ebeamPhotonEnergy {P['energy']:.1f} eV"
    else:
        sys.exit("No photon energy in the powder file; pass --energy-eV or --wavelength-A")
    dd = ring_dspacings(a.calibrant, a.max_rings)
    dd = dd[wl / (2 * dd) < 1]
    tth = np.sort(2 * np.arcsin(wl / (2 * dd)))
    dd = wl / (2 * np.sin(tth / 2))
    log(f"[setup] {tag}  powder {powder} ({P['key']})")
    log(f"[setup] λ = {wl:.5f} Å ({wl_src}); {a.calibrant}: {len(dd)} rings, d = {dd[0]:.3f} … {dd[-1]:.3f} Å")

    hp, X, Y, W = ring_candidates(P, a.ring_frac, a.beamstop)
    tick("candidates")
    log(f"[setup] {len(X)} ring-candidate pixels (top {a.ring_frac}% of high-passed image, r > {a.beamstop} mm)")

    q, qh, runner_up, fits = fit_geometry(X, Y, W, tth, a, log, tick)
    spread = np.abs(qh[0] - qh[1])
    rows = ring_table(q, X, Y, tth, dd)
    log(f"[result] D = {q[0]*1e3:.3f} mm  centre ({q[1]*1e3:+.3f}, {q[2]*1e3:+.3f}) mm  "
        f"tilt ({np.degrees(q[3]):+.3f}, {np.degrees(q[4]):+.3f})°")
    log(f"[check] half-split ΔD = {spread[0]*1e3:.3f} mm, Δcentre = ({spread[1]*1e3:.3f}, {spread[2]*1e3:.3f}) mm; "
        f"runner-up/best score = {runner_up:.2f}")

    warnings = []
    lo, hi = a.dist_range
    if q[0] * 1e3 < lo * 1.02 or q[0] * 1e3 > hi * 0.98:
        warnings.append(f"D is at the edge of --dist-range {lo:g}–{hi:g} mm; widen it.")
    if max(abs(q[1]), abs(q[2])) * 1e3 > a.centre_search * 0.9:
        warnings.append("Beam centre is at the edge of --centre-search; increase it.")
    if runner_up > 0.6:
        warnings.append(f"Another distance scores {runner_up:.0%} of the best; the solution may be ambiguous (check the score plot).")
    if spread[0] > 0.5e-3:
        warnings.append(f"Half-split distances differ by {spread[0]*1e3:.2f} mm; rings may be weak or patchy.")
    if len(rows) < 3:
        warnings.append(f"Only {len(rows)} rings are well matched; the result is unreliable.")
    if a.expected and not (a.expected[0] <= q[0] * 1e3 <= a.expected[1]):
        warnings.append(f"D = {q[0]*1e3:.1f} mm is outside the expected {a.expected[0]:g}–{a.expected[1]:g} mm.")
    for w in warnings:
        log("[WARNING] " + w)

    files = {} if a.no_figures else figures(P, hp, q, tth, dd, wl, a, tag, out, X, Y, W)
    tick("figures")
    poni = write_poni(poni_job.result(), q, wl, os.path.join(out, f"r{a.run:04d}_{a.det}.poni"), log)
    tick("poni")
    if poni:
        files["poni"] = poni

    geo = dict(experiment=a.exp, run=a.run, detector=a.det, calibrant=a.calibrant,
               distance_m=q[0], beam_centre_psana_m=[q[1], q[2]], tilts_rad=[q[3], q[4]],
               wavelength_A=wl, wavelength_source=wl_src, half_split_dD_m=spread[0],
               runner_up_ratio=runner_up, rings=[dict(d_A=r[0], tth_deg=r[1], pixels=r[2], mean_abs_dtth_deg=r[3]) for r in rows],
               powder=powder, warnings=warnings, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out, "geometry.json"), "w") as f:
        json.dump(geo, f, indent=2, default=float)

    with open(os.path.join(out, "SUMMARY.md"), "w") as f:
        f.write(f"# {a.exp} run {a.run}: {a.det} distance from {a.calibrant} (powder_distance.py)\n\n")
        f.write(f"Created {geo['created']}. This is an independent check on BayFAI; see Caveats.\n\n")
        f.write("| Quantity | Value |\n|---|---|\n")
        f.write(f"| Sample → detector-plane distance | **{q[0]*1e3:.2f} mm** |\n")
        f.write(f"| Half-split repeatability (ΔD) | {spread[0]*1e3:.3f} mm |\n")
        f.write(f"| Beam centre (psana x, y) | ({q[1]*1e3:+.3f}, {q[2]*1e3:+.3f}) mm |\n")
        f.write(f"| Tilts | ({np.degrees(q[3]):+.3f}°, {np.degrees(q[4]):+.3f}°) |\n")
        f.write(f"| Wavelength | {wl:.5f} Å ({wl_src}) |\n")
        f.write(f"| Runner-up / best score | {runner_up:.2f} (lower = less ambiguous) |\n")
        if a.expected:
            f.write(f"| Expected band | {a.expected[0]:g}–{a.expected[1]:g} mm |\n")
        f.write(f"| Powder | `{powder}` |\n\n")
        f.write("## Rings matched\n\n| d (Å) | 2θ (°) | pixels within 0.05° | mean abs Δ2θ (°) |\n|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]:.4f} | {r[1]:.3f} | {r[2]} | {r[3]:.4f} |\n")
        f.write("\n## Warnings\n\n" + ("\n".join(f"- {w}" for w in warnings) if warnings else "None.") + "\n")
        f.write("\n## Figures\n\n" + "\n".join(f"- `{os.path.basename(p)}`" for p in files.values()) + "\n")
        f.write("\n## Caveats\n\n- The psana panel geometry is taken as exact (no per-panel refinement).\n"
                "- Small-angle tilt model; there is no polarisation correction.\n"
                "- The beam centre is in the psana lab frame, not pyFAI poni1/poni2. To compare with BayFAI, use the beam centre it reports.\n")
    if a.elog_summary:
        publish_elog(a, tag, q, spread, runner_up, wl, wl_src, rows, warnings, files, out, log)
    log(f"[done] {time.time()-t0:.0f} s -> {out}/SUMMARY.md, geometry.json, " + ", ".join(os.path.basename(p) for p in files.values()))


# --------------------------------------------------------------------------- eLog
def publish_elog(a, tag, q, spread, runner_up, wl, wl_src, rows, warnings, files, out, log):
    """eLog Summaries tab + run status.

    Summaries tab: the eLog shows /sdf/data/lcls/ds/<hutch>/<exp>/stats/summary/<NAME>/<sub>/report.html,
    the same path LUTE's executor writes. Figures are inlined as base64 so the page is
    self-contained. Run status: key/value pairs POSTed to $JID_UPDATE_COUNTERS, which the
    ARP sets for eLog-launched jobs (passed through to the Slurm job by --export=ALL).
    Neither needs Kerberos. Failures are logged, never fatal.
    """
    import base64, html, shutil
    D_mm, cx, cy = q[0] * 1e3, q[1] * 1e3, q[2] * 1e3
    status = "OK" if not warnings else f"{len(warnings)} warning(s)"
    try:
        root = f"/sdf/data/lcls/ds/{a.exp[:3]}/{a.exp}/stats/summary/{a.elog_summary}/r{a.run:04d}_{a.det}"
        os.makedirs(root, exist_ok=True)

        def img(key, title):
            p = files.get(key)
            if not p or not os.path.exists(p):
                return ""
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>'

        trow = lambda k, v: f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>"
        table = "".join([
            trow("Sample → detector distance", f"<b>{D_mm:.2f} mm</b>"),
            trow("Half-split repeatability ΔD", f"{spread[0]*1e3:.3f} mm"),
            trow("Beam centre (psana x, y)", f"({cx:+.3f}, {cy:+.3f}) mm"),
            trow("Tilts", f"({np.degrees(q[3]):+.3f}°, {np.degrees(q[4]):+.3f}°)"),
            trow("Wavelength", f"{wl:.5f} Å ({html.escape(wl_src)})"),
            trow("Calibrant", html.escape(a.calibrant)),
            trow("Runner-up / best score", f"{runner_up:.2f}"),
            trow("Expected band", f"{a.expected[0]:g}–{a.expected[1]:g} mm" if a.expected else "—"),
            trow("poni (LCLSGeom frame)", html.escape(files.get("poni") or "not written")),
            trow("Results directory", html.escape(out)),
        ])
        rings = "".join(f"<tr><td>{r[0]:.4f}</td><td>{r[1]:.3f}</td><td>{r[2]}</td><td>{r[3]:.4f}</td></tr>" for r in rows)
        warn = ("<ul class=warn>" + "".join(f"<li>{html.escape(w)}</li>" for w in warnings) + "</ul>") if warnings \
            else "<p class=ok>No warnings.</p>"
        page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(tag)} detector distance</title><style>
body{{font-family:system-ui,sans-serif;color:#1f2328;margin:1.5em;max-width:1100px}}
h1{{font-size:1.3em}} h2{{font-size:1.05em;margin-top:1.4em}}
table{{border-collapse:collapse;margin:.5em 0}} th,td{{border-bottom:1px solid #d0d7de;padding:.3em .8em;text-align:left;font-size:.92em}}
th{{color:#6e7781;font-weight:500}} .ok{{color:#1a7f37}} .warn li{{color:#d1242f}}
figure{{margin:1em 0}} img{{max-width:100%;border:1px solid #d0d7de}} figcaption{{color:#6e7781;font-size:.85em}}
</style></head><body>
<h1>{html.escape(a.exp)} run {a.run}: {html.escape(a.det)} distance from {html.escape(a.calibrant)} — {D_mm:.2f} mm ({status})</h1>
<p>powder_distance.py (independent of BayFAI), {time.strftime("%Y-%m-%d %H:%M")}.</p>
<table>{table}</table>
<h2>Warnings</h2>{warn}
{img("rings", "Powder image with predicted calibrant rings")}
{img("profile", "Radial profile vs 2θ; dashed = calibrant reflections")}
{img("score", "Fit agreement vs detector distance")}
<h2>Rings matched</h2><table><tr><th>d (Å)</th><th>2θ (°)</th><th>pixels within 0.05°</th><th>mean |Δ2θ| (°)</th></tr>{rings}</table>
</body></html>"""
        with open(os.path.join(root, "report.html"), "w") as f:
            f.write(page)
        if files.get("poni"):
            shutil.copy2(files["poni"], root)
        log(f"[elog] summary page: {root}/report.html (eLog → Summaries → {a.elog_summary})")
    except Exception as e:
        log(f"[elog] summary page FAILED: {e}")

    url = os.environ.get("JID_UPDATE_COUNTERS")
    if not url:
        log("[elog] run status skipped: JID_UPDATE_COUNTERS not set (not launched from the eLog)")
        return
    try:
        import requests
        kv = [("Detector distance (mm)", f"{D_mm:.2f}"), ("Beam centre x,y (mm)", f"{cx:+.2f}, {cy:+.2f}"),
              ("Wavelength (Å)", f"{wl:.4f}"), ("Distance fit", status)]
        r = requests.post(url, json=[{"key": k, "value": v} for k, v in kv], timeout=20)
        log(f"[elog] run status posted (HTTP {r.status_code})")
    except Exception as e:
        log(f"[elog] run status FAILED: {e}")


if __name__ == "__main__":
    try:
        main()
    finally:
        release_idle_ranks()
