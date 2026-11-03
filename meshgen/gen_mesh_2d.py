#!/usr/bin/env python3
"""
gen_mesh_2d.py -- graded Delaunay meshes for 2D Coulomb problems on rectangles.

Cleaned port of the generator that produced the paper's ``*_opt`` meshes
(Code_Sorted/cases/case7_2d_hydrogen_square/mesh_generator/
find_optimal_hmin_2d.py :: build_mesh, which calls the seed builders of
gen_mesh_linear_2d.py).  Requires numpy + scipy only.

Mesh construction (identical to the original):
  * Q = [-Lx, Lx] x [-Ly, Ly], nuclei a_i placed as nodes.
  * linear grading h(r) ~ theta * r, theta = h / max(Lx, Ly):
    concentric rings around every nucleus, first radius r_start, radial
    growth 1 + c_radial*theta, angular spacing c_angular*theta*r, ring j
    rotated by 0.37 j; rings extend to 1.05*sqrt(Lx^2+Ly^2) (points outside
    Q dropped).
  * boundary seeds with spacing c_face*h (4 % jitter, seed 7),
  * jittered Cartesian background with spacing c_bg*h (15 % jitter,
    seed 42), excluded within 20*r_start of a nucleus,
  * deduplication on a grid of size 0.1*c_angular*theta*r_start,
  * scipy Delaunay, triangles kept if the barycentre lies in Q.

r_start ("paper" rule, --r-start paper; this is what the *_opt meshes used):
  start from 0.4*h^2/L, halve until the analytic fan estimate c_0 of the
  patch term is <= c_*_ref (max over regular elements of an initial mesh of
  (h_K/2) * max_{7 Dunavant points}|V - V_K|), then clamp the radius used
  for the mesh to L*1e-7 (Qhull floor).  The reported c0/c*_ref/"dh_paper"
  are the original's heuristic numbers: they refer to the UNCLAMPED radius
  and to a sampled sup, so they are NOT a bound for d_h of the mesh written.
  An explicit float may be given instead (e.g. --r-start 5e-7).

Fixes relative to the original (disable with --paper-compat):
  1. Background points lying outside Q or closer than bg_margin*c_bg*h to dQ
     (default bg_margin = 0.1) are moved (clipped) to that distance; meshes
     without such points are bit-identical to the original.  In the original the
     last background row/column sits at L - 0.01 for spacing 0.18 (h = 0.2)
     and the +-15 % jitter pushes some points OUTSIDE Q (up to y = 5.0164 for
     box h = 0.2).  The Delaunay barycentre filter then keeps part of the
     triangles over those points: the mesh covers a region larger than Q
     and has single edges off dQ.
  2. All triangles are oriented counter-clockwise.
  3. info.txt / mesh_report.json record the actual Lx, Ly, centres (the
     original wrote the module defaults Lx = Ly = 5 for every mesh) and a
     conformity report (single edges off dQ, nodes outside Q, area defect).

Usage:
  python3 gen_mesh_2d.py --Lx 5 --Ly 5 --centers "0,0" --h 0.4 --out meshes/square_h040
  python3 gen_mesh_2d.py --Lx 7 --Ly 5 --centers="-2,0;2,0" --charges "1;1" --h 0.2 --out meshes/box_h020
"""

import argparse
import datetime
import json
import math
import os
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import Delaunay


# --------------------------------------------------------------------------
# seed points
# --------------------------------------------------------------------------

def _ring(r, n, centre, offset):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False) + offset
    return centre + r * np.column_stack([np.cos(a), np.sin(a)])


def ring_seeds(centre, theta, Lx, Ly, r_start, c_radial=0.85, c_angular=0.75):
    """Concentric rings around one nucleus (linear grading h(r) = theta*r)."""
    out = []
    R_max = 1.05 * math.sqrt(Lx ** 2 + Ly ** 2)
    growth = 1.0 + c_radial * theta
    r, j = r_start, 0
    centre = np.asarray(centre, float)
    while r < R_max:
        n = max(4, int(math.ceil(2 * math.pi * r / (c_angular * theta * r))))
        P = _ring(r, n, centre, j * 0.37)
        keep = (np.abs(P[:, 0]) <= Lx + 1e-12) & (np.abs(P[:, 1]) <= Ly + 1e-12)
        out.append(P[keep])
        r *= growth
        j += 1
    return np.vstack(out) if out else np.zeros((0, 2))


def boundary_seeds(Lx, Ly, spacing, jitter=0.04, seed=7):
    """Points on the four sides of Q (corners included, jitter along the side)."""
    rng = np.random.default_rng(seed)
    tol = 0.3 * spacing
    out = []
    for sy in (-1.0, 1.0):
        nx = max(3, int(math.ceil(2 * Lx / spacing)) + 1)
        xs = np.linspace(-Lx, Lx, nx)
        jx = rng.uniform(-jitter * spacing, jitter * spacing, nx)
        jx[~(np.abs(xs) < Lx - tol)] = 0.0
        xs = np.clip(xs + jx, -Lx, Lx)
        out.append(np.column_stack([xs, np.full(nx, sy * Ly)]))
    for sx in (-1.0, 1.0):
        ny = max(3, int(math.ceil(2 * Ly / spacing)) + 1)
        ys = np.linspace(-Ly, Ly, ny)
        jy = rng.uniform(-jitter * spacing, jitter * spacing, ny)
        jy[~(np.abs(ys) < Ly - tol)] = 0.0
        ys = np.clip(ys + jy, -Ly, Ly)
        out.append(np.column_stack([np.full(ny, sx * Lx), ys]))
    return np.vstack(out)


def background_seeds(Lx, Ly, spacing, centres, excl_r, seed=42, margin=None):
    """Jittered Cartesian fill; `margin` (fraction of spacing) = fix 1."""
    rng = np.random.default_rng(seed)
    xs = np.arange(-Lx + spacing / 2, Lx, spacing)
    ys = np.arange(-Ly + spacing / 2, Ly, spacing)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2))
    Gx, Gy = np.meshgrid(xs, ys)
    G = np.column_stack([Gx.ravel(), Gy.ravel()])
    G += rng.uniform(-0.15 * spacing, 0.15 * spacing, G.shape)
    keep = np.ones(len(G), bool)
    for a in centres:
        keep &= np.linalg.norm(G - np.asarray(a), axis=1) > excl_r
    G = G[keep]
    if margin is not None:
        # fix 1: pull jittered points back to distance >= margin*spacing from dQ
        m = margin * spacing
        G[:, 0] = np.clip(G[:, 0], -Lx + m, Lx - m)
        G[:, 1] = np.clip(G[:, 1], -Ly + m, Ly - m)
    return G


def deduplicate(P, tol):
    key = np.round(P / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


def delaunay_rect(P, Lx, Ly):
    tri = Delaunay(P)
    S = tri.simplices
    bc = P[S].mean(axis=1)
    S = S[(np.abs(bc[:, 0]) <= Lx + 1e-10) & (np.abs(bc[:, 1]) <= Ly + 1e-10)]
    used = np.unique(S)
    o2n = np.full(len(P), -1, dtype=np.int64)
    o2n[used] = np.arange(len(used))
    return P[used], o2n[S]


def build_mesh(h, Lx, Ly, centres, r_start, theta=None, c_radial=0.85,
               c_angular=0.75, c_face=1.3, c_bg=0.9, bg_margin=0.1, orient=True):
    centres = np.atleast_2d(np.asarray(centres, float))
    L = max(Lx, Ly)
    theta = h / L if theta is None else theta
    parts = [centres]
    parts += [ring_seeds(a, theta, Lx, Ly, r_start, c_radial, c_angular) for a in centres]
    parts.append(boundary_seeds(Lx, Ly, c_face * h))
    parts.append(background_seeds(Lx, Ly, c_bg * h, centres, max(20 * r_start, 1e-15),
                                  margin=bg_margin))
    P = deduplicate(np.vstack(parts), max(0.1 * c_angular * theta * r_start, 1e-15))
    X, T = delaunay_rect(P, Lx, Ly)
    if orient:
        a = X[T[:, 1]] - X[T[:, 0]]
        b = X[T[:, 2]] - X[T[:, 0]]
        neg = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0
        T[neg] = T[neg][:, [0, 2, 1]]
    return X, T


# --------------------------------------------------------------------------
# geometry checks
# --------------------------------------------------------------------------

def diameters(X, T):
    P = X[T]
    return np.max(np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1)
                            for i, j in ((0, 1), (0, 2), (1, 2))]), axis=0)


def signed_areas(X, T):
    a = X[T[:, 1]] - X[T[:, 0]]
    b = X[T[:, 2]] - X[T[:, 0]]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def conformity_report(X, T, Lx, Ly, tol=1e-9):
    """Single edges off dQ, nodes outside Q, edges shared by > 2 triangles."""
    nv = len(X)
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [0, 2]]]), axis=1)
    key = E[:, 0] * (nv + 1) + E[:, 1]
    u, cnt = np.unique(key, return_counts=True)
    se = np.stack([u // (nv + 1), u % (nv + 1)], 1)[cnt == 1]
    onb = lambda p: (np.abs(np.abs(p[:, 0]) - Lx) < tol) | (np.abs(np.abs(p[:, 1]) - Ly) < tol)
    bad = ~(onb(X[se[:, 0]]) & onb(X[se[:, 1]]))
    A = signed_areas(X, T)
    return {
        "n_single_edges": int(len(se)),
        "interior_free_edges": int(bad.sum()),
        "edges_gt2": int((cnt > 2).sum()),
        "nodes_outside_Q": int(((np.abs(X[:, 0]) > Lx + tol) | (np.abs(X[:, 1]) > Ly + tol)).sum()),
        "max_outside": float(max(0.0, (np.abs(X[:, 0]) - Lx).max(), (np.abs(X[:, 1]) - Ly).max())),
        "area_defect": float(np.abs(A).sum() - 4 * Lx * Ly),
        "n_negative_orientation": int((A < 0).sum()),
        "min_area": float(np.abs(A).min()),
    }


# --------------------------------------------------------------------------
# the paper's r_start rule (heuristic, see module docstring)
# --------------------------------------------------------------------------

_BARY7 = np.array([[1 / 3, 1 / 3, 1 / 3],
                   [0.797426985353087, 0.101286507323456, 0.101286507323456],
                   [0.101286507323456, 0.797426985353087, 0.101286507323456],
                   [0.101286507323456, 0.101286507323456, 0.797426985353087],
                   [0.470142064105115, 0.470142064105115, 0.059715871789770],
                   [0.470142064105115, 0.059715871789770, 0.470142064105115],
                   [0.059715871789770, 0.470142064105115, 0.470142064105115]])
_W7 = np.array([0.225, 0.125939180544827, 0.125939180544827, 0.125939180544827,
                0.132394440674613, 0.132394440674613, 0.132394440674613])


def _V(x, centres, charges):
    v = np.zeros(len(x))
    for a, z in zip(centres, charges):
        v -= z / np.maximum(np.linalg.norm(x - a, axis=1), 1e-14)
    return v


def cstar_ref_sampled(X, T, centres, charges):
    """max over elements not touching a nucleus of (h_K/2) max_7pts |V - V_K| (7-pt rule)."""
    P = X[T]
    d = np.min(np.stack([np.min(np.linalg.norm(P - a, axis=2), axis=1) for a in centres]), axis=0)
    reg = d >= 1e-10
    P = P[reg]
    hK = diameters(X, T[reg])
    Q = [b[0] * P[:, 0] + b[1] * P[:, 1] + b[2] * P[:, 2] for b in _BARY7]
    Vq = np.stack([_V(q, centres, charges) for q in Q])
    ch = _W7 @ Vq
    return float(np.max(0.5 * hK * np.max(np.abs(Vq - ch), axis=0)))


def fan_c0(r, n_ring, charges, S, p, nq_s=8, nq_t=6):
    """Original analytic patch estimate S^2 (sum_i n_ring I_K)^(1/p) for a fan of radius r."""
    xs, ws = leggauss(nq_s); xs = (xs + 1) / 2; ws = ws / 2
    xt, wt = leggauss(nq_t); xt = (xt + 1) / 2; wt = wt / 2
    span = 2 * math.pi / n_ring
    A = np.array([r, 0.0]); B = r * np.array([math.cos(span), math.sin(span)]) - A
    area = abs(A[0] * B[1] - A[1] * B[0]) / 2
    tot = 0.0
    for z in charges:
        dA = np.array([np.linalg.norm(A + t * B) for t in xt])
        ch = 2 * np.sum(ws[:, None] * wt[None, :] * (-z / dA)[None, :])
        V = -z / (xs[:, None] * dA[None, :])
        I = 2 * area * np.sum(ws[:, None] * wt[None, :] * np.abs(V - ch) ** p * xs[:, None])
        tot += n_ring * I
    return S ** 2 * tot ** (1 / p)


def paper_r_start(h, Lx, Ly, centres, charges, S=1.47, p=4 / 3, c_angular=0.75,
                  max_halvings=60, **kw):
    L = max(Lx, Ly); theta = h / L
    r = 0.4 * h ** 2 / L
    X0, T0 = build_mesh(h, Lx, Ly, centres, r, c_angular=c_angular, **kw)
    cref = cstar_ref_sampled(X0, T0, centres, charges)
    n_ring = max(4, int(math.ceil(2 * math.pi / (c_angular * theta))))
    for k in range(max_halvings + 1):
        c0 = fan_c0(r, n_ring, charges, S, p)
        if c0 <= cref:
            break
        r /= 2
    return {"r_start_rule": r, "n_halvings": k, "c_star_ref_sampled": cref, "c0_fan": c0,
            "dh_paper_heuristic": c0 + cref, "r_start_mesh": max(r, L * 1e-7)}


# --------------------------------------------------------------------------

def _parse_centres(s):
    return np.array([[float(v) for v in part.split(",")] for part in s.split(";")])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--Lx", type=float, default=5.0)
    ap.add_argument("--Ly", type=float, default=5.0)
    ap.add_argument("--centers", default="0,0", help='"x,y;x,y"')
    ap.add_argument("--charges", default=None, help='"Z1;Z2" (default all 1)')
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--theta", type=float, default=None, help="grading rate (default h/max(Lx,Ly))")
    ap.add_argument("--r-start", default="paper", help='"paper" or a float (first ring radius)')
    ap.add_argument("--c-radial", type=float, default=0.85)
    ap.add_argument("--c-angular", type=float, default=0.75)
    ap.add_argument("--c-face", type=float, default=1.3)
    ap.add_argument("--c-bg", type=float, default=0.9)
    ap.add_argument("--bg-margin", type=float, default=0.1)
    ap.add_argument("--paper-compat", action="store_true",
                    help="reproduce the original exactly (no background margin, no reorientation)")
    ap.add_argument("--S", type=float, default=1.47, help="S_8 used only by the paper r_start rule")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    C = _parse_centres(a.centers)
    Z = np.ones(len(C)) if a.charges is None else np.array([float(v) for v in a.charges.split(";")])
    margin = None if a.paper_compat else a.bg_margin
    kw = dict(theta=a.theta, c_radial=a.c_radial, c_face=a.c_face, c_bg=a.c_bg,
              bg_margin=margin, orient=not a.paper_compat)
    t0 = time.time()
    if a.r_start == "paper":
        rule = paper_r_start(a.h, a.Lx, a.Ly, C, Z, S=a.S, c_angular=a.c_angular, **kw)
        r_mesh = rule["r_start_mesh"]
    else:
        rule = {}
        r_mesh = float(a.r_start)
    X, T = build_mesh(a.h, a.Lx, a.Ly, C, r_mesh, c_angular=a.c_angular, **kw)
    hK = diameters(X, T)
    rep = conformity_report(X, T, a.Lx, a.Ly)
    os.makedirs(a.out, exist_ok=True)
    np.savetxt(os.path.join(a.out, "nodes.dat"), X, fmt="%.15e")
    np.savetxt(os.path.join(a.out, "elements.dat"), T + 1, fmt="%d")
    info = {"generator": "gen_mesh_2d.py", "paper_compat": bool(a.paper_compat),
            "domain": "rect", "Lx": a.Lx, "Ly": a.Ly, "centers": C.tolist(), "charges": Z.tolist(),
            "h": a.h, "theta": a.theta if a.theta else a.h / max(a.Lx, a.Ly),
            "c_radial": a.c_radial, "c_angular": a.c_angular, "c_face": a.c_face, "c_bg": a.c_bg,
            "bg_margin": margin, "r_start_mesh": r_mesh, **rule,
            "n_nodes": int(len(X)), "n_tris": int(len(T)), "h_max": float(hK.max()),
            "hK_min": float(hK.min()), **rep,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": time.time() - t0}
    with open(os.path.join(a.out, "mesh_report.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(a.out, "info.txt"), "w") as f:
        for k, v in info.items():
            f.write(f"{k} = {v}\n")
    print(json.dumps({k: info[k] for k in ("n_nodes", "n_tris", "h_max", "hK_min", "r_start_mesh",
                                           "interior_free_edges", "nodes_outside_Q", "area_defect")}))


if __name__ == "__main__":
    main()
