#!/usr/bin/env python3
"""
gen_box_h2plus.py -- graded Delaunay tetrahedral mesh of the box
Q = [-Lx,Lx] x [-Ly,Ly] x [-Lz,Lz] with Coulomb centres as mesh vertices.

Cleaned, self-contained port of the generator that produced the paper's H2+ mesh
case6_3d_h2plus_box/meshes_R6/box_R6_h100_opt, i.e.
  case5_3d_hydrogen_cube/mesh_generator/find_optimal_hmin_3d.py  (adaptive r_start loop)
  + gen_mesh_linear_3d.py                                         (seed builders, Delaunay)
(NOT case6_3d_h2plus_box/mesh_generator/box_mesh_h2plus.py, which targets [-7,7]x[-5,5]^2 and
writes a different info.txt). The algorithm and every constant are unchanged:

  theta = h / L_ref, L_ref = max(Lx,Ly,Lz); growth per shell 1 + c_radial*theta;
  shells of Fibonacci points around each centre from r_start outwards, angular spacing
  min(c_angular*theta*r, h); jittered face seeds (c_face*h); jittered Cartesian background
  (c_bg*h) outside 20*r_start of the centres; dedup; scipy Delaunay; keep tets with
  barycentre in Q.
  r_start: start at 0.4*h^2/L_ref; c_*ref = max_{K not touching a centre} (h_K/2) sup_K|V-V_K|
  (5-point Keast sampling) on the first mesh; halve r_start and rebuild until
  c_0 = Sq0^2 (sum_{K touching a centre} ||V-V_K||^{3/2}_{L^{3/2}(K)})^{2/3} <= c_*ref.

NOTE: the quantities c_*ref / c_0 / dh_om0 printed here are the ORIGINAL (sampled, not
certified) estimates used only to choose r_start; the certified d_h is computed by
CoulombCECR.direct_dh. Added here: facet census (conformity check) and mesh comparison.

usage (paper mesh):
  python gen_box_h2plus.py --Lx 8 --Ly 6 --Lz 6 --centers "-2,0,0;2,0,0" --charges "1;1" \
      --h 1.0 --Sq0 0.619 --kappa_sig 0.4 --output box_R6_h100_regen [--compare <orig_dir>]
"""
import argparse, datetime, json, math, os, time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import Delaunay

P0 = 1.5

# ---------------------------------------------------------------- seed points
def fibonacci_sphere(n, shell_index=0):
    # scalar math.* loop kept verbatim from the original so that the seed points are
    # bit-identical (Delaunay of near-cospherical points is sensitive to the last ulp)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    off = shell_index * 2.399963
    pts = []
    for k in range(n):
        th = math.acos(1.0 - 2.0 * (k + 0.5) / n)
        ph = math.fmod(2.0 * math.pi * k / phi + off, 2.0 * math.pi)
        pts.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    return np.array(pts)

def add_atom_shells(pts, atom, theta, L, r_start, c_radial, c_angular, h_cap):
    R_max = math.sqrt(sum(x * x for x in L)) * 1.05
    growth = 1.0 + c_radial * theta
    r, idx = r_start, 0
    while r < R_max:
        sp = min(c_angular * theta * r, h_cap)
        n = max(6, int(math.ceil(4.0 * math.pi * r ** 2 / sp ** 2)))
        P = np.asarray(atom) + r * fibonacci_sphere(n, idx)
        keep = np.all(np.abs(P) <= np.asarray(L) + 1e-12, axis=1)
        pts.extend(P[keep].tolist())
        r *= growth; idx += 1

def add_box_faces(pts, L, spacing, jitter=0.04, seed=7):
    rng = np.random.default_rng(seed)
    edge_tol = 0.3 * spacing
    for axis in range(3):
        i1, i2 = [k for k in range(3) if k != axis]
        L1, L2 = L[i1], L[i2]
        for sign in (-1.0, 1.0):
            n1 = max(3, int(math.ceil(2 * L1 / spacing)) + 1)
            n2 = max(3, int(math.ceil(2 * L2 / spacing)) + 1)
            G1, G2 = np.meshgrid(np.linspace(-L1, L1, n1), np.linspace(-L2, L2, n2))
            face = np.zeros((G1.size, 3))
            face[:, axis] = sign * L[axis]; face[:, i1] = G1.ravel(); face[:, i2] = G2.ravel()
            interior = (np.abs(face[:, i1]) < L1 - edge_tol) & (np.abs(face[:, i2]) < L2 - edge_tol)
            j = rng.uniform(-jitter * spacing, jitter * spacing, (G1.size, 2))
            j[~interior] = 0.0
            face[:, i1] = np.clip(face[:, i1] + j[:, 0], -L1, L1)
            face[:, i2] = np.clip(face[:, i2] + j[:, 1], -L2, L2)
            pts.extend(face.tolist())

def add_box_interior_bg(pts, L, spacing, atoms, excl_r, seed=42):
    rng = np.random.default_rng(seed)
    ax = [np.arange(-l + spacing / 2, l, spacing) for l in L]
    if min(len(a) for a in ax) == 0:
        return
    G = np.column_stack([g.ravel() for g in np.meshgrid(*ax, indexing='ij')])
    G += rng.uniform(-0.15 * spacing, 0.15 * spacing, G.shape)
    mask = np.ones(len(G), dtype=bool)
    for a in atoms:
        mask &= np.linalg.norm(G - np.asarray(a), axis=1) > excl_r
    pts.extend(G[mask].tolist())

def deduplicate(pts, tol):
    arr = np.array(pts)
    _, idx = np.unique(np.round(arr / tol).astype(np.int64), axis=0, return_index=True)
    return arr[np.sort(idx)]

def delaunay_box(pts, L):
    tri = Delaunay(pts)
    bary = pts[tri.simplices].mean(axis=1)
    keep = np.all(np.abs(bary) <= np.asarray(L) + 1e-10, axis=1)
    T = tri.simplices[keep]
    used = np.unique(T)
    o2n = np.full(len(pts), -1, dtype=int); o2n[used] = np.arange(len(used))
    return pts[used], o2n[T]

def build_mesh(h, L, r_start, atoms, c_radial=0.85, c_angular=0.75, c_face=0.65, c_bg=0.57):
    theta = h / max(L)
    pts = [list(a) for a in atoms]
    for a in atoms:
        add_atom_shells(pts, a, theta, L, r_start, c_radial, c_angular, h)
    add_box_faces(pts, L, c_face * h)
    add_box_interior_bg(pts, L, c_bg * h, atoms, max(r_start * 20, 1e-15))
    return delaunay_box(deduplicate(pts, max(0.1 * c_angular * theta * r_start, 1e-15)), L)

# ------------------------------------------------- original r_start criterion
def _gl01(n):
    x, w = leggauss(n); return (x + 1) / 2, w / 2
_XS, _WS = _gl01(6); _XT, _WT = _gl01(5); _XU, _WU = _gl01(4)
_K5L = np.array([[.25, .25, .25], [1/6, 1/6, 1/6], [1/6, 1/6, .5], [1/6, .5, 1/6], [.5, 1/6, 1/6]])
_K5W = np.array([-4/5, 9/20, 9/20, 9/20, 9/20])

def per_elem(X, T, atoms, charges, p0=P0):
    """r_K (min vertex distance to a centre), I_K = ||V-V_K||^p0_{L^p0(K)}, cstar_K, h_K
    exactly as in find_optimal_hmin_3d.per_elem_3d (Duffy 6x5x4 on singular tets, Keast-5 else)."""
    V = [X[T[:, i]] for i in range(4)]
    r_K = np.full(len(T), np.inf)
    for a in atoms:
        r_K = np.minimum(r_K, np.min([np.linalg.norm(v - a, axis=1) for v in V], axis=0))
    h_K = np.max([np.linalg.norm(V[i] - V[j], axis=1) for i in range(4) for j in range(i + 1, 4)], axis=0)
    vol = np.abs(np.einsum('ij,ij->i', V[1] - V[0], np.cross(V[2] - V[0], V[3] - V[0]))) / 6
    I_K = np.zeros(len(T)); cs = np.zeros(len(T))
    sing = r_K < 1e-10
    for k in np.where(sing)[0]:
        v = X[T[k]].copy()
        for a in atoms:
            dv = np.linalg.norm(v - a, axis=1); i = int(np.argmin(dv))
            if dv[i] < 1e-10:
                v = v[[i] + [j for j in range(4) if j != i]]; break
        A, B, C = v[1] - v[0], v[2] - v[1], v[3] - v[2]
        vk = abs(np.dot(v[1] - v[0], np.cross(v[2] - v[0], v[3] - v[0]))) / 6
        cs[k] = np.inf
        if vk < 1e-40:
            continue
        # NB: as in the original, only the singular centre's potential is used on singular tets
        csum = 0.0; Ik = 0.0
        for s, ws in zip(_XS, _WS):
            for t, wt in zip(_XT, _WT):
                for u, wu in zip(_XU, _WU):
                    d = np.linalg.norm(A + t * (B + u * C))
                    for q in charges:
                        csum += ws * wt * wu * (-q / d) * s * t
        ch = 6 * csum
        for s, ws in zip(_XS, _WS):
            for t, wt in zip(_XT, _WT):
                for u, wu in zip(_XU, _WU):
                    d = np.linalg.norm(A + t * (B + u * C))
                    Vq = -sum(q / (s * d) for q in charges)
                    Ik += ws * wt * wu * abs(Vq - ch) ** p0 * s * s * t
        I_K[k] = 6 * vk * Ik
    reg = ~sing
    if reg.any():
        Vr = [v[reg] for v in V]; ch = np.zeros(reg.sum()); sup = np.zeros(reg.sum()); Ik = np.zeros(reg.sum())
        def Vat(lam):
            xq = lam[0] * Vr[0] + lam[1] * Vr[1] + lam[2] * Vr[2] + (1 - sum(lam)) * Vr[3]
            return -sum(q / np.maximum(np.linalg.norm(xq - a, axis=1), 1e-14) for a, q in zip(atoms, charges))
        for q in range(5):
            ch += _K5W[q] * Vat(_K5L[q])
        for q in range(5):
            df = np.abs(Vat(_K5L[q]) - ch); Ik += _K5W[q] * df ** p0 * vol[reg]; sup = np.maximum(sup, df)
        I_K[reg] = Ik; cs[reg] = h_K[reg] / 2 * sup
    return r_K, I_K, cs, h_K

def find_r_start(h, L, atoms, charges, Sq0, kappa_sig, max_halvings=30, c0_factor=1.0):
    r = 0.4 * h ** 2 / max(L)
    X0, T0 = build_mesh(h, L, r, atoms)
    rK, _, cs, _ = per_elem(X0, T0, atoms, charges)
    cref = float(np.max(cs[rK >= 1e-10]))
    print(f"  initial r_start={r:.4e}: {len(X0)} nodes {len(T0)} tets  c*_ref={cref:.4e}", flush=True)
    X, T, nh, c0 = X0, T0, 0, np.inf
    for nh in range(max_halvings + 1):
        if nh > 0:
            try:
                Xc, Tc = build_mesh(h, L, r, atoms)
            except Exception as e:
                print(f"  Delaunay failed at r_start={r:.2e} ({type(e).__name__}); halving"); r /= 2; continue
        else:
            Xc, Tc = X0, T0
        rK, IK, _, _ = per_elem(Xc, Tc, atoms, charges)
        ns = int((rK < 1e-10).sum()); c0 = Sq0 ** 2 * IK[rK < 1e-10].sum() ** (1 / P0)
        print(f"  #{nh:2d} r_start={r:.4e} nodes={len(Xc)} tets={len(Tc)} n_sing={ns} c0={c0:.4e} c0/c*={c0/cref:.3f}", flush=True)
        if ns == 0:
            print("  no singular tets; keeping previous mesh"); break
        X, T = Xc, Tc
        if c0 <= c0_factor * cref:
            break
        r /= 2
    return dict(X=X, T=T, r_start=r, n_halvings=nh, c_star_ref=cref, c0_om0=c0, dh_om0=c0 + cref,
                eps_h_om0=(c0 + cref) / kappa_sig, theta=h / max(L))

# ------------------------------------------------------------ checks / output
def facet_census(X, T, L):
    F = np.sort(np.concatenate([T[:, [1, 2, 3]], T[:, [0, 2, 3]], T[:, [0, 1, 3]], T[:, [0, 1, 2]]]), axis=1)
    u, c = np.unique(F, axis=0, return_counts=True)
    s = u[c == 1]
    onb = (np.abs(np.abs(X) - np.asarray(L)) < 1e-9).any(axis=1)
    bad = int((~onb[s].all(axis=1)).sum())
    return dict(n_facets=int(len(u)), single=int(len(s)), interior_free_facets=bad, over_shared=int((c > 2).sum()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Lx', type=float, default=8.0); ap.add_argument('--Ly', type=float, default=6.0)
    ap.add_argument('--Lz', type=float, default=6.0)
    ap.add_argument('--centers', default='-2,0,0;2,0,0'); ap.add_argument('--charges', default='1;1')
    ap.add_argument('--h', type=float, default=1.0)
    ap.add_argument('--Sq0', type=float, default=0.619); ap.add_argument('--kappa_sig', type=float, default=0.4)
    ap.add_argument('--r_start', type=float, default=None, help='skip the halving loop, use this r_start')
    ap.add_argument('--output', required=True); ap.add_argument('--compare', default=None)
    a = ap.parse_args()
    L = [a.Lx, a.Ly, a.Lz]
    atoms = np.array([[float(x) for x in p.split(',')] for p in a.centers.split(';')])
    charges = np.array([float(x) for x in a.charges.split(';')])
    t0 = time.time()
    if a.r_start is None:
        res = find_r_start(a.h, L, atoms, charges, a.Sq0, a.kappa_sig)
    else:
        X, T = build_mesh(a.h, L, a.r_start, atoms)
        res = dict(X=X, T=T, r_start=a.r_start, n_halvings=-1, theta=a.h / max(L))
    X, T = res['X'], res['T']
    P = X[T]
    hK = np.max([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)], axis=0)
    vol = np.abs(np.einsum('ij,ij->i', P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))) / 6
    cen = facet_census(X, T, L)
    rep = {k: v for k, v in res.items() if k not in ('X', 'T')}
    rep.update(n_nodes=int(len(X)), n_tets=int(len(T)), hmax=float(hK.max()), hK_min=float(hK.min()),
               vol_min=float(vol.min()), vol_total=float(vol.sum()), box_volume=8 * a.Lx * a.Ly * a.Lz,
               elapsed_s=time.time() - t0, **cen)
    if a.compare:
        Xo = np.loadtxt(os.path.join(a.compare, 'nodes.dat')); To = np.loadtxt(os.path.join(a.compare, 'elements.dat'), dtype=np.int64) - 1
        rep['orig_n_nodes'] = int(len(Xo)); rep['orig_n_tets'] = int(len(To))
        rep['orig_census'] = facet_census(Xo, To, L)
        if len(Xo) == len(X):
            rep['max_node_diff'] = float(np.abs(Xo - X).max())
            key = lambda TT: set(map(tuple, np.sort(TT, axis=1).tolist()))
            rep['identical_tets'] = len(key(To) & key(T))
    os.makedirs(a.output, exist_ok=True)
    np.savetxt(os.path.join(a.output, 'nodes.dat'), X, fmt='%.15e')
    np.savetxt(os.path.join(a.output, 'elements.dat'), T + 1, fmt='%d')
    with open(os.path.join(a.output, 'info.txt'), 'w') as f:
        f.write('generator = gen_box_h2plus.py\n')
        for k, v in rep.items():
            f.write(f'{k} = {v}\n')
        f.write(f'created_at = {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n')
    print('JSON ' + json.dumps(rep))

if __name__ == '__main__':
    main()
