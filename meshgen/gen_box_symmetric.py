#!/usr/bin/env python3
"""
gen_box_symmetric.py -- mirror-symmetric graded tetrahedral mesh of the H2+ box
Q = [-Lx,Lx] x [-Ly,Ly] x [-Lz,Lz] (default [-8,8]x[-6,6]^2, nuclei (-2,0,0), (2,0,0)).

A mesh of the octant Q+ = [0,Lx]x[0,Ly]x[0,Lz] is built with the grading recipe of
gen_box_h2plus.py around the nucleus a = (2,0,0) (which lies on the octant edge y=z=0),
then reflected across x=0, y=0, z=0; nodes on the planes are merged. The result is
exactly invariant under the three reflections (negation is exact in floating point),
so CoulombCECR.cecr_eigs_symmetric can split the CECR eigenproblem into 8 sectors.

Recipe (constants of gen_box_h2plus.py; theta is NOT tied to h here):
  * shells around a from r_start, radius growth 1 + c_radial*theta, angular spacing
    sp = min(c_angular*theta*r, h), up to R_max = 1.05*|L|;
    shell = Fibonacci points (n = 4 pi r^2/sp^2, shell-dependent rotation as in the original)
    restricted to y >= band*sp, z >= band*sp  (interior of the quarter sphere)
    + points EXACTLY on the half great circles in the planes z = 0 and y = 0
      (angle step pi/ceil(pi r/sp), shell-dependent offset) + the two poles on the x-axis;
    shell points with x < band*sp are dropped; the shells of the mirror nucleus (-2,0,0) are
    added in the same way (as in the original, both shell families fill the region near x = 0),
    and the plane x = 0 gets exact points on the quarter circles where the shells cut it;
  * face seeds on the 6 octant faces (spacing c_face*h, jitter 0.04*spacing away from the
    face edges); on the symmetry planes no face seed within 20*r_start of a nucleus;
  * jittered Cartesian background (c_bg*h, jitter 0.15*spacing) outside 20*r_start;
  * dedup (tol 0.1*c_angular*theta*r_start), scipy Delaunay, zero-volume tets removed;
  * reflection + merge; nuclei moved to node indices 1..nc.
Defaults: theta = 0.125 (the value of the original at h = 1), r_start = 1.5625e-3
(the paper mesh box_R6_h100_opt).

usage:
  python gen_box_symmetric.py --h 1.0 --output box_sym_h100 [--compare <orig_dir>]
Output: nodes.dat, elements.dat (1-based), info.txt, mesh_report.json.
"""
import argparse, datetime, json, math, os, time
import numpy as np
from scipy.spatial import Delaunay


def fibonacci_sphere(n, shell_index=0):
    # as in gen_box_h2plus.py (vectorised; bit-identity with the original is not needed here)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    k = np.arange(n)
    th = np.arccos(1.0 - 2.0 * (k + 0.5) / n)
    ph = np.fmod(2.0 * math.pi * k / phi + shell_index * 2.399963, 2.0 * math.pi)
    return np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def quarter_shell(a, r, sp, idx, band):
    """points of the sphere |x-a| = r with y >= 0, z >= 0 (a on the line y = z = 0),
    containing exact points on the planes y = 0 and z = 0."""
    n = max(6, int(math.ceil(4.0 * math.pi * r ** 2 / sp ** 2)))
    U = fibonacci_sphere(n, idx)
    U = U[(U[:, 1] * r >= band * sp) & (U[:, 2] * r >= band * sp)]
    P = [a + r * U]
    m = max(2, int(math.ceil(math.pi * r / sp)))
    # different offsets on the two planes: points mirrored by y <-> z would form coplanar
    # (isosceles-trapezoid) quadruples across the edge y = z = 0 -> zero-volume tets
    for plane, s in ((2, math.fmod(idx * 0.6180339887498949, 1.0)),
                     (1, math.fmod(idx * 0.6180339887498949 + 0.5, 1.0))):
        t = math.pi * (np.arange(m) + s) / m
        t = t[(t > 0.3 * math.pi / m) & (t < math.pi - 0.3 * math.pi / m)]
        Q = np.zeros((len(t), 3))
        Q[:, 0] = a[0] + r * np.cos(t); Q[:, 3 - plane] = r * np.sin(t)   # Q[:, plane] = 0 exactly
        P.append(Q)
    P.append(np.array([[a[0] + r, 0.0, 0.0], [a[0] - r, 0.0, 0.0]]))   # poles on the x-axis
    P = np.vstack(P)
    P[:, 1] = np.where(P[:, 1] == 0.0, 0.0, P[:, 1])       # no -0.0
    return P


def plane_x0_points(ax, theta, c_angular, h_cap, L):
    """graded polar point set on the quarter plane {x = 0, 0 <= y <= Ly, 0 <= z <= Lz}, with local
    spacing f(rho) = min(c_angular*theta*r, h), r = sqrt(rho^2 + ax^2) the distance to the nearest
    nucleus (= the angular spacing of the shells there); rings rho_{k+1} = rho_k + f(rho_k),
    quarter-arc points with spacing f (shell-dependent offset), ring ends exactly on the edges."""
    f = lambda rho: min(c_angular * theta * math.sqrt(rho * rho + ax * ax), h_cap)
    out = [np.zeros((1, 3))]
    rho, k = f(0.0), 0
    rmax = math.hypot(L[1], L[2])
    while rho < rmax:
        sp = f(rho)
        m = max(1, int(math.ceil(0.5 * math.pi * rho / sp)))
        s = math.fmod(k * 0.6180339887498949 + 0.25, 1.0)
        t = 0.5 * math.pi * (np.arange(m) + s) / m
        t = t[(t > 0.3 * math.pi / (2 * m)) & (t < 0.5 * math.pi - 0.3 * math.pi / (2 * m))]
        Q = np.zeros((len(t) + 2, 3))
        Q[:len(t), 1] = rho * np.cos(t); Q[:len(t), 2] = rho * np.sin(t)
        Q[len(t), 1] = rho; Q[len(t) + 1, 2] = rho
        Q = Q[(Q[:, 1] <= L[1]) & (Q[:, 2] <= L[2])]
        out.append(Q)
        rho += sp; k += 1
    return np.vstack(out)


def add_atom_shells(pts, a, theta, L, r_start, c_radial, c_angular, h_cap, band):
    R_max = math.sqrt(sum(x * x for x in L)) * 1.05
    growth = 1.0 + c_radial * theta
    r, idx = r_start, 0
    while r < R_max:
        sp = min(c_angular * theta * r, h_cap)
        P = quarter_shell(a, r, sp, idx, band)
        keep = np.all(P <= np.asarray(L), axis=1) & (P[:, 0] >= band * sp) & (P[:, 1] >= 0) & (P[:, 2] >= 0)
        pts.append(P[keep])
        r *= growth; idx += 1
def add_octant_faces(pts, L, spacing, atoms, excl_r, jitter=0.04, seed=7):
    rng = np.random.default_rng(seed)
    edge_tol = 0.3 * spacing
    for axis in range(3):
        i1, i2 = [k for k in range(3) if k != axis]
        L1, L2 = L[i1], L[i2]
        for val in (0.0, L[axis]):
            n1 = max(3, int(math.ceil(L1 / spacing)) + 1)
            n2 = max(3, int(math.ceil(L2 / spacing)) + 1)
            G1, G2 = np.meshgrid(np.linspace(0, L1, n1), np.linspace(0, L2, n2))
            face = np.zeros((G1.size, 3))
            face[:, axis] = val; face[:, i1] = G1.ravel(); face[:, i2] = G2.ravel()
            interior = (face[:, i1] > edge_tol) & (face[:, i1] < L1 - edge_tol) & \
                       (face[:, i2] > edge_tol) & (face[:, i2] < L2 - edge_tol)
            j = rng.uniform(-jitter * spacing, jitter * spacing, (G1.size, 2))
            j[~interior] = 0.0
            face[:, i1] = np.clip(face[:, i1] + j[:, 0], 0, L1)
            face[:, i2] = np.clip(face[:, i2] + j[:, 1], 0, L2)
            if val == 0.0:
                for a in atoms:
                    face = face[np.linalg.norm(face - a, axis=1) > excl_r]
            pts.append(face)


def add_octant_bg(pts, L, spacing, atoms, excl_r, seed=42):
    rng = np.random.default_rng(seed)
    ax = [np.arange(spacing / 2, l, spacing) for l in L]
    G = np.column_stack([g.ravel() for g in np.meshgrid(*ax, indexing='ij')])
    G += rng.uniform(-0.15 * spacing, 0.15 * spacing, G.shape)
    mask = np.all((G > 0) & (G < np.asarray(L)), axis=1)
    for a in atoms:
        mask &= np.linalg.norm(G - a, axis=1) > excl_r
    pts.append(G[mask])


def deduplicate(arr, tol):
    _, idx = np.unique(np.round(arr / tol).astype(np.int64), axis=0, return_index=True)
    return arr[np.sort(idx)]


def tet_measures(X, T):
    P = X[T]
    hK = np.max([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)], axis=0)
    vol = np.abs(np.einsum('ij,ij->i', P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))) / 6
    return hK, vol


def dihedral_stats(X, T):
    P = X[T]
    faces = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    N = []
    for k, (a, b, c) in enumerate(faces):
        n = np.cross(P[:, b] - P[:, a], P[:, c] - P[:, a])
        s = np.sign(np.einsum('ij,ij->i', n, P[:, k] - P[:, a]))
        n = -n * s[:, None]                                   # outward
        N.append(n / np.linalg.norm(n, axis=1)[:, None])
    ang = []
    for i in range(4):
        for j in range(i + 1, 4):
            cosv = np.clip(-np.einsum('ij,ij->i', N[i], N[j]), -1, 1)
            ang.append(np.degrees(np.arccos(cosv)))
    ang = np.array(ang)
    mn = ang.min(axis=0); mx = ang.max(axis=0)
    hK, vol = tet_measures(X, T)
    # normalised shape quality: 6*sqrt(2)*vol / hK^3 (1 for the regular tet)
    q = 6 * math.sqrt(2) * vol / hK ** 3
    return dict(dihedral_min=float(mn.min()), dihedral_max=float(mx.max()),
                dihedral_min_p001=float(np.quantile(mn, 0.001)), dihedral_min_p01=float(np.quantile(mn, 0.01)),
                dihedral_min_median=float(np.median(mn)),
                n_dihedral_lt_5deg=int((mn < 5).sum()), n_dihedral_lt_1deg=int((mn < 1).sum()),
                n_dihedral_gt_175deg=int((mx > 175).sum()),
                shape_q_min=float(q.min()), shape_q_p01=float(np.quantile(q, 0.01)), shape_q_median=float(np.median(q)))


def facet_census(X, T, L):
    F = np.sort(np.concatenate([T[:, [1, 2, 3]], T[:, [0, 2, 3]], T[:, [0, 1, 3]], T[:, [0, 1, 2]]]), axis=1)
    u, c = np.unique(F, axis=0, return_counts=True)
    s = u[c == 1]
    onb = (np.abs(np.abs(X) - np.asarray(L)) < 1e-9)          # per coordinate
    # a single facet is a boundary facet iff its 3 vertices lie on a common box face
    common = onb[s[:, 0]] & onb[s[:, 1]] & onb[s[:, 2]]
    bad = int((~common.any(axis=1)).sum())
    return dict(n_facets=int(len(u)), single=int(len(s)), interior_free_facets=bad, over_shared=int((c > 2).sum()))


def reflect_merge(X, T, axes=(0, 1, 2)):
    for a in axes:
        nv = len(X)
        on = X[:, a] == 0.0
        newid = np.arange(nv)
        off = ~on
        newid[off] = nv + np.arange(off.sum())
        Y = X[off].copy(); Y[:, a] = -Y[:, a]
        X = np.vstack([X, Y]); T = np.vstack([T, newid[T]])
    return X, T


def symmetry_check(X, T):
    """every node and every tet has its mirror image under x_i -> -x_i, i = 1,2,3 (exact)"""
    key = {tuple(x): i for i, x in enumerate(X.tolist())}
    St = set(map(tuple, np.sort(T, axis=1).tolist()))
    out = {}
    for a in range(3):
        Y = X.copy(); Y[:, a] = -Y[:, a]
        Y[Y == 0] = 0.0
        p = np.array([key.get(tuple(y), -1) for y in Y.tolist()])
        nmiss = int((p < 0).sum())
        tm = 0
        if nmiss == 0:
            tm = sum(1 for t in np.sort(p[T], axis=1).tolist() if tuple(t) not in St)
        out[f'axis{a+1}'] = dict(nodes_without_mirror=nmiss, tets_without_mirror=int(tm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Lx', type=float, default=8.0); ap.add_argument('--Ly', type=float, default=6.0)
    ap.add_argument('--Lz', type=float, default=6.0)
    ap.add_argument('--centers', default='2,0,0', help='nuclei in the closed octant (mirrored automatically)')
    ap.add_argument('--h', type=float, default=1.0)
    ap.add_argument('--theta', type=float, default=0.125)
    ap.add_argument('--r_start', type=float, default=1.5625e-3)
    ap.add_argument('--band', type=float, default=0.4, help='shell points kept at distance >= band*sp from symmetry planes')
    ap.add_argument('--output', required=True); ap.add_argument('--compare', default=None)
    ap.add_argument('--no_quality', action='store_true')
    a = ap.parse_args()
    c_radial, c_angular, c_face, c_bg = 0.85, 0.75, 0.65, 0.57
    L = [a.Lx, a.Ly, a.Lz]
    atoms = np.array([[float(x) for x in p.split(',')] for p in a.centers.split(';')])
    assert np.all(atoms[:, 1:] == 0) and np.all(atoms[:, 0] > 0), "nuclei must lie on the positive x-axis"
    t0 = time.time()
    pts = [atoms.copy()]
    # shells of every nucleus of the full box (incl. the mirror images -a), restricted to the octant:
    # the octant point set is then the restriction of a point set of the original type; the plane x=0
    # gets a graded polar point set (plane_x0_points)
    for at in atoms:
        add_atom_shells(pts, at, a.theta, L, a.r_start, c_radial, c_angular, a.h, a.band)
        add_atom_shells(pts, at * np.array([-1.0, 1.0, 1.0]), a.theta, L, a.r_start, c_radial, c_angular, a.h, a.band)
    pts.append(plane_x0_points(float(atoms[:, 0].min()), a.theta, c_angular, a.h, L))
    add_octant_faces(pts, L, c_face * a.h, atoms, 20 * a.r_start)
    add_octant_bg(pts, L, c_bg * a.h, atoms, 20 * a.r_start)
    P = np.vstack(pts)
    P[P == 0] = 0.0
    assert P.min() >= 0 and np.all(P <= np.asarray(L))
    P = deduplicate(P, max(0.1 * c_angular * a.theta * a.r_start, 1e-15))
    t_pts = time.time() - t0
    tri = Delaunay(P)
    T = tri.simplices.astype(np.int64)
    ncop = int(len(tri.coplanar))
    hK, vol = tet_measures(P, T)
    flat = vol <= 1e-12 * hK ** 3
    T = T[~flat]
    used = np.unique(T)
    o2n = np.full(len(P), -1, dtype=np.int64); o2n[used] = np.arange(len(used))
    missing_atoms = [i for i in range(len(atoms)) if o2n[i] < 0]
    assert not missing_atoms, "a nucleus is not a mesh vertex"
    Xo, To = P[used], o2n[T]
    t_del = time.time() - t0 - t_pts
    # octant census: free facets must lie on one of the 6 octant faces
    F = np.sort(np.concatenate([To[:, [1, 2, 3]], To[:, [0, 2, 3]], To[:, [0, 1, 3]], To[:, [0, 1, 2]]]), axis=1)
    u, cnt = np.unique(F, axis=0, return_counts=True); s = u[cnt == 1]
    onf = np.concatenate([Xo == 0.0, Xo == np.asarray(L)], axis=1)
    cen_oct = dict(n_tets=int(len(To)), n_nodes=int(len(Xo)), single=int(len(s)),
                   free_off_faces=int((~(onf[s[:, 0]] & onf[s[:, 1]] & onf[s[:, 2]]).any(axis=1)).sum()),
                   over_shared=int((cnt > 2).sum()), removed_flat=int(flat.sum()), qhull_coplanar=ncop,
                   vol=float(tet_measures(Xo, To)[1].sum()))
    X, T = reflect_merge(Xo, To)
    # nuclei first
    full_atoms = np.vstack([atoms * [-1, 1, 1], atoms])
    order = []
    for at in full_atoms:
        i = np.where(np.all(X == at, axis=1))[0]; assert len(i) == 1; order.append(int(i[0]))
    rest = np.setdiff1d(np.arange(len(X)), order)
    perm = np.concatenate([order, rest]); inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    X, T = X[perm], inv[T]
    elapsed = time.time() - t0
    hK, vol = tet_measures(X, T)
    rep = dict(generator='gen_box_symmetric.py', domain='box3d_mirror_symmetric', h=a.h, theta=a.theta,
               r_start=a.r_start, band=a.band, L=L, centers=full_atoms.tolist(),
               n_nodes=int(len(X)), n_tets=int(len(T)), hmax=float(hK.max()), hK_min=float(hK.min()),
               vol_min=float(vol.min()), vol_total=float(vol.sum()), box_volume=8 * a.Lx * a.Ly * a.Lz,
               octant=cen_oct, t_points_s=t_pts, t_delaunay_s=t_del, elapsed_s=elapsed,
               **facet_census(X, T, L))
    rep['symmetry'] = symmetry_check(X, T)
    if not a.no_quality:
        rep['quality'] = dihedral_stats(X, T)
    if a.compare:
        Xc = np.loadtxt(os.path.join(a.compare, 'nodes.dat')); Tc = np.loadtxt(os.path.join(a.compare, 'elements.dat'), dtype=np.int64) - 1
        hc, vc = tet_measures(Xc, Tc)
        rep['orig'] = dict(dir=a.compare, n_nodes=int(len(Xc)), n_tets=int(len(Tc)), hmax=float(hc.max()),
                           hK_min=float(hc.min()), vol_total=float(vc.sum()), **facet_census(Xc, Tc, L))
        if not a.no_quality:
            rep['orig']['quality'] = dihedral_stats(Xc, Tc)
    os.makedirs(a.output, exist_ok=True)
    np.savetxt(os.path.join(a.output, 'nodes.dat'), X, fmt='%.15e')
    np.savetxt(os.path.join(a.output, 'elements.dat'), T + 1, fmt='%d')
    with open(os.path.join(a.output, 'info.txt'), 'w') as f:
        f.write('domain = box3d\n')
        for k in ('generator', 'h', 'theta', 'r_start', 'band', 'n_nodes', 'n_tets', 'hK_min', 'hmax', 'vol_total',
                  'interior_free_facets', 'elapsed_s'):
            f.write(f'{k} = {rep[k]}\n')
        f.write('symmetry = mirror x=0, y=0, z=0 (octant mesh reflected)\n')
        f.write(f'created_at = {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n')
    with open(os.path.join(a.output, 'mesh_report.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    print('JSON ' + json.dumps(rep))


if __name__ == '__main__':
    main()
