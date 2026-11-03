#!/usr/bin/env python3
"""Compare a regenerated tetrahedral mesh with an original one.
usage: compare_meshes.py <orig_dir> <new_dir> <boundary: ball:R | box:Lx,Ly,Lz>
Prints node/element counts, max node-coordinate difference (if same node count),
h_max, facet census (single facets off the geometric boundary = interior_free_facets)."""
import sys, json, numpy as np

def load(d):
    X = np.loadtxt(f"{d}/nodes.dat"); T = np.loadtxt(f"{d}/elements.dat", dtype=np.int64) - 1
    return X, T

def census(X, T, onb):
    F = np.sort(np.concatenate([T[:, [1, 2, 3]], T[:, [0, 2, 3]], T[:, [0, 1, 3]], T[:, [0, 1, 2]]]), axis=1)
    u, c = np.unique(F, axis=0, return_counts=True)
    s = u[c == 1]
    bnode = onb(X)
    bad = int((~bnode[s].all(axis=1)).sum())
    return dict(n_facets=int(len(u)), single=int(len(s)), interior_free_facets=bad, over_shared=int((c > 2).sum()),
                single_on_boundary=int(len(s) - bad))

def stats(X, T):
    P = X[T]
    e = [np.linalg.norm(P[:, i] - P[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)]
    hK = np.max(e, axis=0)
    vol = np.abs(np.einsum('ij,ij->i', P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))) / 6
    return dict(n_nodes=int(len(X)), n_tets=int(len(T)), h_max=float(hK.max()), hK_min=float(hK.min()),
                vol_min=float(vol.min()), vol_total=float(vol.sum()))

def main():
    o, n, b = sys.argv[1:4]
    kind, par = b.split(":")
    if kind == "ball":
        Rb = float(par); onb = lambda X: np.linalg.norm(X, axis=1) > Rb - 1e-6
    else:
        L = np.array([float(x) for x in par.split(",")]); onb = lambda X: (np.abs(np.abs(X) - L) < 1e-9).any(axis=1)
    out = {}
    for tag, d in (("orig", o), ("new", n)):
        X, T = load(d)
        out[tag] = {**stats(X, T), **census(X, T, onb)}
        if tag == "orig": Xo, To = X, T
    Xn, Tn = load(n)
    if len(Xn) == len(Xo):
        out["max_node_diff"] = float(np.abs(Xn - Xo).max())
        out["same_tet_vertex_sets"] = int(len(np.intersect1d(
            np.ascontiguousarray(np.sort(To, axis=1)).view([('', To.dtype)] * 4).ravel(),
            np.ascontiguousarray(np.sort(Tn, axis=1)).view([('', Tn.dtype)] * 4).ravel())))
    print("JSON " + json.dumps(out))

main()
