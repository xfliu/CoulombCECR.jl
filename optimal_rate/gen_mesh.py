#!/usr/bin/env python3
"""Shape-regular rectangular-ring meshes with an expanded O(h^4) core patch.

No Delaunay triangulation, radius floor, or optional Python packages are used.
The core is a Cartesian grid through the nucleus; homothetic rectangles
connect it to the boundary. Two-centre subdomains share identical interface
nodes. The perturbation theorem permits this expanded union-of-cells patch.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import time


def build(case, h, beta, density):
    if h <= 0 or beta <= 0 or density <= 0:
        raise ValueError("h, beta and density must be positive")
    if case == "2d_one_centre":
        blocks = [(-5., 5., -5., 5., 0., 0.)]
        centers, charges, Lx, Ly = [(0., 0.)], [1.], 5., 5.
    elif case == "2d_two_centre":
        blocks = [(-7., 0., -5., 5., -2., 0.),
                  (0., 7., -5., 5., 2., 0.)]
        centers, charges, Lx, Ly = [(-2., 0.), (2., 0.)], [1., 1.], 7., 5.
    else:
        raise ValueError(case)
    n = max(2, math.ceil(density / h))
    rho = beta * h ** 4
    nodes, triangles, core_ids, layers = [], [], [], []
    # Only outer interface coordinates are merged. Never deduplicate tiny
    # core nodes by an absolute-coordinate rounding grid.
    outer_ids = {}

    def node(x, y, boundary=False):
        if boundary:
            key = (x, y)
            if key in outer_ids:
                return outer_ids[key]
        idx = len(nodes)
        nodes.append((x, y))
        if boundary:
            outer_ids[(x, y)] = idx
        return idx

    def split_quad(a, b, c, d, parity):
        if parity % 2:
            triangles.extend([(a, b, d), (b, c, d)])
        else:
            triangles.extend([(a, b, c), (a, c, d)])

    for xmin, xmax, ymin, ymax, cx, cy in blocks:
        xs = [xmin + (cx - xmin) * i / n for i in range(n)] + [cx + (xmax - cx) * i / n for i in range(n + 1)]
        ys = [ymin + (cy - ymin) * i / n for i in range(n)] + [cy + (ymax - cy) * i / n for i in range(n + 1)]
        maxrad = max(math.hypot(x-cx, y-cy) for x in (xmin, xmax) for y in (ymin, ymax))
        s0 = rho / maxrad
        if not 0 < s0 < 1:
            raise ValueError("core radius must be smaller than subdomain")
        grid = [[node(cx + s0 * (x-cx), cy + s0 * (y-cy)) for x in xs] for y in ys]
        for j in range(2*n):
            for i in range(2*n):
                core_ids.extend((len(triangles), len(triangles)+1))
                split_quad(grid[j][i], grid[j][i+1], grid[j+1][i+1], grid[j+1][i], i+j)
        ij = ([(i, 0) for i in range(2*n)] + [(2*n, j) for j in range(2*n)]
              + [(i, 2*n) for i in range(2*n, 0, -1)] + [(0, j) for j in range(2*n, 0, -1)])
        prev = [grid[j][i] for i, j in ij]
        nl = math.ceil(math.log(1/s0) / math.log1p(1/n))
        layers.append(nl)
        for layer in range(1, nl+1):
            scale = 1. if layer == nl else math.exp(math.log(s0) * (1-layer/nl))
            ring = [node(xs[i] if layer == nl else cx + scale*(xs[i]-cx),
                         ys[j] if layer == nl else cy + scale*(ys[j]-cy), layer == nl) for i, j in ij]
            for k in range(len(ring)):
                kn = (k+1) % len(ring)
                # CCW vertices: inner next, inner current, outer current, outer next.
                split_quad(prev[kn], prev[k], ring[k], ring[kn], k+layer)
            prev = ring

    # Independent geometric checks and explicit G1 diagnostic outside the core.
    edges = collections.Counter()
    area_sum, amin, amax, hmax, theta_max, minangle = 0., math.inf, 0., 0., 0., math.inf
    core_set = set(core_ids)
    core_rmax = 0.
    for e, ids in enumerate(triangles):
        p = [nodes[v] for v in ids]
        area = ((p[1][0]-p[0][0])*(p[2][1]-p[0][1]) - (p[1][1]-p[0][1])*(p[2][0]-p[0][0]))/2
        if area <= 0:
            raise RuntimeError(f"nonpositive element area {e}: {area}")
        area_sum += area
        amin, amax = min(amin, area), max(amax, area)
        lens = [math.dist(p[i], p[j]) for i, j in ((0,1),(1,2),(2,0))]
        hk = max(lens)
        hmax = max(hmax, hk)
        for i in range(3):
            u, v = p[(i+1)%3], p[(i+2)%3]
            ax, ay, bx, by = u[0]-p[i][0], u[1]-p[i][1], v[0]-p[i][0], v[1]-p[i][1]
            minangle = min(minangle, math.atan2(abs(ax*by-ay*bx), ax*bx+ay*by))
            edges[tuple(sorted((ids[i], ids[(i+1)%3])))] += 1
        # Distance from a centre outside this convex triangle to its edges.
        near = math.inf
        nearest_center = None
        for c in centers:
            ds = []
            for i in range(3):
                a, b = p[i], p[(i+1)%3]
                vx, vy = b[0]-a[0], b[1]-a[1]
                t = max(0., min(1., ((c[0]-a[0])*vx+(c[1]-a[1])*vy)/(vx*vx+vy*vy)))
                ds.append(math.hypot((a[0]-c[0])+t*vx, (a[1]-c[1])+t*vy))
            d = min(ds)
            if d < near:
                near, nearest_center = d, c
        if e in core_set:
            core_rmax = max(core_rmax, max(math.dist(x, nearest_center) for x in p))
        else:
            theta_max = max(theta_max, hk/(h*near))

    def same_side(a, b):
        return any(abs(a[d]-s) < 1e-12 and abs(b[d]-s) < 1e-12
                   for d, lim in ((0,Lx),(1,Ly)) for s in (-lim,lim))
    bad = sum(count == 1 and not same_side(nodes[a],nodes[b]) for (a,b),count in edges.items())
    nonmanifold = sum(count > 2 for count in edges.values())
    outside = sum(abs(x)>Lx or abs(y)>Ly for x,y in nodes)
    defect = area_sum-4*Lx*Ly
    if bad or nonmanifold or outside or abs(defect)>1e-8*Lx*Ly:
        raise RuntimeError(f"mesh invalid: free={bad}, nonmanifold={nonmanifold}, outside={outside}, area_defect={defect}")
    report = dict(generator="homothetic_rectangular_rings",case=case,h=h,beta=beta,gamma_p=4,
                  rho_target=rho,core_rmax=core_rmax,core_beta_measured=core_rmax/h**4,
                  density=density,n_per_half_side=n,radial_layers=layers,centers=centers,charges=charges,
                  Lx=Lx,Ly=Ly,n_nodes=len(nodes),n_elem=len(triangles),n_core=len(core_ids),
                  h_max=hmax,min_area=amin,max_area=amax,min_angle_deg=math.degrees(minangle),
                  theta_off_core=theta_max,interior_free_edges=bad,nonmanifold_edges=nonmanifold,
                  nodes_outside=outside,area_defect=defect,patch_definition="expanded Cartesian core, union of cells")
    return nodes, triangles, core_ids, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",choices=["2d_one_centre","2d_two_centre"],required=True)
    ap.add_argument("--h",type=float,required=True)
    ap.add_argument("--beta",type=float,default=1e-6)
    ap.add_argument("--density",type=float,default=4.)
    ap.add_argument("--out",type=Path,required=True)
    args = ap.parse_args()
    start = time.monotonic()
    nodes, triangles, core, report = build(args.case,args.h,args.beta,args.density)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/"nodes.dat").open("w") as f:
        f.writelines(f"{x:.17e} {y:.17e}\n" for x,y in nodes)
    with (args.out/"elements.dat").open("w") as f:
        f.writelines(f"{a+1} {b+1} {c+1}\n" for a,b,c in triangles)
    (args.out/"core_elements.dat").write_text("".join(f"{e+1}\n" for e in core))
    report["elapsed_s"] = time.monotonic()-start
    (args.out/"mesh_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
