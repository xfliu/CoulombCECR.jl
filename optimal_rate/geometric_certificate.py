#!/usr/bin/env python3
"""A non-quadrature O(h^2) perturbation competitor from mesh geometry alone."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def centre_distances(points, centre):
    """Exact segment-distance formula in Float64; caller excludes core cells."""
    distances = np.full(len(points), np.inf)
    a = points - centre
    for i, j in ((0, 1), (1, 2), (2, 0)):
        u, v = a[:, i], a[:, j]
        edge = v - u
        denom = np.einsum("ij,ij->i", edge, edge)
        if np.any(denom <= 0):
            raise ValueError("Zero-length mesh edge")
        t = np.clip(-np.einsum("ij,ij->i", u, edge) / denom, 0.0, 1.0)
        distance = np.linalg.norm(u + t[:, None] * edge, axis=1)
        distances = np.minimum(distances, distance)
    return distances


def certificate(folder):
    report = json.loads((folder / "mesh_report.json").read_text())
    x = np.loadtxt(folder / "nodes.dat")
    t = np.loadtxt(folder / "elements.dat", dtype=int).reshape(-1, 3) - 1
    core_ids = np.loadtxt(folder / "core_elements.dat", dtype=int).reshape(-1) - 1
    if len(np.unique(core_ids)) != len(core_ids):
        raise ValueError(f"Duplicate core element in {folder}")
    if not len(core_ids) or np.min(core_ids) < 0 or np.max(core_ids) >= len(t):
        raise ValueError(f"Invalid core element list in {folder}")
    core = np.zeros(len(t), dtype=bool)
    core[core_ids] = True
    p = x[t]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    double_area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    if np.any(double_area <= 0):
        raise ValueError(f"Nonpositive triangle in {folder}")
    hk = np.maximum.reduce([np.linalg.norm(p[:, i] - p[:, j], axis=1)
                            for i, j in ((0, 1), (1, 2), (2, 0))])
    centres = np.asarray(report["centers"], dtype=float)
    charges = np.asarray(report["charges"], dtype=float)
    if len(centres) != len(charges) or np.any(charges <= 0):
        raise ValueError("Invalid Coulomb centres/charges")
    # Each connected core block belongs to its closest nucleus. Taking the
    # minimum over nuclei of its farthest vertex distance verifies the cover.
    distances = np.stack([np.linalg.norm(p[core] - c, axis=2).max(axis=1)
                          for c in centres], axis=1)
    core_radius = float(distances.min(axis=1).max())
    rho = max(float(report["rho_target"]), core_radius)
    outside = p[~core]
    dc = np.stack([centre_distances(outside, c) for c in centres], axis=1)
    if np.any(dc <= 0):
        raise ValueError("A singularity is not enclosed by the designated core")
    # Explicit containment test: the nucleus-star must be part of the core.
    for c in centres:
        matching = np.all(x == c, axis=1)
        if matching.sum() != 1:
            raise ValueError(f"Nucleus is missing or duplicated: {c}")
        star = np.any(matching[t], axis=1)
        if not np.all(core[star]):
            raise ValueError("The full nucleus-star is not inside the core")
    q = 4.0 / 3.0
    s = 1.466 if len(centres) == 1 else 1.462
    # Bound integral over the union of radius-rho balls for each 1/r_i.
    integrals = np.full(len(centres), 2 * math.pi / (2 - q) * rho ** (2 - q))
    for i, ci in enumerate(centres):
        for j, cj in enumerate(centres):
            if i == j:
                continue
            sep = float(np.linalg.norm(ci - cj))
            if sep <= 2 * rho:
                raise ValueError("The geometric core balls must be disjoint")
            integrals[i] += math.pi * rho ** 2 / (sep - rho) ** q
    patch = 2 * s ** 2 * float(np.sum(charges * integrals ** (1 / q)))
    off_values = hk[~core] ** 2 / 2 * np.sum(charges[None, :] / dc ** 2, axis=1)
    off = float(off_values.max())
    h = float(report["h"])
    theta = float(np.max(hk[~core] / (h * dc.min(axis=1))))
    return {
        "case": report["case"], "h": h, "n_elem": len(t),
        "rho_target": report["rho_target"], "core_radius_measured": core_radius,
        "rho_cover": rho, "rho_over_h4": rho / h ** 4,
        "n_core": len(core_ids), "theta_off_core": theta,
        "d_patch_upper": patch, "d_off_upper": off, "d_h_upper": patch + off,
        "d_h_upper_over_h2": (patch + off) / h ** 2,
        "arithmetic": "Float64 geometric formulas; no potential or Lp quadrature",
        "mesh": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
    }


def local_eoc(previous, row, key):
    if previous is None:
        return None
    return math.log(previous[key] / row[key]) / math.log(previous["h"] / row["h"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    records = [certificate(path.parent)
               for path in sorted((root / "meshes").glob("*/mesh_report.json"))]
    if not records:
        raise SystemExit("No generated meshes found")
    records.sort(key=lambda row: (row["case"], -row["h"]))
    previous = {}
    for row in records:
        prev = previous.get(row["case"])
        for key in ("rho_cover", "d_patch_upper", "d_off_upper", "d_h_upper"):
            row["EOC_" + key] = local_eoc(prev, row, key)
        previous[row["case"]] = row
    output = {
        "description": "A priori full-core competitor in the direct perturbation estimate",
        "certification": "Floating-point geometry, not interval certification",
        "patch_formula": "2 S^2 sum_i Z_i [2pi/(2-p) rho^(2-p) + sum_(j!=i) pi rho^2/(|a_i-a_j|-rho)^p]^(1/p)",
        "off_formula": "max_(K outside core) h_K^2/2 sum_i Z_i/dist(a_i,K)^2",
        "p": 4 / 3,
        "records": records,
    }
    (root / "geometric_certificate.json").write_text(json.dumps(output, indent=2) + "\n")
    keys = ["case", "h", "n_elem", "rho_target", "core_radius_measured", "rho_cover",
            "rho_over_h4", "n_core", "theta_off_core", "d_patch_upper", "d_off_upper",
            "d_h_upper", "d_h_upper_over_h2", "EOC_rho_cover", "EOC_d_patch_upper",
            "EOC_d_off_upper", "EOC_d_h_upper", "arithmetic", "mesh"]
    with (root / "geometric_certificate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    for row in records:
        rate = row["EOC_d_h_upper"]
        print(f'{row["case"]} h={row["h"]:g}: d_upper={row["d_h_upper"]:.8e}; '
              f'd_upper/h^2={row["d_h_upper_over_h2"]:.8e}; EOC={rate}')
    print(f"Wrote geometric certificates for {len(records)} meshes")


if __name__ == "__main__":
    main()

