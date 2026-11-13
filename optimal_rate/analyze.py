#!/usr/bin/env python3
"""Summarize the optimal-rate runs without assuming an infinite-domain reference."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLUMNS = [
    "case", "h", "n_elem", "n_node", "h_max", "hK_min", "rho_target",
    "patch_rmax", "n_patch", "G1_theta_outside_patch", "min_angle_deg",
    "C_h", "C_h_sq", "Gamma_h", "A_h", "d_h", "d_h_patch", "d_h_off",
    "d_h_geometric_upper", "d_h_geometric_upper_over_h2",
    "C_cert", "sigma_opt", "L_opt_1", "L_fixed_1",
    "U_neumann_1", "U_dirichlet_1", "gap_neumann_opt", "gap_dirichlet_opt",
    "gap_neumann_fixed", "gap_dirichlet_fixed",
    "whole_space_lower_gap", "whole_space_upper_gap",
    "cecr_residual_opt_max", "p1_residual_neumann_max",
    "EOC_hmax_d_h", "EOC_hmax_Gamma_h", "EOC_hmax_gap_neumann_opt",
    "interior_free_edges", "area_defect", "wall_s",
]
METRICS = [
    "d_h", "d_h_patch", "d_h_off", "d_h_geometric_upper", "Gamma_h", "A_h", "C_h_sq",
    "patch_rmax", "rho_target", "gap_neumann_opt", "gap_dirichlet_opt",
    "gap_neumann_fixed", "whole_space_lower_gap", "whole_space_upper_gap",
]


def first(value):
    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


def numeric(value):
    value = first(value)
    if isinstance(value, (float, int)) and math.isfinite(value):
        return float(value)
    return None


def difference(a, b):
    a, b = numeric(a), numeric(b)
    return a - b if a is not None and b is not None else None


def pick(record, *keys):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def peak(value):
    if isinstance(value, (tuple, list)):
        values = [numeric(v) for v in value]
        values = [v for v in values if v is not None]
        return max(values) if values else None
    return numeric(value)


def make_row(data):
    mesh = data.get("mesh_report", {})
    if not isinstance(mesh, dict):
        mesh = {}
    row = {key: data.get(key) for key in COLUMNS}
    row["case"] = str(data.get("case", data.get("problem", "unknown")))
    row["h"] = numeric(data.get("h", mesh.get("h")))
    row["rho_target"] = pick(mesh, "rho", "rho_target", "patch_radius", "r_start")
    if row["rho_target"] is None and row["h"] is not None:
        beta = numeric(mesh.get("beta"))
        row["rho_target"] = beta * row["h"] ** 4 if beta is not None else None
    row["G1_theta_outside_patch"] = pick(
        mesh, "G1_theta_outside_patch", "g1_theta", "theta_actual",
        "max_hK_over_h_rK", "g1_ratio_max", "theta_off_core")
    row["min_angle_deg"] = pick(mesh, "min_angle_deg", "min_angle")
    row["interior_free_edges"] = pick(
        mesh, "interior_free_edges", "interior_free_facets")
    row["area_defect"] = mesh.get("area_defect")
    if row["C_h"] is not None:
        row["C_h_sq"] = numeric(row["C_h"]) ** 2
    row["L_opt_1"] = numeric(data.get("L_opt"))
    row["L_fixed_1"] = numeric(data.get("L"))
    row["U_neumann_1"] = numeric(data.get("U_neumann"))
    row["U_dirichlet_1"] = numeric(data.get("U_dirichlet"))
    row["gap_neumann_opt"] = difference(row["U_neumann_1"], row["L_opt_1"])
    row["gap_dirichlet_opt"] = difference(row["U_dirichlet_1"], row["L_opt_1"])
    row["gap_neumann_fixed"] = difference(row["U_neumann_1"], row["L_fixed_1"])
    row["gap_dirichlet_fixed"] = difference(row["U_dirichlet_1"], row["L_fixed_1"])
    # Exact whole-space reference is justified only for the one-centre case.
    centres = pick(mesh, "centers", "centres")
    nc = mesh.get("n_centers", len(centres) if isinstance(centres, list) else None)
    case = row["case"].lower().replace("_", "")
    one_centre = nc == 1 or case in ("case7", "square") or "onecenter" in case or "onecentre" in case
    if one_centre:
        row["whole_space_lower_gap"] = difference(-1.0, row["L_opt_1"])
        row["whole_space_upper_gap"] = difference(row["U_dirichlet_1"], -1.0)
    row["cecr_residual_opt_max"] = peak(data.get("cecr_resid_opt"))
    row["p1_residual_neumann_max"] = peak(data.get("p1_resid_neumann"))
    return row


def eoc(previous, current, key, hkey="h"):
    hp, hc = numeric(previous.get(hkey)), numeric(current.get(hkey))
    vp, vc = numeric(previous.get(key)), numeric(current.get(key))
    if hp is None or hc is None or hp <= hc or hc <= 0:
        return None
    if vp is None or vc is None or min(vp, vc) <= 0:
        return None
    return math.log(vp / vc) / math.log(hp / hc)


def write_csv(path, columns, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def fmt(value, precision=3):
    v = numeric(value)
    return f"{v:.{precision}g}" if v is not None else "—"


def plot_case(case, rows, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: CSV, Markdown and LaTeX summaries written")
        return
    fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.3), constrained_layout=True)
    hs = [r["h"] for r in rows]
    panels = [
        (axs[0], [("d_h", r"$d_h$"), ("d_h_patch", "patch"), ("d_h_off", "off patch"),
                  ("d_h_geometric_upper", "geometry-only upper"),
                  ("Gamma_h", r"$\Gamma_h$"), ("C_h_sq", r"$C_h^2$")],
         "Corrections", "correction"),
        (axs[1], [("gap_neumann_opt", r"$U_N-L_{\rm opt}$"),
                  ("gap_neumann_fixed", r"$U_N-L_{\rm fixed}$")],
         "Same-domain Neumann enclosures", "enclosure width"),
        (axs[2], [("gap_dirichlet_opt", r"$U_D-L_{\rm opt}$"),
                  ("whole_space_lower_gap", r"$-1-L_{\rm opt}$"),
                  ("whole_space_upper_gap", r"$U_D+1$")],
         "Dirichlet / whole-space comparison", "gap"),
    ]
    for ax, curves, title, ylabel in panels:
        reference = None
        for key, label in curves:
            pairs = [(r["h"], numeric(r.get(key))) for r in rows]
            pairs = [(h, v) for h, v in pairs if h and v and v > 0]
            if not pairs:
                continue
            xx, yy = zip(*pairs)
            ax.loglog(xx, yy, "o-", markersize=4, label=label)
            if reference is None:
                reference = (xx[-1], yy[-1])
        if reference:
            h0, v0 = reference
            ax.loglog(hs, [v0 * (h / h0) ** 2 for h in hs], "k--",
                      alpha=0.5, label=r"$O(h^2)$")
            ax.legend(fontsize=8)
        ax.set(title=title, xlabel="nominal h", ylabel=ylabel)
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(case + ": shrinking patch; fixed truncation domain")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"convergence_{case}.{suffix}", dpi=180)
    plt.close(fig)



def plot_mesh(case, h, root):
    """Plot the coarsest mesh globally, near its first nucleus, and in the core."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reports = sorted((root / "meshes").glob("*/mesh_report.json"))
    for report_path in reports:
        report = json.loads(report_path.read_text())
        if report.get("case") != case or abs(report.get("h", -1) - h) > 1e-12:
            continue
        folder = report_path.parent
        vertices = np.loadtxt(folder / "nodes.dat")
        triangles = np.loadtxt(folder / "elements.dat", dtype=int) - 1
        centre = np.asarray(report["centers"][0])
        rho = report["rho_target"]
        fig, axs = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True)
        axs[0].triplot(vertices[:, 0], vertices[:, 1], triangles, lw=0.22, color="#304d6d")
        axs[0].scatter(*np.asarray(report["centers"]).T, color="#b62030", s=18, zorder=4)
        axs[0].set(title="Full domain", xlabel="x", ylabel="y")
        local = vertices - centre
        for ax, radius, scale, title in [
            (axs[1], 0.05, 1.0, "Near the first nucleus"),
            (axs[2], 1.12 * rho, rho, r"Core: coordinates divided by $\rho_h$"),
        ]:
            p = local[triangles]
            chosen = np.all(p.max(axis=1) >= -radius, axis=1) & np.all(p.min(axis=1) <= radius, axis=1)
            ax.triplot(local[:, 0] / scale, local[:, 1] / scale, triangles[chosen],
                       lw=0.35, color="#304d6d")
            ax.scatter([0], [0], color="#b62030", s=18, zorder=4)
            ax.set(xlim=(-radius / scale, radius / scale), ylim=(-radius / scale, radius / scale),
                   title=title, xlabel="x - nucleus" if scale == 1 else r"$(x-a_x)/\rho_h$",
                   ylabel="y - nucleus" if scale == 1 else r"$(y-a_y)/\rho_h$")
        for ax in axs:
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(f"{case}, h={h:g}, target core radius {rho:.3g}")
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(root / f"mesh_{case}_h{h:g}.{suffix}", dpi=180)
        plt.close(fig)
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.results or root / "results"
    paths = sorted(source.glob("*.json"))
    rows = []
    for path in paths:
        with path.open() as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or "d_h" not in data:
            continue
        row = make_row(data)
        if row["h"] is None:
            raise ValueError(f"Missing h in {path}")
        rows.append(row)
    geom_path = root / "geometric_certificate.json"
    if geom_path.exists():
        geometric = json.loads(geom_path.read_text()).get("records", [])
        geom_index = {(r["case"], r["h"]): r for r in geometric}
        for row in rows:
            bound = geom_index.get((row["case"], row["h"]), {})
            row["d_h_geometric_upper"] = bound.get("d_h_upper")
            row["d_h_geometric_upper_over_h2"] = bound.get("d_h_upper_over_h2")
    rows.sort(key=lambda r: (r["case"], -r["h"]))
    if not rows:
        raise SystemExit(f"No computation results with d_h found in {source}")
    groups = {}
    for row in rows:
        groups.setdefault(row["case"], []).append(row)
    convergence = []
    for case, values in groups.items():
        for previous, current in zip(values, values[1:]):
            rate = {
                "case": case, "h_previous": previous["h"], "h": current["h"],
                "h_max_previous": previous.get("h_max"), "h_max": current.get("h_max"),
                **{"EOC_" + key: eoc(previous, current, key) for key in METRICS},
                **{"EOC_hmax_" + key: eoc(previous, current, key, "h_max") for key in METRICS},
            }
            current.update({key: value for key, value in rate.items() if key.startswith("EOC_hmax_")})
            convergence.append(rate)
    write_csv(root / "summary.csv", COLUMNS, rows)
    write_csv(root / "convergence.csv", ["case", "h_previous", "h", "h_max_previous", "h_max"] +
              ["EOC_" + key for key in METRICS] + ["EOC_hmax_" + key for key in METRICS], convergence)
    text = [
        "# Optimal-rate computations", "",
        "The target is second order in nominal h. The Neumann width compares "
        "bounds for the same finite-domain eigenvalue. Dirichlet and whole-space "
        "gaps also include truncation error; they need not decay to zero on a fixed domain. "
        "Tables use nominal h EOCs; convergence.csv also contains actual h_max EOCs.",
        "", "A separate geometry-only full-core competitor is recorded in "
        "[geometric_certificate.csv](geometric_certificate.csv); it avoids potential "
        "and Lp quadrature.",
        "", "All reported bounds use floating-point quadrature and eigensolves; "
        "they are numerical evidence, not interval-certified enclosures.", "",
    ]
    rate_index = {(r["case"], r["h"]): r for r in convergence}
    tex = ["% Generated by optimal_rate/analyze.py"]
    for case, values in groups.items():
        text += ["## " + case, "",
                 "| h | triangles | patch radius | d_h | EOC d_h | Gamma_h | "
                 "EOC Gamma_h | U_N - L_opt | EOC width | U_D - L_opt |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        tex += [r"\begin{table}[htbp]", r"\centering\small",
                r"\begin{tabular}{rrrrrrrr}", r"\hline",
                r"$h$ & $N_T$ & $d_h$ & EOC & $\Gamma_h$ & EOC & $U_N-L_{\rm opt}$ & EOC \\",
                r"\hline"]
        for row in values:
            rate = rate_index.get((case, row["h"]), {})
            entries = [fmt(row["h"]), str(row.get("n_elem", "")),
                       fmt(row.get("patch_rmax")), fmt(row["d_h"]),
                       fmt(rate.get("EOC_d_h")), fmt(row["Gamma_h"]),
                       fmt(rate.get("EOC_Gamma_h")), fmt(row["gap_neumann_opt"]),
                       fmt(rate.get("EOC_gap_neumann_opt")),
                       fmt(row["gap_dirichlet_opt"])]
            text.append("| " + " | ".join(entries) + " |")
            tex_entries = [entries[i] for i in (0, 1, 3, 4, 5, 6, 7, 8)]
            tex.append(" & ".join("-" if v == "—" else v for v in tex_entries) + r" \\")
        text += ["", f"Plot: [convergence_{case}.png](convergence_{case}.png)", ""]
        caption_case = case.replace("_", r"\_")
        tex += [r"\hline", r"\end{tabular}",
                r"\caption{" + caption_case +
                r": shrinking-patch convergence. Floating-point results.}", r"\end{table}", ""]
        plot_case(case, values, root)
        plot_mesh(case, values[0]["h"], root)
    (root / "convergence.md").write_text("\n".join(text) + "\n")
    (root / "tables.tex").write_text("\n".join(tex) + "\n")
    print(f"Wrote {len(rows)} runs across {len(groups)} cases to {root}")


if __name__ == "__main__":
    main()

