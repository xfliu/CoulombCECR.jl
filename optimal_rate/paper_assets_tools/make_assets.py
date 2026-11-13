#!/usr/bin/env python3
"""Build manuscript-only optimal-rate tables and figures from saved JSON.

The numerical files are never changed. Tables distinguish a fixed-epsilon,
form-bound-selected reference shift from per-index epsilon optimization.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.labelsize": 10, "axes.titlesize": 10,
                     "legend.fontsize": 8.5, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def num(x, places=6):
    if x is None or not math.isfinite(float(x)):
        return "--"
    return f"{x:.{places}f}"


def sci(x, digits=2):
    if x == 0:
        return "$0$"
    a, e = f"{x:.{digits}e}".split("e")
    return f"${a}\\times10^{{{int(e)}}}$"


def integer(x):
    return f"{x:,}".replace(",", "\\,")


def hlabel(h):
    for n in (3,):
        if math.isclose(h,2/n,rel_tol=1e-12,abs_tol=1e-14):
            return f"$2/{n}$"
    return f"{h:g}"


def endpoint(x, lower):
    return f"{(math.floor(x*1e6) if lower else math.ceil(x*1e6))/1e6:.6f}"


def selected_mu(r, k):
    if "mu_minus_sigma_best" in r:
        return r["mu_minus_sigma_best"][k]
    target = r["sigma_best"][k]
    solves = r["perk_solves"]
    s = min(solves, key=lambda s: abs(s["sigma"]-target))
    if abs(s["sigma"]-target) > 1e-10:
        raise ValueError("no exact solve at selected shift")
    return s["mu_minus_sigma"][k]


def get_h(r):
    if "h" in r and r["h"] is not None:
        return float(r["h"])
    return int(r["mesh"].split("_h")[-1].split("_")[0])/100


def read_cases(root):
    cases = {}
    for case in ("2d_one_centre", "2d_two_centre"):
        files = sorted((root/"optimal_rate/results_complete").glob(f"{case}_h*.json"))
        rows = [dict(json.loads(p.read_text()),_source_json=str(p.relative_to(root))) for p in files]
        for r in rows:
            r["h"] = get_h(r)
        cases[case] = sorted(rows, key=lambda r: r["h"], reverse=True)
    files = sorted((root/"results/v1.1/3d").glob("3d_hydrogen*.json"))
    rows = [dict(json.loads(p.read_text()),_source_json=str(p.relative_to(root))) for p in files]
    for r in rows:
        r["h"] = get_h(r)
    cases["3d_hydrogen"] = sorted(rows, key=lambda r: r["h"], reverse=True)
    # Capped files are the canonical family. Never mix the exploratory
    # uncapped computations into paper tables or EOC estimates.
    files = sorted((root/"optimal_rate/results_complete").glob("3d_h2plus_cap_h*.json"))
    rows = [dict(json.loads(p.read_text()),_source_json=str(p.relative_to(root))) for p in files]
    for r in rows:
        r["h"] = get_h(r)
    if rows:
        cases["3d_h2plus"] = sorted(rows,key=lambda r:r["h"],reverse=True)
    return cases


def spectral_table(rows, case, out):
    spec = {
        "2d_one_centre": ("one_centre_2d.tex", "tab:coulomb-2d-cecr",
                          r"Two-dimensional one-centre problem on $[-5,5]^2$."),
        "2d_two_centre": ("two_centre_2d.tex", "tab:twocenter-2d-cecr",
                          r"Two-dimensional two-centre problem on $[-7,7]\times[-5,5]$."),
        "3d_hydrogen": ("hydrogen_3d.tex", "tab:coulomb-3d-cecr",
                        r"Three-dimensional hydrogen on $B(0,6)$."),
        "3d_h2plus": ("two_centre_3d.tex", "tab:twocenter-3d-cecr",
                      r"Three-dimensional $\mathrm{H}_2^+$ on $[-8,8]\times[-6,6]^2$, using shrinking-core meshes with $h_{\max}\le h$.")}
    fn, label, caption = spec[case]
    caption += (r" $L_k^{\rm ref}$ uses the fixed $\epsilon$ in \Cref{tab:setup}"
                r" and its form-bound-selected shift; $L_k^{\rm opt}$ uses the displayed $\epsilon_k$."
                r" The CECR value in the third column is recomputed at that selected shift."
                r" $U_k^N$ and $U_k$ are conforming Neumann and Dirichlet Ritz values."
                r" The displayed bounds are rounded outwards to six decimal places.")
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", "\\caption{"+caption+"}",
             "\\label{"+label+"}", r"\begin{tabular}{ccrrrrrr}", r"\toprule",
             r"$h$ & $k$ & $\mu_{k,h}^{\sigma_k}-\sigma_k$ & $\epsilon_k$ & $L_k^{\rm ref}$ & $L_k^{\rm opt}$ & $U_k^N$ & $U_k$ \\",
             r"\midrule"]
    for j, r in enumerate(rows):
        for k in range(3):
            hstr = hlabel(r['h']) if k == 0 else ""
            cells = [hstr, str(k+1), num(selected_mu(r,k)), num(r["eps_best"][k],3),
                     endpoint(r["L_opt"][k],True), endpoint(r["L_best"][k],True),
                     endpoint(r["U_neumann"][k],False), endpoint(r["U_dirichlet"][k],False)]
            lines.append(" & ".join(cells)+r" \\")
        if j != len(rows)-1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out/"tables"/fn).write_text("\n".join(lines)+"\n")


def constants_table(cases, out):
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{3.7pt}",
             r"\caption{Mesh data and correction constants for the optimal-rate families."
             r" $\rho_0$ is the designed singular-core radius, or the first-shell radius for 3D hydrogen."
             r" The displayed $C_\epsilon^{\rm cert}$ and $\sigma_{\rm ref}=C_\epsilon^{\rm cert}+1-\epsilon$"
             r" correspond to the fixed $\epsilon$ in \Cref{tab:setup}.}",
             r"\label{tab:optimal-mesh-constants}",
             r"\begin{tabular}{crrrrrrr}", r"\toprule",
             r"$h$ & $N_{\rm cell}$ & $h_{\max}$ & $\rho_0$ & $\Gamma_h$ & $d_h$ & $C_\epsilon^{\rm cert}$ & $\sigma_{\rm ref}$ \\",
             r"\midrule"]
    titles = {"2d_one_centre": "2D, one centre", "2d_two_centre": "2D, two centres",
              "3d_hydrogen": "3D, hydrogen", "3d_h2plus": r"3D, $\mathrm{H}_2^+$"}
    for ci, (case,rows) in enumerate(cases.items()):
        if not rows:
            continue
        lines.append(r"\multicolumn{8}{l}{\textit{"+titles[case]+r"}} \\")
        for r in rows:
            if case.startswith("2d"):
                rho=1e-6*r["h"]**4
            elif case=="3d_hydrogen":
                rho=r["h"]**2/60
            else:
                rep=r.get("mesh_report",{})
                rho=r.get("rho_target",rep.get("rho_target",rep.get("r_start_mesh",r["patch_rmax"])))
            cells = [hlabel(r['h']), integer(r["n_elem"]), num(r["h_max"],4),
                     sci(rho), sci(r["Gamma_h"]), sci(r["d_h"]),
                     num(r["C_cert"],4), num(r["sigma_opt"],4)]
            lines.append(" & ".join(cells)+r" \\")
        if ci != len(cases)-1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (out/"tables/optimal_mesh_constants.tex").write_text("\n".join(lines)+"\n")


def save(fig, path):
    for ext in ("pdf","png","svg"):
        fig.savefig(path.with_suffix("."+ext),dpi=220)
    plt.close(fig)


def convergence(rows, case, out):
    if not rows:
        return
    h = np.array([r["h"] for r in rows])
    width_ref = np.array([r["U_neumann"][0]-r["L_opt"][0] for r in rows])
    width_best = np.array([r["U_neumann"][0]-r["L_best"][0] for r in rows])
    fig, axes = plt.subplots(1,2,figsize=(7.05,2.7))
    ax = axes[0]
    ax.loglog(h,width_ref,"o-",color="#254b82",label=r"$U_1^N-L_1^{\rm ref}$")
    ax.loglog(h,width_best,"s-",color="#bd5b23",label=r"$U_1^N-L_1^{\rm opt}$")
    ax.loglog(h,width_ref[-1]*(h/h[-1])**2,"--",color="#727272",lw=1,label=r"$O(h^2)$")
    ax.set_title("(a) Neumann enclosure widths",loc="left")
    ax.set_ylabel("Width")
    ax.legend(frameon=False,loc="best")
    ax = axes[1]
    for key,label,marker,color in (("Gamma_h",r"$\Gamma_h$","o","#254b82"),
                                   ("A_h",r"$A_h$","s","#bd5b23"),
                                   ("d_h",r"$d_h$","^","#3f7a55")):
        ax.loglog(h,[r[key] for r in rows],marker+"-",color=color,label=label)
    dlast = rows[-1]["d_h"]
    ax.loglog(h,dlast*(h/h[-1])**2,"--",color="#727272",lw=1,label=r"$O(h^2)$")
    ax.set_title("(b) Correction terms",loc="left")
    ax.set_ylabel("Magnitude")
    ax.legend(frameon=False,loc="upper left",ncol=2,columnspacing=1.,handlelength=2.)
    for ax in axes:
        ax.set_xlabel(r"Mesh parameter $h$")
        ax.set_xticks(sorted(h))
        ax.set_xticklabels([hlabel(x) for x in sorted(h)])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True,which="major",lw=.4,alpha=.3)
    fig.tight_layout(w_pad=1.3)
    save(fig,out/"figures"/f"convergence_{case}")


def mesh_figure(root, case, out):
    mesh = root/"optimal_rate/meshes"/f"{case}_h0.4"
    if not mesh.exists():
        return
    X = np.loadtxt(mesh/"nodes.dat")
    T = np.loadtxt(mesh/"elements.dat",dtype=int)-1
    report = json.loads((mesh/"mesh_report.json").read_text())
    plot_mesh_views(X,T,report,out/"figures"/f"mesh_optimal_{case}")


def plot_mesh_views(X,T,report,path,full_title="Full domain"):
    centre = np.array(report["centers"][0])
    rho = report["rho_target"]
    edges = np.vstack((T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]))
    seg = X[edges]
    fig, axes = plt.subplots(1,3,figsize=(7.05,2.25))
    views = [(np.zeros(2), None, full_title), (centre,.2,"Near a nucleus"),
             (centre,rho*1.15,"Shrinking Cartesian core")]
    for ax,(c,span,title) in zip(axes,views):
        if span is None:
            mask = np.ones(len(seg),dtype=bool)
            local = seg
        else:
            local = seg-c
            mask = np.any(np.max(np.abs(local),axis=2)<=span*1.1,axis=1)
        ax.add_collection(LineCollection(local[mask],colors="#4f6479",linewidths=.18,rasterized=True))
        if span is None:
            ax.set_xlim(-report["Lx"],report["Lx"])
            ax.set_ylim(-report["Ly"],report["Ly"])
            ax.plot(*np.array(report["centers"]).T,"*",ms=5,color="#aa302c")
        else:
            ax.set_xlim(-span,span);ax.set_ylim(-span,span)
            ax.plot(0,0,"*",ms=5,color="#aa302c")
            ax.ticklabel_format(axis="both",style="sci",scilimits=(-2,2),useMathText=True)
        ax.set_aspect("equal")
        ax.set_title(title,fontsize=9)
        ax.tick_params(labelsize=7)
    fig.tight_layout(w_pad=.8)
    save(fig,path)


def mesh_figure_3d(root,rows,out):
    if not rows:
        return
    r=rows[0]
    mesh=root/"optimal_rate/meshes"/r["mesh"]
    X=np.loadtxt(mesh/"nodes.dat")
    T=np.loadtxt(mesh/"elements.dat",dtype=int)-1
    # The z=0 symmetry plane consists of exact tetrahedral facets.
    F=np.concatenate([T[:,[1,2,3]],T[:,[0,2,3]],T[:,[0,1,3]],T[:,[0,1,2]]])
    mask=np.all(X[F,2]==0.,axis=1)
    F=np.unique(np.sort(F[mask],axis=1),axis=0)
    rep=json.loads((mesh/"mesh_report.json").read_text())
    report=dict(centers=[p[:2] for p in rep["centers"]],rho_target=rep["rho_target"],Lx=8.,Ly=6.)
    plot_mesh_views(X[:,:2],F,report,out/"figures/mesh_optimal_3d_h2plus",full_title=r"Mid-plane $z=0$")
    (out/"figures/mesh_optimal_3d_h2plus.json").write_text(json.dumps(dict(
        mesh=r["mesh"],h=r["h"],plane="z=0",slice_triangles=len(F),
        rho_target=rep["rho_target"],global_cap=rep["global_cap"]),indent=2)+"\n")


def summaries(cases,out):
    metrics = []
    for case,rows in cases.items():
        previous = None
        for r in rows:
            row = {"case":case,"h":r["h"],"mesh":r["mesh"],"source_json":r["_source_json"],
                   "h_max":r["h_max"],"n_elem":r["n_elem"],
                   "d_h":r["d_h"],"Gamma_h":r["Gamma_h"],
                   "width_ref":r["U_neumann"][0]-r["L_opt"][0],
                   "width_best":r["U_neumann"][0]-r["L_best"][0],
                   "L_best_1":r["L_best"][0],"U_N_1":r["U_neumann"][0],
                   "U_D_1":r["U_dirichlet"][0],"eps_best_1":r["eps_best"][0]}
            for metric in ("d_h","Gamma_h","width_ref","width_best"):
                row["EOC_"+metric] = None if previous is None else math.log(previous[metric]/row[metric])/math.log(previous["h"]/row["h"])
                row["EOC_hmax_"+metric] = None if previous is None else math.log(previous[metric]/row[metric])/math.log(previous["h_max"]/row["h_max"])
            metrics.append(row);previous=row
    if metrics:
        with (out/"paper_metrics.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
        (out/"paper_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")

    eigenrows=[]
    for case,rows in cases.items():
        for r in rows:
            for k in range(3):
                row={"case":case,"h":r["h"],"mesh":r["mesh"],"source_json":r["_source_json"],
                     "h_max":r["h_max"],"n_elem":r["n_elem"],
                     "k":k+1,"d_h":r["d_h"],"Gamma_h":r["Gamma_h"],
                     "C_reference":r["C_cert"],"sigma_reference":r["sigma_opt"],
                     "mu_reference":r["mu_minus_sigma_opt"][k],"L_reference":r["L_opt"][k],
                     "eps_selected":r["eps_best"][k],"C_selected":r["C_best"][k],
                     "sigma_selected":r["sigma_best"][k],"mu_selected":selected_mu(r,k),
                     "L_optimal":r["L_best"][k],"U_neumann":r["U_neumann"][k],
                     "U_dirichlet":r["U_dirichlet"][k]}
                eigenrows.append(row)
    if eigenrows:
        with (out/"paper_eigenvalues.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(eigenrows[0]));w.writeheader();w.writerows(eigenrows)
    prose=[]
    def bracket(lo,hi):
        # Printed intervals are rounded outwards, independent of signs.
        lo=math.floor(lo*1e6)/1e6;hi=math.ceil(hi*1e6)/1e6
        return f"[{lo:.6f},{hi:.6f}]"
    for case in ("2d_one_centre","2d_two_centre"):
        rows=cases.get(case,[])
        if not rows:
            continue
        r=rows[-1]
        title="one-centre" if case=="2d_one_centre" else "two-centre"
        text=(f"On the finest {title} mesh ($h={r['h']:g}$), the ground-state "
              "Neumann and whole-space enclosures are, respectively,\n"
              r"\[\lambda_1^{R,N}\in"
              +bracket(r["L_best"][0],r["U_neumann"][0])+r",\qquad\lambda_1\in"
              +bracket(r["L_best"][0],r["U_dirichlet"][0])+r".\]"+"\n")
        if case=="2d_two_centre" and r["U_dirichlet"][1]<-.3714:
            text+=(r"The confinement condition also holds for $k=2$, giving $\lambda_2\in"
                   +bracket(r["L_best"][1],r["U_dirichlet"][1])+"$. ")
        if r["U_neumann"][0]<r["L_best"][1]:
            text+=(r"Since $U_1^N<L_2^{\rm opt}$, the first Neumann eigenvalue is isolated. ")
        prose.append(text)
    if prose:
        prose.insert(0,r"\paragraph{Finest-mesh bounds} The following numerical enclosures are printed with their endpoints rounded outwards to six decimal places; the computations use floating-point arithmetic.")
        (out/"tables/optimal_summary.tex").write_text("\n\n".join(prose)+"\n")


def summary_3d(rows,out):
    if not rows:
        return
    r=rows[-1]
    lower=r["L_best"][0];un=r["U_neumann"][0];ud=r["U_dirichlet"][0]
    reference=-.5513
    if not lower<=reference<=ud:
        raise ValueError("The H2+ whole-space interval misses the cited reference value")
    if not ud<-.3162:
        raise ValueError("H2+ ground-state confinement check failed")
    bracket=lambda lo,hi:"["+endpoint(lo,True)+","+endpoint(hi,False)+"]"
    text=(r"\paragraph{Finest three-dimensional two-centre bounds} "
          f"On the finest capped mesh ($h={hlabel(r['h']).strip('$')}$), "
          "the numerical ground-state Neumann and whole-space enclosures are\n"
          r"\[\lambda_1^{R,N}\in"+bracket(lower,un)
          +r",\qquad\lambda_1\in"+bracket(lower,ud)+r".\]"+"\n"
          r"The condition $U_1<\sext(\Omega)=-0.3162$ verifies confinement. "
          r"The whole-space interval contains the reference value $-0.5513$; "
          r"that value is not used as a reference eigenvalue for the finite-box Neumann problem. "
          "The displayed endpoints are rounded outwards to six decimal places, "
          "and the computations use floating-point arithmetic. ")
    if r["U_dirichlet"][1]<-.3162:
        text+=(r"The confinement condition also holds for $k=2$, giving $\lambda_2\in"
               +bracket(r["L_best"][1],r["U_dirichlet"][1])+"$. ")
    if un<r["L_best"][1]:
        text+=r"Since $U_1^N<L_2^{\rm opt}$, the first Neumann eigenvalue is isolated. "
    (out/"tables/optimal_summary_3d.tex").write_text(text+"\n")


def convergence_table_3d(rows,out):
    if not rows:
        return
    data=[];previous=None
    for r in rows:
        d=dict(h=r["h"],hmax=r["h_max"],dh=r["d_h"],Gamma=r["Gamma_h"],
               Wref=r["U_neumann"][0]-r["L_opt"][0],
               Wopt=r["U_neumann"][0]-r["L_best"][0])
        for key in ("dh","Gamma","Wref","Wopt"):
            d["p_"+key]=None if previous is None else math.log(previous[key]/d[key])/math.log(previous["h"]/d["h"])
            d["pmax_"+key]=None if previous is None else math.log(previous[key]/d[key])/math.log(previous["hmax"]/d["hmax"])
        data.append(d);previous=d
    p=lambda x:"--" if x is None else f"{x:.3f}"
    lines=[r"\begin{table}[htbp]",r"\centering\footnotesize",r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Measured convergence on the capped 3D two-centre family. "
           r"$W_{\rm ref}=U_1^N-L_1^{\rm ref}$ and $W_{\rm opt}=U_1^N-L_1^{\rm opt}$ compare bounds for the same finite-box Neumann eigenvalue. "
           r"The orders $p_h$ and $p_{h_{\max}}$ use full-precision data and the actual ratios of nominal $h$ and measured $h_{\max}$, respectively.}",
           r"\label{tab:h2plus-optimal-convergence}",r"\begin{tabular}{crrrrrr}",r"\toprule",
           r"$h$ & $d_h$ & $p_h(d_h)$ & $p_{h_{\max}}(d_h)$ & $\Gamma_h$ & $p_h(\Gamma_h)$ & $p_{h_{\max}}(\Gamma_h)$ \\",r"\midrule"]
    for d in data:
        lines.append(" & ".join([hlabel(d["h"]),sci(d["dh"]),p(d["p_dh"]),p(d["pmax_dh"]),sci(d["Gamma"]),p(d["p_Gamma"]),p(d["pmax_Gamma"])])+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\par\medskip",r"\begin{tabular}{crrrrrr}",r"\toprule",
              r"$h$ & $W_{\rm ref}$ & $p_h(W_{\rm ref})$ & $p_{h_{\max}}(W_{\rm ref})$ & $W_{\rm opt}$ & $p_h(W_{\rm opt})$ & $p_{h_{\max}}(W_{\rm opt})$ \\",r"\midrule"]
    for d in data:
        lines.append(" & ".join([hlabel(d["h"]),sci(d["Wref"]),p(d["p_Wref"]),p(d["pmax_Wref"]),sci(d["Wopt"]),p(d["p_Wopt"]),p(d["pmax_Wopt"])])+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (out/"tables/optimal_convergence_3d.tex").write_text("\n".join(lines)+"\n")


def validate_assets(cases,out):
    filenames={"2d_one_centre":"one_centre_2d.tex","2d_two_centre":"two_centre_2d.tex",
               "3d_hydrogen":"hydrogen_3d.tex","3d_h2plus":"two_centre_3d.tex"}
    checks=[];n=0
    for case,rows in cases.items():
        if not rows:
            continue
        printed=[]
        for line in (out/"tables"/filenames[case]).read_text().splitlines():
            parts=line.removesuffix(r" \\").split(" & ")
            if len(parts)==8 and parts[1] in ("1","2","3"):
                printed.append(parts)
        if len(printed)!=3*len(rows):
            raise ValueError("Incorrect number of spectral rows: "+case)
        i=0
        for r in rows:
            if case=="3d_h2plus":
                assert "3d_h2plus_cap_h" in r["_source_json"]
                assert math.isclose(r["mesh_report"]["global_cap"],r["h"],rel_tol=1e-12)
                assert r["h_max"]<=r["h"]*(1+1e-10)
            for k in range(3):
                q=printed[i];i+=1;n+=1
                assert int(q[1])==k+1
                assert abs(float(q[2])-selected_mu(r,k))<=5.01e-7
                assert abs(float(q[3])-r["eps_best"][k])<1e-12
                for col,key,lower in ((4,"L_opt",True),(5,"L_best",True),
                                      (6,"U_neumann",False),(7,"U_dirichlet",False)):
                    value=float(q[col]);raw=r[key][k]
                    assert value<=raw+1e-12 if lower else value>=raw-1e-12
                    assert abs(value-raw)<1.000001e-6
                assert r["L_best"][k]>=r["L_opt"][k]-1e-11
                assert r["L_best"][k]<=r["U_neumann"][k]<=r["U_dirichlet"][k]
            checks.append(dict(case=case,h=r["h"],source_json=r["_source_json"],
                               all_three_rows_validated=True))
    report=dict(status="passed",spectral_rows=n,mesh_levels=len(checks),
                canonical_3d_family="global diameter cap; uncapped results excluded",checks=checks)
    with (out/"paper_metrics.csv").open() as f:
        metrics={(r["case"],float(r["h"])):r for r in csv.DictReader(f)}
    assert len(metrics)==len(checks)
    verified_rates=0
    for case,rows in cases.items():
        prev=None
        for r in rows:
            record=metrics[(case,r["h"])]
            values={"d_h":r["d_h"],"Gamma_h":r["Gamma_h"],
                    "width_ref":r["U_neumann"][0]-r["L_opt"][0],
                    "width_best":r["U_neumann"][0]-r["L_best"][0]}
            for key,value in values.items():
                assert math.isclose(float(record[key]),value,rel_tol=1e-13,abs_tol=1e-14)
                if prev is not None:
                    prev_value={"d_h":prev["d_h"],"Gamma_h":prev["Gamma_h"],
                                "width_ref":prev["U_neumann"][0]-prev["L_opt"][0],
                                "width_best":prev["U_neumann"][0]-prev["L_best"][0]}[key]
                    for denom,field in ((math.log(prev["h"]/r["h"]),"EOC_"+key),
                                        (math.log(prev["h_max"]/r["h_max"]),"EOC_hmax_"+key)):
                        value_rate=math.log(prev_value/value)/denom
                        assert math.isclose(float(record[field]),value_rate,rel_tol=1e-12,abs_tol=1e-13)
                        verified_rates+=1
            prev=r
    report["full_precision_EOC_fields_validated"]=verified_rates
    (out/"paper_assets_validation.json").write_text(json.dumps(report,indent=2)+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    out=args.out or args.root/"optimal_rate/paper"
    (out/"tables").mkdir(parents=True,exist_ok=True)
    (out/"figures").mkdir(parents=True,exist_ok=True)
    cases=read_cases(args.root)
    constants_table(cases,out)
    for case,rows in cases.items():
        if rows:
            spectral_table(rows,case,out);convergence(rows,case,out)
        if case.startswith("2d"):
            mesh_figure(args.root,case,out)
    summaries(cases,out)
    h2=cases.get("3d_h2plus",[])
    summary_3d(h2,out)
    convergence_table_3d(h2,out)
    mesh_figure_3d(args.root,h2,out)
    validate_assets(cases,out)
    print("Wrote paper tables, figures and metrics:", {k:len(v) for k,v in cases.items()},"->",out)


if __name__=="__main__":
    main()
