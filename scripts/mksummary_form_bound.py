import json, csv, os
REL="/mnt/ssd2T-2/xfliu/CoulombCECR_release/results/"
NEW="/mnt/ssd2T-2/xfliu/CoulombCECR_trackA/results_new/"
runs=[("2d_one_centre","square_h040_opt","2d"),("2d_one_centre","square_h020_fixed","2d"),
      ("2d_one_centre","square_h010_opt","2d"),("2d_two_centre","box_h040_opt","2d"),
      ("2d_two_centre","box_h020_fixed","2d"),
      ("3d_hydrogen","ball_R6_h140_conf","3d"),("3d_hydrogen","ball_R6_h120_conf","3d"),
      ("3d_hydrogen","ball_R6_h100_conf","3d"),("3d_hydrogen","ball_R6_h080_conf","3d"),
      ("3d_h2plus","box_R6_h100_opt","3d_analytic_only")]
cols=["case","mesh","k","C_cert_old","C_cert_old_how","C_cert_new","C_cert_new_how","C_cert_analytic_new",
      "d_h","L_opt_old","L_opt_new","eps_best","C_best","C_best_how","sigma_best","L_best","L_best_pred_minus_exact",
      "L1_trivial_radial","U_dirichlet","U_neumann","wall_s_new"]
rows=[]; missing=[]
for case,mesh,rd in runs:
    fo=f"{REL}{rd}/{case}_{mesh}.json"; fn=f"{NEW}{case}_{mesh}.json"
    if not os.path.exists(fn): missing.append(fn); continue
    O=json.load(open(fo)); N=json.load(open(fn))
    for k in range(len(N["L_opt"])):
        rows.append(dict(case=case,mesh=mesh,k=k+1,C_cert_old=O["C_cert"],C_cert_old_how=O["C_cert_how"],
            C_cert_new=N["C_cert"],C_cert_new_how=N["C_cert_how"],C_cert_analytic_new=N["C_cert_analytic"],
            d_h=N["d_h"],L_opt_old=O["L_opt"][k],L_opt_new=N["L_opt"][k],eps_best=N["eps_best"][k],
            C_best=N["C_best"][k],C_best_how=(N["C_best_how"][k] if "C_best_how" in N else ("radial:direct" if N["C_cert_how"]=="radial:direct" else "?")),sigma_best=N["sigma_best"][k],L_best=N["L_best"][k],
            L_best_pred_minus_exact=N["L_best_pred_minus_exact"][k],
            L1_trivial_radial=N["L1_trivial_radial"] if k==0 else "",
            U_dirichlet=N["U_dirichlet"][k],U_neumann=N["U_neumann"][k] if "U_neumann" in N else "",
            wall_s_new=N["wall_s"]))
with open(NEW+"form_bound_summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k:(f"{v:.6g}" if isinstance(v,float) else v) for k,v in r.items()})
print("rows",len(rows),"missing",missing)
for r in rows:
    print(r["case"][:6],r["mesh"],r["k"],"C %.4f->%.4f"%(r["C_cert_old"],r["C_cert_new"]),"Lopt %.4f->%.4f"%(r["L_opt_old"],r["L_opt_new"]),
          "best eps %.3f L %.4f"%(r["eps_best"],r["L_best"]),"U %.4f"%r["U_dirichlet"], r["U_neumann"] and "UN %.4f"%r["U_neumann"], r["L1_trivial_radial"] and "triv %.4f"%r["L1_trivial_radial"])
