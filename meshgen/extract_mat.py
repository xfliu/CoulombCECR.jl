
import scipy.io as sio, numpy as np, json, os, sys
R=sys.argv[1]
out={}
def g(s, *path):
    for p in path:
        s = getattr(s, p) if hasattr(s, p) else s[p]
    a=np.asarray(s); return a.tolist() if a.size>1 else float(a)
for case, meshes in [("case7_2d_hydrogen_square",["square_h040_opt","square_h020_opt","square_h010_opt"]),
                     ("case8_2d_h2plus_box",["box_h040_opt","box_h020_opt"])]:
    for m in meshes:
        d=os.path.join(R,case,"meshes",m); r={}
        try:
            p=sio.loadmat(d+"/paper_results.mat",squeeze_me=True,struct_as_record=False)["paper"]
            for k in ["n_tris","h_max","sigma","epsilon","kappa_sig","C6","C_h_PW","Gamma_h","A_h","eps_h_A","eps_h_comb","B_patch","sup_off","eps_h_used","mu_h","Lk_mu_sig","Lk_cert","lambda_ub"]:
                try: r["paper_results."+k]=g(p,k)
                except Exception as e: pass
        except Exception as e: r["paper_results_err"]=str(e)
        try:
            q=sio.loadmat(d+"/cecr_bounds_opt_patch.mat",squeeze_me=True,struct_as_record=False)
            r["opt_patch.dh_om0"]=g(q["dh_om0"]); r["opt_patch.eps_h_om0_kap"]=g(q["eps_h_om0_kap"])
            r["opt_patch.C_eps"]=g(q["C_eps_val"]); r["opt_patch.mu_h_sig"]=g(q["m2out"],"mu_h_sig")
            r["opt_patch.Lk_mu_sig"]=g(q["m2out"],"Lk_mu_sig"); r["opt_patch.Lk"]=g(q["m6out_patch"],"Lk")
            r["opt_patch.lambda_ub"]=g(q["m3out"],"lambda_ub")
        except Exception as e: r["opt_patch_err"]=str(e)
        try:
            q=sio.loadmat(d+"/cecr_bounds_opt_sigma.mat",squeeze_me=True,struct_as_record=False)
            r["opt_sigma.C_eps_cert"]=g(q["C_eps_cert"]); r["opt_sigma.sigma_opt"]=g(q["sigma_opt"])
            r["opt_sigma.eps_for_opt"]=g(q["eps_for_opt"]); r["opt_sigma.mu_h_sig"]=g(q["m2out_opt"],"mu_h_sig")
            r["opt_sigma.Lk_mu_sig"]=g(q["m2out_opt"],"Lk_mu_sig"); r["opt_sigma.Lk"]=g(q["m6out_opt"],"Lk")
            r["opt_sigma.Gamma_h"]=g(q["cnst_opt"],"Gamma_h"); r["opt_sigma.Ch"]=g(q["cnst_opt"],"Ch_PW")
        except Exception as e: r["opt_sigma_err"]=str(e)
        try:
            q=sio.loadmat(d+"/cecr_bounds_sigma1250.mat",squeeze_me=True,struct_as_record=False)
            r["s1250.eps_h"]=g(q["m5out"],"eps_h"); r["s1250.Lk"]=g(q["m6out"],"Lk"); r["s1250.Sq0"]=g(q["c"],"Sq0")
        except Exception as e: r["s1250_err"]=str(e)
        info={}
        for line in open(d+"/info.txt"):
            if "=" in line:
                k,v=line.split("=",1); info[k.strip()]=v.strip()
        r["info"]=info
        out[m]=r
print(json.dumps(out))
