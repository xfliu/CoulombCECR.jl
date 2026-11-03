
import scipy.io as sio, numpy as np, sys, glob, json
root = sys.argv[1]
def g(o, *path):
    for p in path:
        if o is None: return None
        o = getattr(o, p, None) if not isinstance(o, dict) else o.get(p)
    if o is None: return None
    a = np.asarray(o, dtype=float)
    return a.tolist() if a.size > 1 else float(a)
out = {}
for h in ["140","120","100","080","060","040"]:
    fn = f"{root}/case5_3d_hydrogen_cube/meshes_R6_ball/ball_R6_h{h}/cecr_bounds_ball_R6.mat"
    try: d = sio.loadmat(fn, squeeze_me=True, struct_as_record=False)
    except Exception as e: out[h] = str(e); continue
    out["ball_R6_h"+h] = dict(N_elem=len(d["cnst"].h_K), h_max=float(np.max(d["cnst"].h_K)), Gamma_h=g(d,"cnst","Gamma_h"), A_h=g(d,"cnst","A_h"),
        C_h=g(d,"cnst","Ch_PW"), d_h=g(d,"dh_om0"), eps_h=g(d,"eps_h"), C_eps_cert=g(d,"C_eps_cert"), sigma_opt=g(d,"sigma_opt"),
        mu_minus_sigma=g(d,"m2out","mu_h"), L_mu_sigma=g(d,"m2out","Lk_mu_sig"), L=g(d,"Lk_cert"), U=g(d,"lambda_ub"),
        L_opt=g(d,"Lk_cert_opt"), mu_minus_sigma_opt=g(d,"m2out_opt","mu_h"), L_mu_sigma_opt=g(d,"m2out_opt","Lk_mu_sig"))
b = f"{root}/case6_3d_h2plus_box/meshes_R6/box_R6_h100_opt/"
d = sio.loadmat(b+"cecr_bounds_sigma540.mat", squeeze_me=True, struct_as_record=False)
e = sio.loadmat(b+"cecr_bounds_opt_sigma.mat", squeeze_me=True, struct_as_record=False)
out["box_R6_h100_opt"] = dict(N_elem=len(d["cnst"].h_K), h_max=float(np.max(d["cnst"].h_K)), Gamma_h=g(d,"cnst","Gamma_h"), A_h=g(d,"cnst","A_h"),
        C_h=g(d,"cnst","Ch_PW"), d_h=g(d,"dh_om0"), eps_h=g(d,"eps_h"), C_eps_cert=g(e,"C_eps_cert"), sigma_opt=g(e,"sigma_opt"),
        mu_minus_sigma=g(d,"m2out","mu_h"), L_mu_sigma=g(d,"m2out","Lk_mu_sig"), L=g(d,"Lk_cert"), U=g(d,"m3out","lambda_ub"),
        L_opt=g(e,"Lk_cert_opt"), mu_minus_sigma_opt=g(e,"m2out_opt","mu_h"), L_mu_sigma_opt=g(e,"m2out_opt","Lk_mu_sig"))
print("JSON" + json.dumps(out))
