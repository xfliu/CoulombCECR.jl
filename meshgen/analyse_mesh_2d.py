
import sys, json, numpy as np, pandas as pd
def load(d):
    X = pd.read_csv(d+"/nodes.dat", sep=r"\s+", header=None).values.astype(float)
    T = pd.read_csv(d+"/elements.dat", sep=r"\s+", header=None).values.astype(np.int64)-1
    return X, T
def analyse(d, Lx, Ly, centers, detail=False):
    X, T = load(d)
    P = X[T]
    e = [np.linalg.norm(P[:,i]-P[:,j],axis=1) for i,j in ((0,1),(0,2),(1,2))]
    hK = np.maximum(np.maximum(e[0],e[1]),e[2])
    a = P[:,1]-P[:,0]; b = P[:,2]-P[:,0]
    sa = 0.5*(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    E = np.sort(np.concatenate([T[:,[0,1]],T[:,[1,2]],T[:,[0,2]]]),axis=1)
    key = E[:,0]*(len(X)+1)+E[:,1]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    single = cnt==1
    ue = np.stack([u//(len(X)+1), u%(len(X)+1)],1)
    onb = lambda p: (np.abs(np.abs(p[:,0])-Lx)<1e-9)|(np.abs(np.abs(p[:,1])-Ly)<1e-9)
    se = ue[single]
    bad = ~(onb(X[se[:,0]]) & onb(X[se[:,1]]))
    used = np.zeros(len(X),bool); used[T.ravel()]=True
    r = {"mesh":d.rstrip('/').split('/')[-1], "n_nodes":int(len(X)), "n_tris":int(len(T)),
         "h_max":float(hK.max()), "hK_min":float(hK.min()),
         "bbox":[float(X[:,0].min()),float(X[:,0].max()),float(X[:,1].min()),float(X[:,1].max())],
         "area_sum":float(np.abs(sa).sum()), "area_exact":4*Lx*Ly,
         "n_neg_orient":int((sa<0).sum()), "min_area":float(np.abs(sa).min()),
         "n_edges_gt2":int((cnt>2).sum()), "n_single":int(single.sum()),
         "interior_free_edges":int(bad.sum()), "unused_nodes":int((~used).sum())}
    # dist of nucleus vertices
    r["centre_is_node"]=[float(np.min(np.linalg.norm(X-np.array(c),axis=1))) for c in centers]
    # innermost ring radius: min nonzero distance of nodes to centres
    dd = np.min(np.stack([np.linalg.norm(X-np.array(c),axis=1) for c in centers]),0)
    r["r_first_ring"]=float(np.min(dd[dd>1e-14]))
    if detail and bad.any():
        be = se[bad]
        pts = X[be].reshape(-1,2)
        r["bad_edge_xy_range"]=[float(pts[:,0].min()),float(pts[:,0].max()),float(pts[:,1].min()),float(pts[:,1].max())]
        r["bad_edge_samples"]=X[be[:8]].tolist()
        r["bad_edge_len"]=[float(np.linalg.norm(X[i]-X[j])) for i,j in be[:8]]
        r["bad_nodes_dist_to_boundary"]=sorted(set(float(min(Lx-abs(p[0]),Ly-abs(p[1]))) for p in pts))[:10]
    return r
if __name__=="__main__":
    d=sys.argv[1]; Lx=float(sys.argv[2]); Ly=float(sys.argv[3])
    cs=[[float(v) for v in s.split(',')] for s in sys.argv[4].split(';')]
    print(json.dumps(analyse(d,Lx,Ly,cs,detail=True)))
