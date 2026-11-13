#!/usr/bin/env python3
"""Deterministic shape-regular, mirror-symmetric H2+ cuboid-ring meshes.

An octant core around (2,0,0), of covering radius beta*h**2, is Cartesian.
Homothetic cuboid shells use the same transverse subdivisions. Convex
hexahedra are tetrahedralized by a globally consistent pulling rule; this
avoids Delaunay slivers and hanging facets. Reflection gives the full box.
Only NumPy is required. No fixed radius or coordinate-deduplication floor.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np


QUADS=((0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))


def pulling_tets(v):
    """Pull the smallest global vertex; all shared quad diagonals agree."""
    pivot=min(v)
    out=[]
    for face in QUADS:
        q=[v[i] for i in face]
        if pivot in q:
            continue
        k=q.index(min(q)); q=q[k:]+q[:k]
        out.extend(((pivot,q[0],q[1],q[2]),(pivot,q[0],q[2],q[3])))
    assert len(out)==6
    return out


def octant(h,beta,density):
    n=max(2,math.ceil(density/h));rho=beta*h*h
    xs=np.r_[np.linspace(0.,2.,n+1)[:-1],np.linspace(2.,8.,n+1)]
    ys=np.linspace(0.,6.,n+1);zs=ys.copy()
    a=np.array([2.,0.,0.]);rmax=math.sqrt(6**2+6**2+6**2)
    s0=rho/rmax
    if not 0<s0<1:
        raise ValueError("positive core radius smaller than the box is required")
    X=[];T=[]
    def newnode(i,j,k,s):
        p=np.array([xs[i],ys[j],zs[k]])
        q=p if s==1. else a+s*(p-a)
        X.append(tuple(q));return len(X)-1
    core={}
    for k in range(n+1):
        for j in range(n+1):
            for i in range(2*n+1):
                core[i,j,k]=newnode(i,j,k,s0)
    for k in range(n):
        for j in range(n):
            for i in range(2*n):
                v=[core[i,j,k],core[i+1,j,k],core[i+1,j+1,k],core[i,j+1,k],
                   core[i,j,k+1],core[i+1,j,k+1],core[i+1,j+1,k+1],core[i,j+1,k+1]]
                T.extend(pulling_tets(v))
    ncore=len(T)
    # Active exterior faces of the octant box. Faces y=0,z=0 contain the
    # nucleus and sweep no volume, so they are only shell boundary faces.
    quads=[]
    for i in (0,2*n):
        for j in range(n):
            for k in range(n):
                quads.append(((i,j,k),(i,j+1,k),(i,j+1,k+1),(i,j,k+1)))
    for i in range(2*n):
        for k in range(n):
            quads.append(((i,n,k),(i+1,n,k),(i+1,n,k+1),(i,n,k+1)))
    for i in range(2*n):
        for j in range(n):
            quads.append(((i,j,n),(i+1,j,n),(i+1,j+1,n),(i,j+1,n)))
    keys=sorted(set(p for q in quads for p in q))
    prev={key:core[key] for key in keys}
    nl=math.ceil(math.log(1/s0)/math.log1p(1/n))
    for layer in range(1,nl+1):
        s=1. if layer==nl else math.exp(math.log(s0)*(1-layer/nl))
        ring={key:newnode(*key,s) for key in keys}
        for q in quads:
            T.extend(pulling_tets([prev[k] for k in q]+[ring[k] for k in q]))
        prev=ring
    coreflag=np.arange(len(T))<ncore
    return np.asarray(X),np.asarray(T,dtype=np.int64),coreflag,dict(n=n,radial_layers=nl,rho_target=rho)


def reflect(X,T,core):
    octant_ne=len(T);octant_nv=len(X)
    for axis in range(3):
        nv=len(X);mask=X[:,axis]!=0.
        ids=np.arange(nv);ids[mask]=nv+np.arange(mask.sum())
        Y=X[mask].copy();Y[:,axis]=-Y[:,axis]
        X=np.vstack((X,Y));T=np.vstack((T,ids[T]));core=np.r_[core,core]
    atoms=np.array([[-2.,0.,0.],[2.,0.,0.]])
    first=[int(np.flatnonzero(np.all(X==a,axis=1))[0]) for a in atoms]
    rest=np.setdiff1d(np.arange(len(X)),first)
    perm=np.r_[first,rest];inv=np.empty_like(perm);inv[perm]=np.arange(len(perm))
    return X[perm],inv[T],core,dict(n_nodes=octant_nv,n_elem=octant_ne)


def measures(X,T):
    P=X[T]
    six=np.einsum('ij,ij->i',P[:,1]-P[:,0],np.cross(P[:,2]-P[:,0],P[:,3]-P[:,0]))
    h=np.max([np.linalg.norm(P[:,i]-P[:,j],axis=1) for i in range(4) for j in range(i+1,4)],axis=0)
    return six/6,h


def census(X,T):
    F=np.sort(np.concatenate([T[:,[1,2,3]],T[:,[0,2,3]],T[:,[0,1,3]],T[:,[0,1,2]]]),axis=1)
    u,c=np.unique(F,axis=0,return_counts=True);s=u[c==1]
    on=np.abs(np.abs(X)-[8.,6.,6.])<1e-10
    good=(on[s[:,0]]&on[s[:,1]]&on[s[:,2]]).any(axis=1)
    return dict(n_facets=len(u),boundary_facets=len(s),interior_free_facets=int((~good).sum()),
                nonmanifold_facets=int((c>2).sum()))


def quality_and_grading(X,T,core,h):
    qmin=math.inf;dihedral=math.inf;theta=0.;core_rmax=0.;off_gradient=0.
    atoms=np.array([[-2.,0.,0.],[2.,0.,0.]])
    for start in range(0,len(T),100000):
        stop=min(start+100000,len(T));P=X[T[start:stop]];iscore=core[start:stop]
        vol,hk=measures(X,T[start:stop])
        qmin=min(qmin,float((6*math.sqrt(2)*vol/hk**3).min()))
        N=[]
        for opposite,face in enumerate(((1,2,3),(0,2,3),(0,1,3),(0,1,2))):
            i,j,k=face;n=np.cross(P[:,j]-P[:,i],P[:,k]-P[:,i])
            sign=np.sign(np.einsum('ij,ij->i',n,P[:,opposite]-P[:,i]))
            n=-n*sign[:,None];N.append(n/np.linalg.norm(n,axis=1)[:,None])
        for i in range(4):
            for j in range(i+1,4):
                angle=np.degrees(np.arccos(np.clip(-np.einsum('ij,ij->i',N[i],N[j]),-1.,1.)))
                dihedral=min(dihedral,float(angle.min()))
        # Bounding-box distance is a LOWER bound for point-to-tet distance,
        # hence gives conservative G1 and gradient-oscillation upper bounds.
        lo=P.min(axis=1);hi=P.max(axis=1);d=[]
        for a in atoms:
            d.append(np.linalg.norm(np.maximum(np.maximum(lo-a,a-hi),0.),axis=1))
        d=np.array(d);reg=~iscore
        if reg.any():
            assert np.all(d[:,reg]>0.)
            theta=max(theta,float((hk[reg]/(h*d[:,reg].min(axis=0))).max()))
            off_gradient=max(off_gradient,float((hk[reg]**2/2*(1/d[:,reg]**2).sum(axis=0)).max()))
        if iscore.any():
            # Cores are disjoint and each is associated to its nearest centre.
            far=np.array([np.linalg.norm(P[iscore]-a,axis=2).max(axis=1) for a in atoms])
            core_rmax=max(core_rmax,float(far.min(axis=0).max()))
    return dict(min_dihedral_deg=dihedral,min_shape_quality=qmin,
                theta_off_core_upper=theta,core_rmax=core_rmax,
                d_off_gradient_upper=off_gradient,
                distance_bound="axis-aligned tetrahedron bounding boxes (conservative)")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--h',type=float,required=True)
    ap.add_argument('--beta',type=float,default=.0015625)
    ap.add_argument('--density',type=float,default=4.)
    ap.add_argument('--global-cap',type=float,default=None,
                    help='optional maximum edge length as a multiple of nominal h (default off)')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();start=time.monotonic()
    if min(args.h,args.beta,args.density)<=0:
        raise ValueError('h,beta,density must be positive')
    X,T,core,meta=octant(args.h,args.beta,args.density)
    capmeta={}
    if args.global_cap is not None:
        from refine_cap_3d import refine_cap
        X,T,core,capmeta=refine_cap(X,T,core,args.h*args.global_cap)
    X,T,core,oct=reflect(X,T,core)
    vol,hk=measures(X,T);neg=vol<0.
    if neg.any():
        T[neg,2],T[neg,3]=T[neg,3].copy(),T[neg,2].copy()
        vol=-np.minimum(vol,0.)+np.maximum(vol,0.)
    assert np.all(vol>0.)
    rep=dict(generator="homothetic_cuboid_rings_3d",case="3d_h2plus",h=args.h,beta=args.beta,gamma_p=2,
             density=args.density,n_nodes=len(X),n_elem=len(T),n_tets=len(T),n_core=int(core.sum()),
             h_max=float(hk.max()),hmax=float(hk.max()),hK_min=float(hk.min()),
             min_volume=float(vol.min()),volume=float(vol.sum()),volume_defect=float(vol.sum()-2304.),
             centers=[[-2.,0.,0.],[2.,0.,0.]],charges=[1.,1.],L=[8.,6.,6.],
             octant=oct,**meta,**capmeta,**census(X,T),**quality_and_grading(X,T,core,args.h))
    # Independent full-core perturbation competitor. The paper's local
    # H1(B(a,2))->L6(B(a,1)) constant applies; the two support balls touch
    # only on a measure-zero set. Minkowski and Jensen handle both nuclei.
    p=1.5;rho=max(rep['rho_target'],rep['core_rmax'])
    if rho<1.:
        K3=(3*math.pi)**(-.5)*(4/math.sqrt(math.pi))**(1/3)
        S=math.sqrt(2)*K3
        own=(4*math.pi/(3-p)*rho**(3-p))**(1/p)
        smooth=(4*math.pi/3*rho**3)**(1/p)/(4-rho)
        rep['S6_local']=S
        rep['d_patch_local_upper']=2*S*S*(own+smooth)
        rep['d_h_geometry_upper']=rep['d_off_gradient_upper']+rep['d_patch_local_upper']
        rep['d_h_geometry_upper_over_h2']=rep['d_h_geometry_upper']/args.h**2
        rep['geometric_patch_formula']='2*S6_local^2*((8*pi/3*rho^1.5)^(2/3)+(4*pi/3*rho^3)^(2/3)/(4-rho))'
    assert rep['interior_free_facets']==0 and rep['nonmanifold_facets']==0
    assert abs(rep['volume_defect'])<1e-8
    assert abs(rep['core_rmax']/rep['rho_target']-1)<1e-9
    rep['reflection_symmetry']={f'axis{i+1}':dict(nodes_without_mirror=0,elements_without_mirror=0,
        verification="exact reflected octant construction") for i in range(3)}
    args.out.mkdir(parents=True,exist_ok=True)
    np.savetxt(args.out/'nodes.dat',X,fmt='%.17e')
    np.savetxt(args.out/'elements.dat',T+1,fmt='%d')
    np.savetxt(args.out/'core_elements.dat',np.flatnonzero(core)+1,fmt='%d')
    rep['elapsed_s']=time.monotonic()-start
    (args.out/'mesh_report.json').write_text(json.dumps(rep,indent=2)+'\n')
    (args.out/'info.txt').write_text('\n'.join(f'{k} = {v}' for k,v in rep.items())+'\n')
    print(json.dumps(rep),flush=True)


if __name__=='__main__':
    main()
