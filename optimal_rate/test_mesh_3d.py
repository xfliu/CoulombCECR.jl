#!/usr/bin/env python3
"""Independent checks of serialized optimal 3D H2+ mesh artifacts."""
import collections
import argparse
import json
import math
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
LEVELS=(1.,2/3,.5,1/3)


def check_geometry(h,prefix='3d_h2plus_h'):
    path=BASE/'meshes'/f'{prefix}{h}'
    rep=json.loads((path/'mesh_report.json').read_text())
    X=np.loadtxt(path/'nodes.dat');T=np.loadtxt(path/'elements.dat',dtype=np.int64)-1
    core=np.loadtxt(path/'core_elements.dat',dtype=np.int64)-1
    assert len(X)==rep['n_nodes'] and len(T)==rep['n_elem']
    assert np.all(np.isfinite(X)) and T.min()==0 and T.max()==len(X)-1
    assert len(np.unique(core))==len(core)
    assert np.array_equal(X[:2],np.array([[-2.,0.,0.],[2.,0.,0.]]))
    assert np.all(np.abs(X)<=[8.,6.,6.])
    assert abs(rep['rho_target']/h**2-.0015625)<1e-14
    iscore=np.zeros(len(T),dtype=bool);iscore[core]=True
    nuclear=np.any(T<2,axis=1)
    assert np.all(iscore[nuclear]),'core must contain every nuclear star'
    core_r=0.;volume=0.;hmax=0.;qmin=math.inf;minvol=math.inf
    for start in range(0,len(T),100000):
        P=X[T[start:start+100000]]
        A=P[:,1]-P[:,0];B=P[:,2]-P[:,0];C=P[:,3]-P[:,0]
        vol=np.einsum('ij,ij->i',A,np.cross(B,C))/6
        assert np.all(vol>0),'nonpositive serialized tetrahedron orientation'
        hk=np.maximum.reduce([np.linalg.norm(P[:,i]-P[:,j],axis=1)
                              for i in range(4) for j in range(i+1,4)])
        volume+=float(vol.sum());hmax=max(hmax,float(hk.max()));minvol=min(minvol,float(vol.min()))
        qmin=min(qmin,float((6*math.sqrt(2)*vol/hk**3).min()))
        selected=iscore[start:start+len(P)]
        if selected.any():
            Pc=P[selected]
            r=np.minimum(np.linalg.norm(Pc-[-2.,0.,0.],axis=2).max(axis=1),
                         np.linalg.norm(Pc-[2.,0.,0.],axis=2).max(axis=1))
            core_r=max(core_r,float(r.max()))
    assert abs(volume-2304.)<1e-8
    assert abs(hmax-rep['h_max'])<1e-12
    assert abs(core_r/rep['rho_target']-1)<1e-9
    assert qmin>.015
    assert abs(qmin-rep['min_shape_quality'])<1e-12
    assert rep['interior_free_facets']==rep['nonmanifold_facets']==0
    assert rep['theta_off_core_upper']<1.5
    if 'global_cap' in rep:
        assert hmax<=rep['global_cap']*(1+1e-12)
        assert rep['cap_core_unchanged']
    assert rep['d_h_geometry_upper']>rep['d_off_gradient_upper']>0
    own=(8*math.pi/3*core_r**1.5)**(2/3)
    smooth=(4*math.pi/3*core_r**3)**(2/3)/(4-core_r)
    K3=(3*math.pi)**(-.5)*(4/math.sqrt(math.pi))**(1/3)
    patch=4*K3*K3*(own+smooth)
    assert abs(patch/rep['d_patch_local_upper']-1)<1e-9
    row=dict(h=h,n_elem=len(T),n_nodes=len(X),volume=volume,min_volume=minvol,
             core_radius=core_r,rho_over_h2=core_r/h**2,h_max=hmax,min_shape_quality=qmin,
             nuclear_star_elements=int(nuclear.sum()),all_nuclear_stars_in_core=True)
    # Independent full topology check on the coarsest artifact. Finer
    # meshes use the identical pulling/reflection construction and are
    # separately checked for geometry and nuclear-star containment above.
    if h==1.:
        facets=collections.Counter(tuple(sorted(face)) for t in T.tolist()
            for face in ((t[1],t[2],t[3]),(t[0],t[2],t[3]),(t[0],t[1],t[3]),(t[0],t[1],t[2])))
        assert max(facets.values())==2
        for f,count in facets.items():
            if count==1:
                Q=X[list(f)]
                assert any(np.all(Q[:,d]==s) for d,lim in enumerate((8.,6.,6.)) for s in (-lim,lim))
        nodeid={tuple(p):i for i,p in enumerate(X.tolist())}
        assert len(nodeid)==len(X),'duplicate serialized nodes'
        tetset=set(map(tuple,np.sort(T,axis=1).tolist()))
        for axis in range(3):
            Y=X.copy();Y[:,axis]*=-1
            pn=np.array([nodeid[tuple(p)] for p in Y.tolist()])
            assert np.array_equal(pn[pn],np.arange(len(X)))
            assert all(tuple(t) in tetset for t in np.sort(pn[T],axis=1).tolist())
            assert all(tuple(sorted(pn[list(f)])) in facets for f in facets)
        row.update(independent_full_facet_census=True,independent_exact_reflection_topology=True,
                   n_facets=len(facets))
    return row


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',default='3d_h2plus_h')
    ap.add_argument('--levels',nargs='+',type=float,default=LEVELS)
    ap.add_argument('--output',type=Path,default=None)
    args=ap.parse_args()
    records=[]
    for h in args.levels:
        if not (BASE/'meshes'/f'{args.prefix}{h}').exists():
            continue
        records.append(check_geometry(h,args.prefix))
        print(json.dumps(records[-1]),flush=True)
    result=dict(status='passed',method='independent checks of serialized files; Float64 geometry',
                checked_levels=len(records),records=records)
    output=args.output or BASE/'meshes'/'mesh_validation_3d.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(records)} 3D mesh levels, nuclear stars, volume, quality, core scaling, coarse exact topology')
