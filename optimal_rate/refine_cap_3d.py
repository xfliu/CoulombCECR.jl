#!/usr/bin/env python3
"""Conforming global longest-edge bisection of the octant's outer mesh.

Only tetrahedra containing an overlong edge are tracked. All tetrahedra
incident to a chosen edge are split, so no hanging node or face is created.
Near-core elements are unchanged. The caller reflects the refined octant.
"""
import heapq
import math
import time
import numpy as np

EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))


def refine_cap(X,T,core,cap):
    if not cap>0:
        raise ValueError('cap must be positive')
    start=time.monotonic();cap2=cap*cap
    P=X[T]
    max2=np.maximum.reduce([np.einsum('ij,ij->i',P[:,i]-P[:,j],P[:,i]-P[:,j]) for i,j in EDGES])
    bad=max2>cap2*(1+2e-14)
    untouched=T[~bad];untouched_core=core[~bad]
    coords=[tuple(p) for p in X.tolist()]
    active={};incident={};heap=[];finished=[];finished_core=[];counter=0;splits=0;maxdepth=0

    def longedges(v):
        out=[]
        for i,j in EDGES:
            u,w=sorted((v[i],v[j]));a=coords[u];b=coords[w]
            length2=sum((a[k]-b[k])**2 for k in range(3))
            if length2>cap2*(1+2e-14):
                out.append(((u,w),length2))
        return out

    def insert(v,iscore=False,depth=0):
        nonlocal counter,maxdepth
        maxdepth=max(maxdepth,depth)
        le=longedges(v)
        if not le:
            finished.append(v);finished_core.append(iscore);return
        idx=counter;counter+=1;active[idx]=(v,iscore,[e for e,l in le],depth)
        for e,length2 in le:
            if e not in incident:
                incident[e]=set();heapq.heappush(heap,(-length2,*e))
            incident[e].add(idx)

    for idx in np.flatnonzero(bad):
        insert(tuple(int(i) for i in T[idx]),bool(core[idx]))
    while heap:
        _,u,v=heapq.heappop(heap);edge=(u,v)
        if edge not in incident:
            continue
        parents=list(incident[edge])
        mid=len(coords);coords.append(tuple((coords[u][i]+coords[v][i])/2 for i in range(3)))
        children=[]
        for idx in parents:
            tet,iscore,le,depth=active.pop(idx)
            for e in le:
                incident[e].remove(idx)
                if not incident[e]:
                    del incident[e]
            assert u in tet and v in tet
            children.extend(((tuple(mid if x==v else x for x in tet),iscore,depth+1),
                             (tuple(mid if x==u else x for x in tet),iscore,depth+1)))
        for tet,iscore,depth in children:
            insert(tet,iscore,depth)
        splits+=1
    assert not active and not incident
    tail=np.asarray(finished,dtype=np.int64).reshape((-1,4))
    Tout=np.vstack((untouched,tail))
    Cout=np.r_[untouched_core,np.asarray(finished_core,dtype=bool)]
    Xout=np.asarray(coords)
    assert int(Cout.sum())==int(core.sum()),'core should not be split for these parameters'
    report=dict(global_cap=cap,cap_octant_elements_before=len(T),cap_octant_elements_after=len(Tout),
                cap_octant_bad_original=int(bad.sum()),cap_new_octant_nodes=len(Xout)-len(X),
                cap_edge_bisections=splits,cap_elapsed_s=time.monotonic()-start,
                cap_core_unchanged=True,cap_max_bisection_depth=maxdepth)
    return Xout,Tout,Cout,report
