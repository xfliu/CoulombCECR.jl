
import sys, json, numpy as np, pandas as pd
def load(d):
    X = pd.read_csv(d+"/nodes.dat", sep=r"\s+", header=None).values.astype(float)
    T = pd.read_csv(d+"/elements.dat", sep=r"\s+", header=None).values.astype(np.int64)
    return X, T
a, b = sys.argv[1], sys.argv[2]
X1,T1 = load(a); X2,T2 = load(b)
r = {"orig": a.split('/')[-1], "regen": b.split('/')[-1], "nodes": [len(X1), len(X2)], "tris": [len(T1), len(T2)]}
if X1.shape == X2.shape:
    r["max_node_diff"] = float(np.abs(X1-X2).max()); r["elements_identical"] = bool(np.array_equal(T1, T2))
print(json.dumps(r))
