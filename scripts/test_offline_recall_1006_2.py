"""Tests for denominators, query costs, representative failure and label separation."""
import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).parent/"offline_recall_1006_2.py"
s=importlib.util.spec_from_file_location("offline",p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
x=np.array([[1,0],[.9,.1],[.8,.2],[.7,.3],[.6,.4],[0,1],[.1,.9],[.2,.8]],np.float32)
labels=np.array([0,0,1,1,-1,-1,2,2])
cams=np.array([0,0,1,1,2,2,0,1])
pids=np.array([1,2,1,1,1,9,9,9])
pool=m.Pool(x,labels,cams)
gold,info=m.gold_pairs(pool,pids)
assert gold=={(0,1):2,(0,3):1,(1,3):1,(2,4):1},gold
assert info["majority_tie_clusters"]==[0]
r,_,_=m.metrics(pool,pids,gold,np.array([[1,2]]))
assert r["weighted_recall"]==.4 and r["effective_weighted_recall"]==0
r,_,_=m.metrics(pool,pids,gold,np.array([[0,2],[0,3],[4,2]]))
assert r["queries"]==3 and r["covered_pairs"]==2 and r["weighted_recall"]==.6
assert r["effective_weighted_recall"]==.6 and r["M_pair_yield"]==1
assert len(m.dedup([(0,2),(2,0),(0,0)]))==1
queries=m.camera_queries(pool)
assert all(pool.cams[i]!=pool.cams[j] and pool.gid[i]!=pool.gid[j] for i,j in queries)
before=m.random_queries(pool,10,42)
m.gold_pairs(pool,pids[::-1])
after=m.random_queries(pool,10,42)
assert np.array_equal(before,after)
assert not hasattr(pool,"pids")
assert np.array_equal(m.old_queries(pool,1,rule=True),m.old_queries(pool,1,rule=True))
previous=0
for b in range(len(queries)+1):
    r,_,_=m.metrics(pool,pids,gold,queries[:b])
    assert previous<=r["weighted_recall"]<=1
    previous=r["weighted_recall"]
print("PASS: gold singleton edges, representative failures, duplicate parent costs, label isolation, camera eligibility, monotonic recall")
