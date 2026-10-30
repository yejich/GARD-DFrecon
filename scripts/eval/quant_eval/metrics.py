"""Explicit geometry metrics. Distances are unsquared, similarities per group."""
import numpy as np
from scipy.spatial import cKDTree
from itertools import combinations

def centers(e):return -np.einsum('vji,vj->vi',e[:,:3,:3],e[:,:3,3])
def fit_sim(p,q,robust=False):
 p=np.asarray(p,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
 good=np.isfinite(p).all(-1)&np.isfinite(q).all(-1);p=p[good];q=q[good]
 if len(p)<4:raise ValueError('Not enough alignment correspondences')
 ids=np.linspace(0,len(p)-1,min(len(p),20000),dtype=int);p=p[ids];q=q[ids];sel=np.ones(len(p),bool)
 for _ in range(4 if robust else 1):
  x=p[sel];y=q[sel];a=x.mean(0);b=y.mean(0);xc=x-a;yc=y-b
  u,d,vt=np.linalg.svd(yc.T@xc/len(x));sign=np.ones(3);sign[-1]=np.linalg.det(u@vt);r=(u*sign)@vt
  var=np.mean(np.sum(xc*xc,axis=-1))
  if var<1e-14:raise ValueError('Degenerate alignment')
  s=float(np.sum(d*sign)/var);t=b-s*r@a
  residual=np.linalg.norm(s*p@r.T+t-q,axis=-1)
  sel=residual<=np.quantile(residual,.8)
 return s,r,t,dict(correspondences=len(p),rmse=float(np.sqrt(np.mean(residual**2))),median=float(np.median(residual)))
def transform(p,sim):s,r,t=sim[:3];return s*np.asarray(p)@r.T+t

def pose_metrics(pred,ref):
 pred=np.asarray(pred,dtype=np.float64).copy();ref=np.asarray(ref,dtype=np.float64).copy()
 defects={}
 for name,arr in [('prediction',pred),('reference',ref)]:
  r=arr[:,:3,:3];defects[name]=float(np.max(abs(r@r.transpose(0,2,1)-np.eye(3))))
  u,_,vt=np.linalg.svd(r);sign=np.ones((len(r),3));sign[:,-1]=np.linalg.det(u@vt);arr[:,:3,:3]=(u*sign[:,None,:])@vt
 rows=[];base=np.median([np.linalg.norm(centers(ref)[i]-centers(ref)[j]) for i,j in combinations(range(len(ref)),2)])
 for i,j in combinations(range(len(ref)),2):
  rr=ref[j,:3,:3]@ref[i,:3,:3].T;tr=ref[j,:3,3]-rr@ref[i,:3,3]
  if np.linalg.norm(tr)<max(1e-10,base*1e-6):continue
  rp=pred[j,:3,:3]@pred[i,:3,:3].T;tp=pred[j,:3,3]-rp@pred[i,:3,3]
  rre=float(np.degrees(np.arccos(np.clip((np.trace(rp@rr.T)-1)/2,-1,1))))
  rte=float(np.degrees(np.arccos(np.clip(np.dot(tp,tr)/(np.linalg.norm(tp)*np.linalg.norm(tr)),-1,1)))) if np.linalg.norm(tp)>1e-12 and np.isfinite(tp).all() else 180.
  if not np.isfinite(rre):rre=180.
  rows.append(dict(i=i,j=j,rre=rre,rte=rte,error=max(rre,rte)))
 result={'rotation_orthogonality_defect_before_so3_projection':defects,'pairs':rows,'rre_median':float(np.median([x['rre'] for x in rows])),'rte_median':float(np.median([x['rte'] for x in rows]))}
 # Exact area under empirical CDF = mean(max(1-e/T,0)).
 for t in [5,10,20,30]:result[f'auc_{t}']=float(np.mean([max(1-x['error']/t,0) for x in rows]))
 try:
  pc=centers(pred);qc=centers(ref);sim=fit_sim(pc,qc);singular=np.linalg.svd(qc-qc.mean(0),compute_uv=False)
  result['ate_rmse_sim3']=float(np.sqrt(np.mean(np.sum((transform(pc,sim)-qc)**2,axis=-1))))
  result['ate_reference_second_to_first_singular_ratio']=float(singular[1]/max(singular[0],1e-12))
  result['ate_degenerate']=bool(singular[1]/max(singular[0],1e-12)<1e-4)
 except ValueError:result['ate_rmse_sim3']=None;result['ate_degenerate']=True
 return result

def confidence_keep(data,vgtw=False):
 c=data['depth_conf'];c=c[...,0] if c.ndim==4 else c
 pop=c[np.isfinite(c)]
 if not vgtw and 'sky' in data:
  sky=data['sky']>=.5
  if (~sky & np.isfinite(c)).sum()>10:pop=c[~sky & np.isfinite(c)]
 if not len(pop):return np.zeros_like(c,dtype=bool),None
 threshold=float(np.percentile(pop,50)) if vgtw else float(min(max(1.05,np.percentile(pop,40)),np.percentile(pop,90)))
 keep=np.isfinite(c)&(c>=threshold)&np.isfinite(data['depth'])&(data['depth']>0)
 if vgtw:keep&=(c>1e-5)&(data['distractor_valid_mask']>.5)
 return keep,threshold

def voxel(p,c=None,size=.005):
 good=np.isfinite(p).all(-1);p=p[good];c=c[good] if c is not None else None
 if not len(p):return p,c
 _,ids=np.unique(np.floor(p/size).astype(np.int64),axis=0,return_index=True);ids=np.sort(ids)
 # Deterministic fixed limit applied identically after voxelization.
 if len(ids)>300000:ids=ids[np.linspace(0,len(ids)-1,300000,dtype=int)]
 return p[ids],None if c is None else c[ids]
def cloud_metrics(p,q,thresholds):
 if not len(q):return dict(status='no_reference')
 if not len(p):return dict(status='empty_prediction',accuracy=None,completeness=None,chamfer_l1=None,**{f'fscore_{t:g}':0. for t in thresholds})
 dp=cKDTree(q).query(p,workers=4)[0];dq=cKDTree(p).query(q,workers=4)[0]
 m=dict(accuracy=float(dp.mean()),completeness=float(dq.mean()),chamfer_l1=float((dp.mean()+dq.mean())/2),pred_points=len(p),reference_points=len(q))
 for t in thresholds:
  pr=float(np.mean(dp<t));rc=float(np.mean(dq<t));m.update({f'precision_{t:g}':pr,f'recall_{t:g}':rc,f'fscore_{t:g}':2*pr*rc/(pr+rc) if pr+rc else 0.})
 return m

def write_ply(path,p,c):
 a=np.empty(len(p),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
 for k,n in enumerate(['x','y','z']):a[n]=p[:,k]
 for k,n in enumerate(['red','green','blue']):a[n]=np.uint8(np.clip(c[:,k],0,1)*255)
 with open(path,'wb') as f:
  f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(p))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode());a.tofile(f)

def selftest():
 rng=np.random.default_rng(3);p=rng.normal(size=(100,3));r=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);q=2*p@r.T+[1,2,3];s=fit_sim(p,q);assert np.max(abs(transform(p,s)-q))<1e-10
 m=cloud_metrics(q,q,[.01]);assert m['chamfer_l1']==0 and m['fscore_0.01']==1
 m=cloud_metrics(q[:50],q,[1e-8]);assert m['recall_1e-08']==.5
 e=np.tile(np.eye(4)[:3],(4,1,1));e[:,0,3]=np.arange(4);m=pose_metrics(e,e);assert m['auc_5']==1
 print('metric selftests passed',flush=True)
if __name__=='__main__':selftest()
